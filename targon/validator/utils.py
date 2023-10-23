
# Utils for checkpointing and saving the model.
import os
import sys
import torch
import wandb
import copy
import bittensor as bt
import targon
import requests
import time
import math
import hashlib as rpccheckhealth
from math import floor
from typing import Callable, Any
from functools import lru_cache, update_wrapper
from targon import __version__

def autoupdate():
    '''
    Updates the targon codebase if a newer version is available.
    '''
    # try:
    bt.logging.trace("Checking for updates...")
    response = requests.get(
        "https://raw.githubusercontent.com/manifold-inc/targon/main/VERSION"
    )
    response.raise_for_status()
    # try:
    repo_version = response.content.decode()
    
    latest_version = [int(v) for v in repo_version.split(".")]
    bt.logging.debug(f"Current version: {__version__}")
    bt.logging.debug(f"Latest version: {latest_version}")
    if latest_version > __version__:
        bt.logging.trace("A newer version of Targon is available. Downloading...")
        # download latest version with git pull

        # step backwards in the path until we find targon
        while os.path.basename(base_path) != "targon":
            base_path = os.path.dirname(base_path)
        

        os.system(f"cd {base_path} && git pull")
        # checking local VERSION
        with open(base_path, "VERSION") as f:
            new__version__ = f.read().strip()
            # convert to list of ints
            new__version__ = [int(v) for v in new__version__.split(".")]
            if new__version__ == latest_version:
                bt.logging.debug("Targon updated successfully.")
                bt.logging.debug("Restarting...")
                bt.logging.debug(f"Running: {sys.executable} {sys.argv}")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                bt.logging.error("Targon git pull failed you will need to manually update and restart for latest code.")
    #     except Exception as e:
    #         bt.logging.error("Failed to convert response to json: {}".format(e))
    #         bt.logging.trace("Response: {}".format(response.text))            
    # except Exception as e:
    #     bt.logging.error("Failed to check for updates: {}".format(e))


# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper


def _ttl_hash_gen(seconds: int):
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    return self.subtensor.get_current_block()

def should_reinit_wandb(self):
    # Check if wandb run needs to be rolled over.
    return not self.config.wandb.off and self.step and self.step % self.config.wandb.run_step_length == 0


def init_wandb(self, reinit=False):
    """Starts a new wandb run."""
    tags = [self.wallet.hotkey.ss58_address,
            targon.__version__,
            str(targon.__spec_version__),
            f'netuid_{self.metagraph.netuid}']

    if self.config.mock:
        tags.append("mock")
    if self.config.neuron.use_custom_gating_model:
        tags.append("custom_gating_model")
    if self.config.neuron.disable_set_weights:
        tags.append("disable_set_weights")
    if self.config.neuron.disable_log_rewards:
        tags.append("disable_log_rewards")

    wandb_config = {key: copy.deepcopy(self.config.get(key, None)) for key in ('neuron', 'reward', 'netuid', 'wandb')}
    wandb_config['neuron'].pop('full_path', None)

    self.wandb = wandb.init(
        anonymous="allow",
        reinit=reinit,
        project=self.config.wandb.project_name,
        entity=self.config.wandb.entity,
        config=wandb_config,
        mode="offline" if self.config.wandb.offline else "online",
        dir=self.config.neuron.full_path,
        tags=tags,
        notes=self.config.wandb.notes,
    )
    bt.logging.success(
        prefix="Started a new wandb run",
        sufix=f"<blue> {self.wandb.name} </blue>",
    )


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    self.wandb.finish()
    init_wandb(self, reinit=True)


def should_checkpoint(self):
    # Check if enough epoch blocks have elapsed since the last checkpoint.
    return (
        ttl_get_block(self) % self.config.neuron.checkpoint_block_length
        < self.prev_block % self.config.neuron.checkpoint_block_length
    )


def checkpoint(self):
    """Checkpoints the training process."""
    bt.logging.info("checkpoint()")
    resync_metagraph(self)
    save_state(self)


def resync_metagraph(self: 'targon.neuron.neuron'):
    """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
    bt.logging.info("resync_metagraph()")

    # Copies state of metagraph before syncing.
    previous_metagraph = copy.deepcopy(self.metagraph)

    # Sync the metagraph.
    self.metagraph.sync(subtensor=self.subtensor)

    # Check if the metagraph axon info has changed.
    metagraph_axon_info_updated = previous_metagraph.axons != self.metagraph.axons

    if metagraph_axon_info_updated:
        bt.logging.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
        # Reconstruct the dendrite pool with the new endpoints.
        self.dendrite_pool.resync(self.metagraph)

        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.moving_averaged_scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            new_moving_average[: len(self.hotkeys)] = self.moving_averaged_scores
            self.moving_averaged_scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
    
    


def resync_linear_layer(
    linear_layer: torch.nn.Module,
    previous_metagraph: "bt.metagraph.Metagraph",
    metagraph: "bt.metagraph.Metagraph",
):
    """Resync the linear layer with the latest state of the network
    Args:
         linear_layer (:obj: torch.nn.Module): Linear layer to be resynced
         previous_metagraph (:obj: bt.metagraph.Metagraph):
             Previous state of metagraph before updated resync
         metagraph (:obj: bt.metagraph.Metagraph):
             Latest state of the metagraph with updated uids and hotkeys
    """
    uids_hotkeys_state_dict = dict(zip(previous_metagraph.uids.tolist(), previous_metagraph.hotkeys))
    latest_uids_hotkeys_state_dict = dict(zip(metagraph.uids.tolist(), metagraph.hotkeys))

    updated_uids_indices = []
    for uid, latest_hotkey in latest_uids_hotkeys_state_dict.items():
        if uids_hotkeys_state_dict.get(uid) != latest_hotkey:
            updated_uids_indices.append(uid)

    for index in updated_uids_indices:
        # Reinitialize the bias of the selected index of the linear layer
        torch.nn.init.zeros_(linear_layer.bias[index])
        # Clone the weights of the selected index of the linear layer
        weights = linear_layer.weight[index].clone()
        # Adds a dimension to the weights tensor to make it compatible with the xavier_uniform_ function
        torch.nn.init.xavier_uniform_(weights.unsqueeze(0))
        reinitialized_weights = weights.squeeze(0)
        # Copy the reinitialized weights back to the selected index of the linear layer
        linear_layer.weight[index].data.copy_(reinitialized_weights)


def check_uid_availability(metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def save_state(self):
    r"""Save hotkeys, gating model, neuron model and moving average scores to filesystem."""
    bt.logging.info("save_state()")
    try:
        neuron_state_dict = {
            "neuron_weights": self.moving_averaged_scores.to('cpu').tolist(),
            "neuron_hotkeys": self.hotkeys,
        }
        torch.save(neuron_state_dict, f"{self.config.neuron.full_path}/model.torch")
        bt.logging.success(
            prefix="Saved model",
            sufix=f"<blue>{ self.config.neuron.full_path }/model.torch</blue>",
        )
    except Exception as e:
        bt.logging.warning(f"Failed to save model with error: {e}")

    try:
        # Save the gating model.
        gating_model_linear_layer_dict = self.gating_model.linear.state_dict()
        gating_model_name = self.config.gating.model_name.replace("/", "_")
        gating_model_file_path = f"{self.config.neuron.full_path}/{gating_model_name}_gating_linear_layer.pth"
        torch.save(gating_model_linear_layer_dict, gating_model_file_path)

        if not self.config.wandb.off:
            wandb.log({
                "step": self.step,
                "block": ttl_get_block(self),
                **neuron_state_dict
            })
        if not self.config.wandb.off and self.config.wandb.track_gating_model:
            model_artifact = wandb.Artifact(f"{gating_model_name}_gating_linear_layer", type="model")
            model_artifact.add_file(gating_model_file_path)
            self.wandb.log_artifact(model_artifact)

        bt.logging.success(prefix="Saved gating model", sufix=f"<blue>{gating_model_file_path}</blue>")
    except Exception as e:
        bt.logging.warning(f"Failed to save gating model with error: {e}")

    try:
        # Save diversity model.
        diversity_model_dict = {"historic_embeddings": self.diversity_model.historic_embeddings.to('cpu')}
        diversity_model_file_path = f"{self.config.neuron.full_path}/diversity_model.pth"
        torch.save(diversity_model_dict, diversity_model_file_path)
        bt.logging.success(
            prefix="Saved diversity model",
            sufix=f"<blue>{diversity_model_file_path}</blue> {list(self.diversity_model.historic_embeddings.shape)}",
        )
    except Exception as e:
        bt.logging.warning(f"Failed to save diversity model with error: {e}")

    # empty cache
    torch.cuda.empty_cache()


def load_state(self):
    r"""Load hotkeys and moving average scores from filesystem."""
    bt.logging.info("load_state()")
    try:
        state_dict = torch.load(f"{self.config.neuron.full_path}/model.torch")
        # Check for nans in saved state dict
        neuron_weights = torch.tensor(state_dict["neuron_weights"])
        if not torch.isnan(neuron_weights).any():
            self.moving_averaged_scores = neuron_weights.to(self.device)
        self.hotkeys = state_dict["neuron_hotkeys"]
        bt.logging.success(
            prefix="Reloaded model",
            sufix=f"<blue>{ self.config.neuron.full_path }/model.torch</blue>",
        )
    except Exception as e:
        bt.logging.warning(f"Failed to load model with error: {e}")

    try:
        # Load diversity model.
        diversity_model_file_path = f"{self.config.neuron.full_path}/diversity_model.pth"
        diversity_model_dict = torch.load(diversity_model_file_path)
        self.diversity_model.historic_embeddings = diversity_model_dict["historic_embeddings"].to(self.device)
        bt.logging.success(
            prefix="Reloaded diversity model",
            sufix=f"<blue>{diversity_model_file_path}</blue> {list(self.diversity_model.historic_embeddings.shape)}",
        )
    except Exception as e:
        bt.logging.warning(f"Failed to load diversity model with error: {e}")