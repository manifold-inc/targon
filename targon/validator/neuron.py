import copy
import torch
import asyncio
import bittensor as bt
from targon.validator import AsyncDendritePool, check_config, add_args, config, run, init_wandb, ttl_get_block
from targon.validator.dataset import CodingDataset, QADataset, ReasoningDataset
from targon.protocol import TargonDendrite
from targon.validator.signals import (
    NSFWRewardModel,
    AccuracyRewardSignal,
    CorrectnessRewardSignal,
    RelevanceRewardModel,
    DiversityRewardModel,
    MockRewardModel,
    RewardModelType,
)

class neuron:
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    def run(self):
        run(self)


    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"


    def __init__(self):
        # config
        self.config = neuron.config()
        self.check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        print(self.config)
        bt.logging.info("neuron.__init__()")

        # Init device.
        bt.logging.debug("loading", "device")
        self.device = torch.device(self.config.neuron.device)
        bt.logging.debug(str(self.device))


        # Init subtensor
        bt.logging.debug("loading", "subtensor")
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.debug(str(self.subtensor))

        # Init wallet.
        bt.logging.debug("loading", "wallet")
        self.wallet = bt.wallet(config=self.config)
        self.wallet.create_if_non_existent()
        if not self.config.wallet._mock:
            if not self.subtensor.is_hotkey_registered_on_subnet(hotkey_ss58=self.wallet.hotkey.ss58_address, netuid=self.config.netuid):
                raise Exception(f'Wallet not currently registered on netuid {self.config.netuid}, please first register wallet before running')
                
        bt.logging.debug(str(self.wallet))

        # Init metagraph.
        bt.logging.debug("loading", "metagraph")
        self.metagraph = bt.metagraph(netuid=self.config.netuid, network=self.subtensor.network, sync=False) # Make sure not to sync without passing subtensor
        self.metagraph.sync(subtensor=self.subtensor) # Sync metagraph with subtensor.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        bt.logging.debug(str(self.metagraph))

        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to(self.device)
        bt.logging.debug(str(self.moving_averaged_scores))


        bt.logging.debug("loading", "dataset")
        self.coding_dataset = CodingDataset()
        self.qa_dataset = QADataset()
        self.reasoning_dataset = ReasoningDataset()
        bt.logging.debug(str(self.dataset))


        if not self.config.neuron.axon_off:
            bt.logging.debug('serving ip to chain...')
            try:
                axon = bt.axon( 
                    wallet=self.wallet, metagraph=self.metagraph, config=self.config 
                )

                try:
                    self.subtensor.serve_axon(
                        netuid=self.config.netuid,
                        axon=axon,
                        use_upnpc=False,
                        wait_for_finalization=True,
                    )
                except Exception as e:
                    bt.logging.error(f'Failed to serve Axon with exception: {e}')
                    pass

                del axon
            except Exception as e:
                bt.logging.error(f'Failed to create Axon initialize with exception: {e}')
                pass

        else:
            bt.logging.debug('axon off, not serving ip to chain.')

        self.reward_weights = torch.tensor(
            [
                self.config.reward.dpo_weight
            ],
            dtype=torch.float32,
        ).to(self.device)

        self.reward_functions = [
            AccuracyRewardSignal(device=self.device),
            CorrectnessRewardSignal(device=self.device),

        ]

        relevance_model = (
            RelevanceRewardModel(device=self.device) if not self.config.neuron.relevance_off
            else MockRewardModel(RewardModelType.relevance.value)
        )

        # TODO: Retrain diversity model
        # self.diversity_model = (
        #     DiversityRewardModel(device=self.device) if not self.config.neuron.diversity_off
        #     else MockRewardModel(RewardModelType.diversity.value)
        # )
        nsfw_model = (
            NSFWRewardModel(device=self.device) if not self.config.neuron.nsfw_off
            else MockRewardModel(RewardModelType.nsfw.value)              
        )

        self.masking_functions = [relevance_model, nsfw_model]


        self.dendrite_pool = AsyncDendritePool(wallet=self.wallet, metagraph=self.metagraph)

        # Init the event loop.
        self.loop = asyncio.get_event_loop()

        self.config.neuron.epoch_length = 300
        self.prev_block = ttl_get_block(self)
        self.step = 0


def main():
    neuron().run()


if __name__ == "__main__":
    main()