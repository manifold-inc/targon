from time import sleep
import random
from typing import Any, Dict, List, Optional, Tuple
import re
import math
import docker
import bittensor as bt
import subprocess
from accelerate.commands import estimate

from docker.models.containers import Container
from docker.types import DeviceRequest
import requests

from targon.config import IMAGE_TAG
from targon.types import Config, Endpoints
from targon.utils import (
    fail_with_none,
)
from targon.request import get_tool_parser_for_model


def get_gpu_with_space(gpus: List[Tuple[int, int, int]], required: int):
    "[GPU_ID, free, total] in MB"
    bt.logging.info(f"Need: {required}, have: {gpus}")

    # find unsused GPUS
    unused = [gpu for gpu in gpus if gpu[1] / gpu[2] > 0.9]

    # find first gpu with enough space
    for gpu in unused:
        if gpu[1] >= required * 1.2:
            return [gpu]

    # if we need multiple gpu, only used unused
    total_free = 0
    next_gpus = []
    for gpu in unused:
        total_free += gpu[1]
        next_gpus.append(gpu)
        if total_free > required * 1.2:
            return next_gpus
    return None


def bytes_to_mib(bytes_value):
    mib_value = bytes_value / (1024**2)  # 1024^2 = 1,048,576
    return math.ceil(mib_value)


@fail_with_none("Failed estimating max size")
def estimate_max_size(model_name):
    "Returns size in MiB, what nvidia smi prints"
    try:
        model = estimate.create_empty_model(
            model_name, library_name="transformers", trust_remote_code=False
        )
    except (RuntimeError, OSError) as e:
        library = estimate.check_has_model(e)
        if library != "unknown":
            raise RuntimeError(
                f"Tried to load `{model_name}` with `{library}` but a possible model to load was not found inside the repo."
            )
        return None

    total_size, _ = estimate.calculate_maximum_sizes(model)
    return bytes_to_mib(total_size)


MANIFOLD_VERIFIER = "manifoldlabs/sn4-verifier"


def load_docker():
    client = docker.from_env()
    return client


def get_free_gpus() -> List[Tuple[int, int, int]]:
    "[GPU_ID, free, total] in MB"
    res = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if res.returncode != 0:
        bt.logging.error(res.stdout.decode("utf-8"))
        raise Exception("Failed to detect nvida gpus")

    lines = [line.split(" ") for line in res.stdout.decode("utf-8").strip().split("\n")]
    gpus = [(i, int(line[0]), int(line[2])) for i, line in enumerate(lines)]
    return gpus


def remove_containers(client):
    containers: List[Container] = client.containers.list(  # type: ignore
        filters={"label": "model"}
    )
    for container in containers:
        model = container.labels.get("model")
        bt.logging.info(f"Removing {container.name}: {model}")
        container.remove(force=True)


def sync_output_checkers(
    client: docker.DockerClient, models: List[str], config: Optional[Config]
) -> Dict[str, Dict[str, Any]]:
    # Get new image hash (if any)
    image_name = f"{MANIFOLD_VERIFIER}:{IMAGE_TAG}"
    try:
        client.images.pull(image_name)  # type: ignore
    except Exception as e:
        bt.logging.error(str(e))
    bt.logging.info(f"Syncing {models}")

    # Remove all containers
    remove_containers(client)
    verification_ports = {}
    used_ports = []
    random.shuffle(models)
    min_port = 5555

    # Clear containers that arent running
    client.containers.prune()

    # Load all models
    bt.logging.info(f"Starting subset of {list(models)}")
    for model in models:
        container_name = re.sub(r"[\W_]", "-", model).lower()

        # Delete if existing and out of date
        existing_containers: List[Container] = client.containers.list(filters={"name": container_name})  # type: ignore
        if len(existing_containers):
            existing_containers[0].remove(force=True)

        # Determine GPU free
        free_gpus = get_free_gpus()
        required_vram = estimate_max_size(model)
        if required_vram is None:
            bt.logging.error(f"Failed to find or load model {model}")
            continue
        gpus = get_gpu_with_space(free_gpus, required_vram)
        if gpus is None:
            bt.logging.info(f"Not enough space to run {model}")
            continue

        # Find Port
        while min_port in used_ports:
            min_port += 1
        used_ports.append(min_port)

        env_vars = [
            f"MODEL={model}",
            f"TENSOR_PARALLEL={len(gpus)}",
        ]

        tool_call_parser = get_tool_parser_for_model(model)

        if tool_call_parser:
            bt.logging.info(
                f"Enabling tool calling for {model} with parser {tool_call_parser}"
            )
            env_vars.extend(
                [
                    f"TOOL_CALL_PARSER={tool_call_parser}",
                    "ENABLE_AUTO_TOOL_CHOICE=true",
                ]
            )

        # Init new container
        bt.logging.info(f"Loading {model} on gpu(s) {[gpu[0] for gpu in gpus]}")
        docker_config: Dict[str, Any] = {
            "image": image_name,
            "ports": {f"80/tcp": min_port},
            "environment": env_vars,
            "volumes": ["/var/targon/huggingface/cache:/root/.cache/huggingface"],
            "runtime": "nvidia",
            "detach": True,
            "ipc_mode": "host",
            "name": container_name,
            "extra_hosts": {"host.docker.internal": "host-gateway"},
            "labels": {"model": str(model), "port": str(min_port)},
            "device_requests": [
                DeviceRequest(
                    device_ids=[str(gpu[0]) for gpu in gpus], capabilities=[["gpu"]]
                )
            ],
        }
        client.containers.run(**docker_config)  # type: ignore
        while True:
            ready = True
            std_model = re.sub(r"[\W_]", "-", model).lower()
            containers: List[Container] = client.containers.list(filters={"name": std_model}, all=True)  # type: ignore
            if not len(containers):
                bt.logging.info(
                    f"Failed starting container {std_model}: Removing from verifiers"
                )
                break
            (container,) = containers
            if container.health == "unhealthy":
                container_logs = container.logs()
                bt.logging.error(
                    f"Failed starting container {std_model}: Removing from verifiers"
                )
                bt.logging.error("---- Verifier Logs ----")
                bt.logging.error(str(container_logs))
                bt.logging.error("-----------------------")
                break
            if container.health != "healthy":
                bt.logging.info(f"{container.name}: {container.health}")
                ready = False
            if ready:
                verification_ports[model] = {"port": min_port}
                metadata = requests.get(f"http://localhost:{min_port}/metadata").json()
                endpoints = [Endpoints(e.upper()) for e in metadata["endpoints"]]
                verification_ports[model]["endpoints"] = endpoints
                verification_ports[model]["url"] = "http://localhost"
                verification_ports[model]["max_model_len"] = metadata.get(
                    "max_model_len", 2048
                )
                break
            bt.logging.info("Checking again in 5 seconds")
            sleep(5)

    if config and config.verification_ports:
        extra_ports = {}
        for k, v in config.verification_ports.items():
            extra_ports[k] = {
                "url": v.url,
                "port": v.port,
                "endpoints": [Endpoints(e.upper()) for e in v.endpoints],
            }
        verification_ports = verification_ports | extra_ports

    bt.logging.info("Successfully started verifiers")
    bt.logging.info(str(verification_ports))
    if len(list(verification_ports.keys())) == 0:
        bt.logging.error("No verification ports")
        exit()
    return verification_ports
