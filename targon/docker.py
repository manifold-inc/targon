from time import sleep
from typing import Any, Dict, List, Tuple
import re
import math
import docker
import bittensor as bt
import subprocess
from accelerate.commands import estimate
from docker.client import DockerClient

from docker.models.containers import Container
from docker.types import DeviceRequest


def get_gpu_with_space(gpus: List[Tuple[int, int, int]], required: int):
    bt.logging.info(f"Need: {required}, have: {gpus}")
    gpus.sort(key=lambda x: x[1])
    for gpu in gpus:
        if gpu[1] >= required * 1.2:
            return gpu
    return None


def bytes_to_mib(bytes_value):
    mib_value = bytes_value / (1024**2)  # 1024^2 = 1,048,576
    return math.ceil(mib_value)


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
    try:
        client.images.pull(MANIFOLD_VERIFIER)  # type: ignore
    except Exception as e:
        bt.logging.error(str(e))
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


def down_containers(client: DockerClient):
    containers: List[Container] = client.containers.list(  # type: ignore
        filters={"ancestor": MANIFOLD_VERIFIER}
    )
    for container in containers:
        container.remove(force=True)


def sync_output_checkers(
    client: docker.DockerClient, models: List[str]
) -> Dict[str, int]:
    bt.logging.info(f"Syncing {models}")
    containers: List[Container] = client.containers.list(  # type: ignore
        filters={"ancestor": MANIFOLD_VERIFIER}
    )
    verification_ports = {}
    existing = []

    # delete any unused containers
    for container in containers:
        bt.logging.info(f"Found {container.name}")
        model = container.labels.get("model")
        if model not in models:
            bt.logging.info(f"Removing {container.name}: {model}")
            container.remove(force=True)
            continue
        verification_ports[model] = int(container.labels.get("port", 0))
        existing.append(model)
    bt.logging.info(f"Existing: {existing}, needed: {models}")
    needed_models = set(models) - set(existing)
    used_ports = list(verification_ports.values())
    min_port = 5555

    # Load all models
    bt.logging.info(f"Starting {list(needed_models)}")
    for model in needed_models:
        container_name = re.sub(r"[\W_]", "-", model).lower()

        # Delete if existing and out of date
        existing_containers: List[Container] = client.containers.list(filters={"name": container_name})  # type: ignore
        if len(existing_containers):
            existing_containers[0].remove(force=True)

        # Determine GPU free
        free_gpus = get_free_gpus()
        required_vram = estimate_max_size(model)
        if required_vram is None:
            bt.logging.error(f"Failed to find model {model}")
            continue
        gpu = get_gpu_with_space(free_gpus, required_vram)
        if gpu is None:
            bt.logging.info(f"Not enough space to run {model}")
            continue

        # Find Port
        while min_port in used_ports:
            min_port += 1
        used_ports.append(min_port)

        memory_util = round((required_vram * 1.2) / gpu[2], 3)

        # Init new container
        bt.logging.info(f"Loading {model} on gpu {gpu[0]} using {memory_util}% vram")
        config: Dict[str, Any] = {
            "image": MANIFOLD_VERIFIER,
            "ports": {f"80/tcp": min_port},
            "environment": [f"MODEL={model}", f"GPU_MEMORY_UTIL={memory_util}"],
            "runtime": "nvidia",
            "detach": True,
            "healthcheck": {
                "test": [
                    "CMD-SHELL",
                    "curl --silent --fail http://localhost/ > /dev/null || exit 1",
                ],
                "interval": int(1e9 * 10),
                "retries": 15,
                "start_period": int(1e9 * 45),
                "start_interval": int(1e9 * 30),
            },
            "auto_remove": True,
            "name": container_name,
            "extra_hosts": {"host.docker.internal": "host-gateway"},
            "labels": {"model": str(model), "port": str(min_port)},
            "device_requests": [
                DeviceRequest(device_ids=[str(gpu[0])], capabilities=[["gpu"]])
            ],
        }
        containers.append(client.containers.run(**config))  # type: ignore
        verification_ports[model] = min_port

    bt.logging.info("Waiting for containers to startup")
    while True:
        ready = True
        models = list(verification_ports.keys())
        for model in models:
            std_model = re.sub(r"[\W_]", "-", model).lower()
            containers: List[Container] = client.containers.list(filters={"name": std_model})  # type: ignore

            if not len(containers):
                bt.logging.info(
                    f"Failed starting container {std_model}: Removing from verifiers"
                )
                verification_ports.pop(model, None)
                continue
            (container,) = containers
            if container.health != "healthy":
                bt.logging.info(f"{container.name}: {container.health}")
                ready = False
        if ready:
            break
        bt.logging.info("Checking again in 5 seconds")
        sleep(5)
    bt.logging.info("Successfully started verifiers")
    if len(list(verification_ports.keys())) == 0:
        bt.logging.error("No verification ports")
        exit()
    return verification_ports
