# 4.3.0

- Model Rotation
  - Validators can now host any number of models. These are rotated out each
    interval randomly. Valis are garunteed to send atleast 1 verifiable request
    every 3 ticks. Validators are now only limited to the largest model they can
    run, not the number of models.
- Dask -> Datasets
  - Datasets provides huggingface caching for the synthetic dataset. Vali
    startup time after warmup reduced from ~5 min to \< 5 seconds. (not
    including model rotation time)
- Heartbeat
  - Validators now self-monitor for system hangs. If no requests are
    successfully sent within 5 mintues, validators will auto-restart. This is
    opt-in, and can be enabled with an env file. See readme for details, under
    the `Validator .env` header.
- Bittensor 8.4.5
  - Bumped bittensor version to help with stability when connecting to the chain
    and getting metagraph / posting weights
