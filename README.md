# TARGON: A Redundant Deterministic Verification of Large Language Models

TARGON (Bittensor Subnet 4) is a redundant deterministic verification mechanism
that can be used to interpret and analyze ground truth sources and a query.

NOTICE: Using this software, you must agree to the Terms and Agreements provided
in the terms and conditions document. By downloading and running this software,
you implicitly agree to these terms and conditions.

```
Miner:
py neurons/miner.py --wallet.name [your coldkey] --netuid [40/4] --wallet.hotkey [your hotkey] --subtensor.network [test/finney] --neuron.model_endpoint [your model endpoint] --nueron.api_key 1234

Validator:
py neurons/validator.py --wallet.name [your coldkey] --netuid [40/4] --wallet.hotkey [your hotkey] --subtensor.network [test/finney] --neuron.model_endpoint [your model endpoint] --nueron.api_key 1234
```
