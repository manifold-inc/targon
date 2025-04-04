# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Manifold Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from enum import Enum
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class Endpoints(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


class NeuronType(Enum):
    Validator = "VALIDATOR"
    Miner = "MINER"


class InferenceStats(BaseModel):
    time_to_first_token: float
    time_for_all_tokens: float
    total_time: float
    tps: float
    tokens: List[Any]
    gpus: int
    verified: bool
    error: Optional[str] = None
    cause: Optional[str] = None


class OrganicStats(InferenceStats):
    model: str
    max_tokens: int
    seed: int
    temperature: float
    uid: int
    hotkey: str
    coldkey: str
    endpoint: str
    total_tokens: int
    pub_id: str


class VerificationPortsConfig(BaseModel):
    api_key: Optional[str] = None
    port: Optional[int] = None
    url: str
    endpoints: List[str]


class MinerEndpoint(BaseModel):
    port: int
    url: str
    qps: int

class ValidatorConfig(BaseModel):
    skip_weight_set: Optional[bool] = False
    verification_ports: Optional[Dict[str, VerificationPortsConfig]] = None
    set_weights_on_start: Optional[bool] = False
    max_concurrent_organics: Optional[int] = 2


class MinerConfig(BaseModel):
    miner_endpoints: Optional[Dict[str, MinerEndpoint]] = None
    miner_api_key: Optional[str] = "1234"
    miner_nodes: Optional[List[str]] = None
    cvm_nodes: Optional[List[str]] = None
