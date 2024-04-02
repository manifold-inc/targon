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


EPOCH_LENGTH = 360 * (20 * 5) # 5 days of blocks

CHALLENGE_FAILURE_REWARD = -0.01
MONITOR_FAILURE_REWARD = -0.002
INFERENCE_FAILURE_REWARD = -0.05

REQUEST_LIMIT_CHALLENGER = 100_000 # 100k every 360 blocks
REQUEST_LIMIT_GRANDMASTER = 25_000 # 10k every 360 blocks
REQUEST_LIMIT_GOLD = 10_000 # 10k every 360 blocks
REQUEST_LIMIT_GOLD = 10_000 # 10k every 360 blocks
REQUEST_LIMIT_SILVER = 5_000 # 5k every 360 blocks
REQUEST_LIMIT_BRONZE = 500 # 1k every 360 blocks


COSINE_SIMILARITY_THRESHOLD_CHALLENGER = 0.98
COSINE_SIMILARITY_THRESHOLD_GRANDMASTER = 0.95
COSINE_SIMILARITY_THRESHOLD_GOLD = 0.90
COSINE_SIMILARITY_THRESHOLD_SILVER = 0.85
COSINE_SIMILARITY_THRESHOLD_BRONZE = 0.80

# Requirements for each tier. These must be maintained for a prover to remain in that tier.
CHALLENGER_INFERENCE_SUCCESS_RATE = 0.999  # 1/1000 chance of failure
CHALLENGER_CHALLENGE_SUCCESS_RATE = 0.999  # 1/1000 chance of failure
GRANDMASTER_INFERENCE_SUCCESS_RATE = 0.989  # 1/100 chance of failure
GRANDMASTER_CHALLENGE_SUCCESS_RATE = 0.989  # 1/100 chance of failure
GOLD_INFERENCE_SUCCESS_RATE = 0.949  # 1/50 chance of failure
GOLD_CHALLENGE_SUCCESS_RATE = 0.949  # 1/50 chance of failure
SILVER_INFERENCE_SUCCESS_RATE = 0.949  # 1/20 chance of failure
SILVER_CHALLENGE_SUCCESS_RATE = 0.949  # 1/20 chance of failure

CHALLENGER_TIER_REWARD_FACTOR = 1.0  # Get 100% rewards
GRANDMASTER_TIER_REWARD_FACTOR = 0.888  # Get 88.8% rewards
GOLD_TIER_REWARD_FACTOR = 0.777  # Get 77.7% rewards
SILVER_TIER_REWARD_FACTOR = 0.555  # Get 55.5% rewards
BRONZE_TIER_REWARD_FACTOR = 0.444  # Get 44.4% rewards

CHALLENGER_TIER_TOTAL_SUCCESSES = 4_000
GRANDMASTER_TIER_TOTAL_SUCCESSES = 2_000
GOLD_TIER_TOTAL_SUCCESSES = 500  # 50
SILVER_TIER_TOTAL_SUCCESSES = 250

TIER_CONFIG = {
    "Bronze": {
        "success_rate": 0.50, 
        "request_limit": 500, 
        "reward_factor": 0.444,
        "similarity_threshold": 5/16
    },
    "Silver": {
        "success_rate": 0.60, 
        "request_limit": 1000, 
        "reward_factor": 0.555,
        "similarity_threshold": 6/16
    },
    "Gold": {
        "success_rate": 0.70, 
        "request_limit": 5000, 
        "reward_factor": 0.666,
        "similarity_threshold": 7/16
    },
    "Platinum": {
        "success_rate": 0.72, 
        "request_limit": 7500, 
        "reward_factor": 0.777,
        "similarity_threshold": 1/2
    },
    "Diamond": {
        "success_rate": 0.74, 
        "request_limit": 10000, 
        "reward_factor": 0.888,
        "similarity_threshold": 9/16
    },
    "Emerald": {
        "success_rate": 0.78, 
        "request_limit": 12500, 
        "reward_factor": 0.900,
        "similarity_threshold": 10/16
    },
    "Ruby": {
        "success_rate": 0.82, 
        "request_limit": 15000, 
        "reward_factor": 0.920,
        "similarity_threshold": 11/16
    },
    "Jade": {
        "success_rate": 0.88, 
        "request_limit": 17500, 
        "reward_factor": 0.940,
        "similarity_threshold": 12/16
    },
    "Master": {
        "success_rate": 0.92, 
        "request_limit": 20000, 
        "reward_factor": 0.960,
        "similarity_threshold": 13/16
    },
    "Grandmaster": 
    {
        "success_rate": 0.96, 
        "request_limit": 22500, 
        "reward_factor": 0.980,
        "similarity_threshold": 14/16
    },
    "Challenger": {
        "success_rate": 0.99, 
        "request_limit": 25000, 
        "reward_factor": 1.0,
        "similarity_threshold": 15/16
    },
}
