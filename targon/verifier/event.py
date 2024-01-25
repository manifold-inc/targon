# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 philanthrope <-- main author
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

import bittensor as bt
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class EventSchema:
    task_name: str  # Task type, e.g. 'store', 'challenge', 'retrieve' 'broadcast'
    successful: List[bool]  # List of whether or not the task was successful or not
    completion_times: List[float]  # List of completion times for a given task
    task_status_messages: List[
        str
    ]  # List of completion status messages for a given prompt
    task_status_codes: List[str]  # List of completion status codes for a given prompt
    block: float  # Current block at given step
    uids: List[int]  # Queried uids
    step_length: float  # Elapsed time between the beginning of a run step to the end of a run step
    best_uid: str  # Best completion for given task
    best_hotkey: str  # Best hotkey for given task

    # Reward data
    rewards: List[float]  # Reward vector for given step

    # Weights data and moving averages
    set_weights: Optional[List[List[float]]] = None
    moving_averaged_scores: Optional[List[float]] = None

    @staticmethod
    def from_dict(event_dict: dict) -> "EventSchema":
        """Converts a dictionary to an EventSchema object."""

        return EventSchema(
            task_name=event_dict["task_name"],
            successful=event_dict["successful"],
            completion_times=event_dict["completion_times"],
            task_status_messages=event_dict["task_status_messages"],
            task_status_codes=event_dict["task_status_codes"],
            block=event_dict["block"],
            uids=event_dict["uids"],
            step_length=event_dict["step_length"],
            best_uid=event_dict["best_uid"],
            best_hotkey=event_dict["best_hotkey"],
            rewards=event_dict["rewards"],
            set_weights=event_dict["set_weights"],
            moving_averaged_scores=event_dict["moving_averaged_scores"],
        )