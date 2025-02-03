import os
from datasets import load_dataset
import logging
import random
from datetime import datetime
from typing import Dict, Iterable, Union
from openai.types.chat import ChatCompletionMessageParam

from targon.types import Endpoints

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NAMES = [line.strip() for line in open(os.path.join(BASE_DIR, "names.txt")).readlines()]
COUNTRIES = [
    line.strip() for line in open(os.path.join(BASE_DIR, "countries.txt")).readlines()
]


def create_search_prompt(
    query: str, endpoint: Endpoints
) -> Dict[str, Union[str, Iterable[ChatCompletionMessageParam]]]:
    # Format the current date for inclusion in the prompt
    date = datetime.now().strftime("%Y-%m-%d")
    system_message = f"""
### Current Date: {date}
### Instruction:
You are to take on the role of {random.choice(NAMES)}, an expert language model
developed in {random.choice(COUNTRIES)}, tasked with generating responses to user queries.
Your answer should be relevant to the query, and you must start all responses
by briefly introducing yourself, re-stating the query in your own words from 
your perspective ensuring you include today's date (which was provided above),
then provide the response to the query. You should always respond in English.
"""
    # Compile the chat components into a structured format
    match endpoint:
        case Endpoints.CHAT:
            messages: Iterable[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ]
            return {"messages": messages}
        case Endpoints.COMPLETION:
            prompt: str = f"""{system_message}\n\n{query}"""
            return {"prompt": prompt}
        case _:
            raise Exception("Unknown Endpoint")


def create_query_prompt(query: str) -> Iterable[ChatCompletionMessageParam]:
    # Format the current date for inclusion in the prompt
    date = datetime.now().strftime("%Y-%m-%d")

    # Construct the system message with dynamic content
    system_message = f"""
### Current Date: {date}
### Instruction:
You are to take the query information that is passed from you and create a search query for the query data. 
Do not answer the information, just create a search query. The search query should not be longer than a sentence.
Assistant should always start the response with "Search query: "
"""

    # Compile the chat components into a structured format
    chats: Iterable[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    # Apply the chat template without tokenization
    return chats


def download_dataset():
    logger = logging.getLogger("huggingface_hub.utils._http")
    logger.setLevel(logging.CRITICAL + 1)
    ds = load_dataset("manifoldlabs/Infinity-Instruct", "7M")
    return ds

def download_tool_dataset():
    """Hardcoded OpenAI-compliant tools. Will migrate to a more flexible dataset in the future."""
    logger = logging.getLogger("huggingface_hub.utils._http")
    logger.setLevel(logging.CRITICAL + 1)
    
    # Return hardcoded OpenAI-compliant tools
    return {
        "train": [
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather information for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City and country e.g. Paris, France"
                                    }
                                },
                                "required": ["location"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "search_restaurants",
                            "description": "Search for restaurants in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City or area to search in"
                                    },
                                    "cuisine": {
                                        "type": "string",
                                        "description": "Type of cuisine"
                                    }
                                },
                                "required": ["location"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "create_reminder",
                            "description": "Create a reminder for a specific date and time",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "description": "Title of the reminder"
                                    },
                                    "date": {
                                        "type": "string",
                                        "description": "Date for the reminder (YYYY-MM-DD)"
                                    },
                                    "time": {
                                        "type": "string",
                                        "description": "Time for the reminder (HH:MM)"
                                    }
                                },
                                "required": ["title", "date"]
                            }
                        }
                    }
                ],
                "question": "What's the weather like in Paris today?"
            },
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "send_message",
                            "description": "Send a message to a specified recipient",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "recipient": {
                                        "type": "string",
                                        "description": "Name or identifier of recipient"
                                    },
                                    "message": {
                                        "type": "string",
                                        "description": "Content of the message"
                                    }
                                },
                                "required": ["recipient", "message"]
                            }
                        }
                    }
                ],
                "question": "Send a message to John saying I'll be late"
            },
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "schedule_meeting",
                            "description": "Schedule a meeting in the calendar",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "description": "Title of the meeting"
                                    },
                                    "date": {
                                        "type": "string",
                                        "description": "Date of the meeting (YYYY-MM-DD)"
                                    },
                                    "start_time": {
                                        "type": "string",
                                        "description": "Start time (HH:MM)"
                                    },
                                    "duration": {
                                        "type": "integer",
                                        "description": "Duration in minutes"
                                    },
                                    "attendees": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of attendee email addresses"
                                    }
                                },
                                "required": ["title", "date", "start_time"]
                            }
                        }
                    }
                ],
                "question": "Schedule a team meeting for tomorrow at 2pm"
            },
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "translate_text",
                            "description": "Translate text between languages",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "Text to translate"
                                    },
                                    "source_language": {
                                        "type": "string",
                                        "description": "Source language code (e.g., 'en', 'es', 'fr')"
                                    },
                                    "target_language": {
                                        "type": "string",
                                        "description": "Target language code (e.g., 'en', 'es', 'fr')"
                                    }
                                },
                                "required": ["text", "target_language"]
                            }
                        }
                    }
                ],
                "question": "Translate 'Hello, how are you?' to Spanish"
            },
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "description": "Perform mathematical calculations",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "expression": {
                                        "type": "string",
                                        "description": "Mathematical expression to evaluate"
                                    },
                                    "precision": {
                                        "type": "integer",
                                        "description": "Number of decimal places for the result"
                                    }
                                },
                                "required": ["expression"]
                            }
                        }
                    }
                ],
                "question": "Calculate 15% of 85.50"
            },
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "manage_file",
                            "description": "Perform file operations like create, read, write, or delete",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "operation": {
                                        "type": "string",
                                        "enum": ["create", "read", "write", "delete"],
                                        "description": "Type of file operation"
                                    },
                                    "file_path": {
                                        "type": "string",
                                        "description": "Path to the file"
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content to write (for create/write operations)"
                                    }
                                },
                                "required": ["operation", "file_path"]
                            }
                        }
                    }
                ],
                "question": "Create a new file called notes.txt"
            },
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "send_email",
                            "description": "Send an email to specified recipients",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "to": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of recipient email addresses"
                                    },
                                    "subject": {
                                        "type": "string",
                                        "description": "Email subject"
                                    },
                                    "body": {
                                        "type": "string",
                                        "description": "Email body content"
                                    },
                                    "cc": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of CC recipient email addresses"
                                    },
                                    "attachments": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of file paths to attach"
                                    }
                                },
                                "required": ["to", "subject", "body"]
                            }
                        }
                    }
                ],
                "question": "Send an email to team@company.com about the project update"
            },
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "convert_units",
                            "description": "Convert between different units of measurement",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "value": {
                                        "type": "number",
                                        "description": "Value to convert"
                                    },
                                    "from_unit": {
                                        "type": "string",
                                        "description": "Source unit (e.g., 'km', 'mi', 'kg', 'lb')"
                                    },
                                    "to_unit": {
                                        "type": "string",
                                        "description": "Target unit (e.g., 'km', 'mi', 'kg', 'lb')"
                                    }
                                },
                                "required": ["value", "from_unit", "to_unit"]
                            }
                        }
                    }
                ],
                "question": "Convert 5 kilometers to miles"
            }
        ]
    }
