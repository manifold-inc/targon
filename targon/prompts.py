import re
import random

tasks = [
    {
    'name': 'coding',
    'flavor': ['python', 'javascript'],
    },
    {
    'name': 'qa',
    'flavor': ['math', 'open_orca'],
    },
    {
    'name': 'reasoning',
    'flavor': ['reasoning'],
    },
]


math_qa_prompt = '''{problem}
Multiple Choice Options:
{options}

provide a rationale for your answer in the following format - "explanation" : explanation text : "option" letter
'''

javascript_prompt = '''{code}

Provide a detailed explanation of the code above.
'''

reasoning_prompt = '''{instruction}
{input}
'''

def process_reasoning( data ):
    instruction = data['instruction'] if data['instruction'] != '' else None
    input = data['input'] if data['input'] != '' else None
    output = data['output'] if data['output'] != '' else None

    return instruction, input, output

def process_python( data ):
    instruction = data['INSTRUCTION'] if data['INSTRUCTION'] != '' else None
    response = data['RESPONSE'] if data['RESPONSE'] != '' else None
    return instruction, response

def process_javascript( data ):
    code = data['code'] if data['code'] != '' else None
    docstring = data['docstring'] if data['docstring'] != '' else None

    return code, docstring

def process_open_orca( data ):
    system_prompt = data['system_prompt'] if data['system_prompt'] != '' or data['system_prompt'] != None else None
    question = data['question'] if data['question'] != '' else None
    response = data['response'] if data['response'] != '' else None

    return system_prompt, question, response

def process_math_qa( data ):
    problem = data['Problem'] if data['Problem'] != '' else None
    options = data['options'] if data['options'] != '' else None
    rationale = data['Rationale'] if data['Rationale'] != '' else None
    correct_option = data['correct_option'] if data['correct_option'] != '' else None

    return problem, options, rationale, correct_option
