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

from datetime import datetime

def create_prompt( challenge_data ):
    """
    Creates a prompt for the prover to respond to.

    Parameters:
    - challenge_data (protocol.Challenge): The challenge data sent to the prover.

    Returns:
    - str: The prompt for the prover to respond to.
    """

    output = '''
### Current Date: {date}
### Instruction: 
You are an expert language model that is tasked with performing a search over the given query and search results.
your answer should be short, two paragraphs exactly, and should be relevant to the query and search results.

Search Results:
{sources}

Query:
{query}

Response:
'''.format(date=(datetime.now().strftime('%Y-%m-%d')), query=challenge_data['query'], sources='\n'.join(challenge_data['sources'])) 

    return output