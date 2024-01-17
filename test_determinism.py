import re
import hashlib
import requests
import tiktoken

input = "me oh my oh my"
url = "http://207.188.6.137:8080/generate"

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


# def math_score(reference, completion):
# 	# Extract all the digits and . from the completion and take only the last one
# 	numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", completion)
# 	print(numbers)
# 	completion_digits = float(numbers[-1])
# 	try:
# 		# Convert the string to a float
# 		completion_digits = float(completion_digits)
# 		# Convert the reference to a float
# 		reference = float(reference)
# 		if completion_digits == reference:
# 			return 1.0
# 		# Compute the difference
# 		diff = abs(reference - completion_digits)/(reference + 1e-6)
# 		# Make sure the difference is between 0 and 1
# 		diff = min(abs(diff), 1)
# 		return 1.0 - diff
# 	except ValueError:
# 		return 0.0


def hashing_function(input):
    hashed_input = hashlib.sha256(input.encode()).hexdigest()
    return hashed_input

def generate(input, seed):
	inputs = {
			"inputs": input,
			"parameters": {
					"max_new_tokens": 28,
					"seed": seed
				}
		}
	
	headers = {
		'Content-Type': 'application/json'
	}

	response = requests.post(url, json=inputs, headers=headers).json()
	# print(response.text)
	output = response["generated_text"]

	return output


generation_1 = generate(input, 342983472920)
generation_2 = generate(input, 342983472920)

print(generation_1)
print(generation_2)

encoding_1 = encoding.encode(generation_1)
encoding_2 = encoding.encode(generation_2)

print(sum(encoding_1))
print(sum(encoding_2))

# print(hashing_function(generation_1))
# print(hashing_function(generation_2))

# print(math_score(generation_1, generation_2))


	
