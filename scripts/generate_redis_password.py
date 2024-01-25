import string
import random

def generate_strong_password(length=12):
    all_characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(all_characters) for i in range(length))
    return password



if __name__ == "__main__":
    print(generate_strong_password())