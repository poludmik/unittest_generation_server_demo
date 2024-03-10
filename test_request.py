import requests
import json

# Base URL of your Flask application
BASE_URL = 'http://127.0.0.1:8080'

url = f'{BASE_URL}/generate_unittest'
headers = {'Content-Type': 'application/json'}

code_chunk_to_test = "def findLength(str):\ncounter = 0   \nfor i in str:\ncounter += 1\nreturn counter"

data = {'code_chunk': code_chunk_to_test} # can specify 'model_name' here as well, e.g. ["gpt-3.5-turbo-instruct", "codellama/CodeLlama-7b-Instruct-hf"]
# Codellama needs enough VRAM, like 20 GB

response = requests.get(url, headers=headers, json=data)

print(response.content) # a string with unittests
