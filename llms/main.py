import os

from litellm import completion

# os.environ["OPENAI_API_KEY"] = "your-openai-key"

messages = [{"content": "Hello, how are you?", "role": "user"}]
response = completion(model="openai/gpt-4o", messages=messages)
print(response)
response = completion(model="ollama/llama3.2", messages=messages)
print(response)
