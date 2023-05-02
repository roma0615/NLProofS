import os
import openai
import json

openai.api_key = os.environ["OPENAI_API_KEY"]

# Reads the JSON file 'examples.txt' and stores each line as an entry in the list 'examples'
with open("examples.txt") as f:
    examples = f.read().splitlines()

# Parses the list of strings into JSON
examples = [json.loads(example) for example in examples]

context_size=10
messages = [{"role": "system", "content": "You are an automated proof solver that generates the next step in a proof given a hypothesis and previous steps taken."}]

for i in range(context_size):
    messages.append({"role": "user", "content": examples[i]['input_seq']})
    messages.append({"role": "assistant", "content": examples[i]['output_seq']})

messages.append({"role": "user", "content": examples[context_size]['input_seq']})
print(examples[context_size]['input_seq'])
print()

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
)

# Prints the response
print(response["choices"][0]["message"]["content"])