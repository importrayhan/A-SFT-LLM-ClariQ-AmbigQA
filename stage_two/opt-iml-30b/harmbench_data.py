import requests
import pandas as pd
from datasets import load_dataset
import json
import io

def save_instructions_to_json(instructions, filename):
    """Saves a list of instructions to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=4)
    print(f"Saved instructions to {filename}")


url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
response = requests.get(url)

dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
instructions = dataset['goal'].tolist()

save_instructions_to_json(instructions, "/mnt/scratch/users/40645696/opt-iml-30b/advbench.json")




hf_path = 'tatsu-lab/alpaca'
dataset = load_dataset(hf_path)

# filter for instructions that do not have inputs
instructions2 = []
for i in range(len(dataset['train'])):
    if dataset['train'][i]['input'].strip() == '':
        instructions2.append(dataset['train'][i]['instruction'])
save_instructions_to_json(instructions2, "/mnt/scratch/users/40645696/opt-iml-30b/alpaca.json")

