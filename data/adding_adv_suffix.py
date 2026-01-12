
import json
import random

# Predefined suffix list
suffixes = ['DirectiveLob ! ! ! ! ! !', '.ucz.questions ! ! ! ! ! !', 'questions ! ! ! ! ! ', 'Lob ! ! ! ! ! ! ! ! ! ', 'cudaregister ! ! ! ! ! !']

# Load JSON file
with open("/mnt/scratch/users/40645696/LLaMA-Factory/data/claric_sip_train.json", "r") as f:
    data = json.load(f)

# Iterate over all elements
for i in range(len(data)):
    for conv in data[i].get("conversations", []):
        if conv.get("from") == "human":
            # Toss a coin
            if random.choice([True, False]):
                # Pick a random suffix
                suffix = random.choice(suffixes)
                conv["value"] += " " + suffix
                print(conv["value"])

# Save updated JSON back to file
with open("/mnt/scratch/users/40645696/LLaMA-Factory/data/claric_sip_adv_train.json", "w") as f:
    json.dump(data, f, indent=2)

