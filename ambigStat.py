import json
from collections import Counter
# Replace 'data.json' with your file path
with open('/mnt/scratch/users/40645696/LLaMA-Factory/data/ambigQA_train_SPLADE3.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
total_elements = len(data)
print("Total elements:", total_elements)


# 2 & 3. Count unique 'gpt' 'value' fields
gpt_values = []
for item in data:
    for convo in item.get("conversations", []):
        if convo.get("from") == "gpt":
            gpt_values.append(convo.get("value"))

# Count occurrences
gpt_value_counts = Counter(gpt_values)
unique_gpt_values = len(gpt_value_counts)

print("Total elements:", total_elements)
print("Number of unique GPT values:", unique_gpt_values)
print("Each GPT value's count:")
for value, count in gpt_value_counts.items():
    print(f"{repr(value)}: {count}")

from itertools import zip_longest

# Load the two JSON files
with open("/mnt/scratch/users/40645696/LLaMA-Factory/data/ambigQA_train_SPLADE3.json", "r", encoding="utf-8") as f1, open("/mnt/scratch/users/40645696/LLaMA-Factory/data/ambigQA_dev.json", "r", encoding="utf-8") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# Compare lengths
len1, len2 = len(data1), len(data2)
print(f"Length of file1: {len1}, Length of file2: {len2}")

# Check if overall structure is similar
similar_length = len1 == len2
print("Files have same length:", similar_length)

# Compare elements at the same positions (up to the length of the shorter list)
sample_diff = []
for i, (elem1, elem2) in enumerate(zip_longest(data1, data2)):
    if elem1 != elem2:
        sample_diff.append((i, elem1, elem2))

# Show a few sample differences
print(f"Total differing elements: {len(sample_diff)}")
print("Sample differing elements (up to 5 shown):")
for i, e1, e2 in sample_diff[:2]:
    print(f"Index {i}:\nFile1: {e1}\nFile2: {e2}\n")
