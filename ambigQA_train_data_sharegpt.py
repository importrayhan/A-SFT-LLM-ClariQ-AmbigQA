import json
from collections import Counter

def transform_json_structure(input_data):
    """
    Transform the input JSON structure to the target format according to specifications.

    Args:
        input_data: List of dictionaries with the input structure

    Returns:
        List of dictionaries with the transformed structure
    """

    # Define the constant system message
    SYSTEM_MESSAGE = "You are a search engine performance predictor. A user is looking for something which is often not clearly understood. Your task is to decide if a search request is oscure. You have a subset of context which you can use to accomplish this task. Given the user request, do you think you have a clearly specified search query to find exactly one answer with no contradiction? Now think of a four level indicator ranging from 1 green, 2 yellow, 3 orange, 4 red. Green means the answer can only have one exact answer, yellow means the question could have two specific questions or answeres, orange means three question-answer pairs possible, red means extreme ambigious query which has no exact answer or can be multiple topics. Your task is not to provide answer, only give one digit value ranging from 1 to 4. Again, DO NOT ATTEMPT TO ANSWER!"

    transformed_data = []

    for item in input_data:
        # Extract required fields
        question = item.get("question", "")
        annotations = item.get("annotations", [])
        articles_html_text = item.get("articles_html_text", [""])

        # Get context from articles_html_text (up to 4000 characters)
        context = "".join(articles_html_text)[200:10000]

        # Determine GPT value based on annotations logic
        gpt_value = determine_gpt_value(annotations)

        # Create human message value with the specified template
        human_value = f"How many answers can be found for the question: {question} Based on the information {{[\"context:{context}']]}}"

        # Create the transformed structure
        transformed_item = {
            "conversations": [
                {
                    "from": "human",
                    "value": human_value
                },
                {
                    "from": "gpt", 
                    "value": str(gpt_value)
                }
            ],
            "system": SYSTEM_MESSAGE
        }

        transformed_data.append(transformed_item)

    return transformed_data

def determine_gpt_value(annotations):
    """
    Determine the GPT value based on annotations structure.

    Args:
        annotations: List of annotation dictionaries

    Returns:
        Integer value (1, 2, 3, or 4)
    """
    # If annotations has only one element, return "1"
    print(annotations)
    #if len(annotations) == 1:
    #    return 1
    
    # Look for multipleQAs type and count qaPairs
    qa_pairs_count = 0
    for annotation in annotations:
        if annotation.get("type") == "singleAnswer":
           return 1
        if annotation.get("type") == "multipleQAs":
            qa_pairs = annotation.get("qaPairs", [])
            qa_pairs_count = len(qa_pairs)
            break

    # Map qaPairs count to GPT value
    if qa_pairs_count == 2:
        return 2
    elif qa_pairs_count == 3:
        return 3
    elif qa_pairs_count > 3:
        return 4
    else:
        return 1  # Default case

def validate_output_json(transformed_data):
    """
    Validate all fields in the output JSON structure.

    Args:
        transformed_data: List of transformed dictionaries

    Returns:
        Tuple of (is_valid: bool, validation_errors: list)
    """
    validation_errors = []

    for i, item in enumerate(transformed_data):
        # Check if item is a dictionary
        if not isinstance(item, dict):
            validation_errors.append(f"Item {i}: Not a dictionary")
            continue

        # Check required top-level fields
        required_fields = ['conversations', 'system']
        for field in required_fields:
            if field not in item:
                validation_errors.append(f"Item {i}: Missing required field '{field}'")

        # Validate conversations field
        if 'conversations' in item:
            conversations = item['conversations']
            if not isinstance(conversations, list):
                validation_errors.append(f"Item {i}: 'conversations' should be a list")
            elif len(conversations) != 2:
                validation_errors.append(f"Item {i}: 'conversations' should have exactly 2 elements")
            else:
                # Validate each conversation entry
                for j, conv in enumerate(conversations):
                    if not isinstance(conv, dict):
                        validation_errors.append(f"Item {i}, conversation {j}: Should be a dictionary")
                        continue

                    # Check required fields
                    if 'from' not in conv:
                        validation_errors.append(f"Item {i}, conversation {j}: Missing 'from' field")
                    elif conv['from'] not in ['human', 'gpt']:
                        validation_errors.append(f"Item {i}, conversation {j}: 'from' should be 'human' or 'gpt'")

                    if 'value' not in conv:
                        validation_errors.append(f"Item {i}, conversation {j}: Missing 'value' field")
                    elif not isinstance(conv['value'], str):
                        validation_errors.append(f"Item {i}, conversation {j}: 'value' should be a string")

                # Validate specific conversation structure
                if len(conversations) == 2:
                    if conversations[0].get('from') != 'human':
                        validation_errors.append(f"Item {i}: First conversation should be from 'human'")
                    if conversations[1].get('from') != 'gpt':
                        validation_errors.append(f"Item {i}: Second conversation should be from 'gpt'")

                    # Validate GPT value is 1, 2, 3, or 4
                    gpt_value = conversations[1].get('value')
                    if gpt_value not in ['1', '2', '3', '4']:
                        validation_errors.append(f"Item {i}: GPT value should be '1', '2', '3', or '4', got '{gpt_value}'")

        # Validate system field
        if 'system' in item:
            if not isinstance(item['system'], str):
                validation_errors.append(f"Item {i}: 'system' should be a string")
            elif len(item['system'].strip()) == 0:
                validation_errors.append(f"Item {i}: 'system' should not be empty")

    is_valid = len(validation_errors) == 0
    return is_valid, validation_errors

def generate_gpt_value_statistics(transformed_data):
    """
    Generate statistics of GPT 'value' fields distribution.

    Args:
        transformed_data: List of transformed dictionaries

    Returns:
        Dictionary with statistics
    """
    gpt_values = []

    # Extract all GPT values
    for item in transformed_data:
        conversations = item.get('conversations', [])
        for conv in conversations:
            if conv.get('from') == 'gpt':
                gpt_values.append(conv.get('value'))

    # Count occurrences of each value
    value_counts = Counter(gpt_values)

    # Calculate statistics
    total_items = len(gpt_values)
    statistics = {
        'total_items': total_items,
        'value_counts': dict(value_counts),
        'value_percentages': {}
    }

    # Calculate percentages
    for value, count in value_counts.items():
        percentage = (count / total_items * 100) if total_items > 0 else 0
        statistics['value_percentages'][value] = round(percentage, 2)

    return statistics

def print_statistics(statistics):
    """
    Print formatted statistics.
    """
    print("=" * 50)
    print("GPT VALUE DISTRIBUTION STATISTICS")
    print("=" * 50)
    print(f"Total items processed: {statistics['total_items']}")
    print()

    print("Value Distribution:")
    print("-" * 30)
    for value in ['1', '2', '3', '4']:
        count = statistics['value_counts'].get(value, 0)
        percentage = statistics['value_percentages'].get(value, 0.0)
        print(f"Value '{value}': {count} items ({percentage}%)")

    print()
    print("Raw counts:", statistics['value_counts'])

# Example usage:
if __name__ == "__main__":
    #Load your JSON data
    with open('/mnt/scratch/users/40645696/train_with_evidence_articles.json', 'r') as f:
        input_data = json.load(f)

    #Transform the data
    transformed_data = transform_json_structure(input_data)

    #Validate the results
    is_valid, errors = validate_output_json(transformed_data)

    #Generate statistics
    stats = generate_gpt_value_statistics(transformed_data)
    print_statistics(stats)

    #Save the results
    with open('/mnt/scratch/users/40645696/LLaMA-Factory/data/ambigQA_train.json', 'w') as f:
        json.dump(transformed_data, f, indent=2)

    # pass
