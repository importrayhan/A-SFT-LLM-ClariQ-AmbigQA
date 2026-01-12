import json
from collections import Counter
import re
from typing import List, Dict, Any, Tuple
import math
from neural_cherche import models, retrieve
import torch



def chunk_text_with_overlap(text: str, chunk_size: int = 500, overlap: int = 20) -> List[Dict[str, Any]]:
    """
    Chunk text into fixed-size chunks with overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters  
        overlap: Overlap between chunks in characters
    
    Returns:
        List of dictionaries with chunk information
    """
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Avoid empty chunks
        if chunk_text.strip():
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "title": f"Document Chunk {chunk_id}",
                "text": chunk_text.strip(),
                "start_pos": start,
                "end_pos": min(end, len(text))
            })
            chunk_id += 1
        
        # Move start position considering overlap
        start = end - overlap
        
        # Break if we've reached the end
        if end >= len(text):
            break
    
    return chunks

def calculate_bm25_similarity(query: str, document: str, k1: float = 1.5, b: float = 0.75) -> float:
    """
    Calculate BM25 similarity score between query and document.
    This is a more sophisticated retrieval algorithm than simple TF-IDF.
    
    Args:
        query: Query text
        document: Document text
        k1: Term frequency saturation parameter
        b: Length normalization parameter
    
    Returns:
        BM25 similarity score
    """
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    query_tokens = tokenize(query)
    doc_tokens = tokenize(document)
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    # Calculate term frequencies
    doc_tf = {}
    for token in doc_tokens:
        doc_tf[token] = doc_tf.get(token, 0) + 1
    
    # Document length
    doc_length = len(doc_tokens)
    avg_doc_length = doc_length  # Simplified for single document
    
    score = 0.0
    
    # Calculate BM25 score for each query term
    for query_term in set(query_tokens):
        if query_term in doc_tf:
            tf = doc_tf[query_term]
            
            # BM25 formula components
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            
            # IDF approximation (simplified)
            idf = math.log(2.0)  # Simplified IDF
            
            score += idf * (numerator / denominator)
    
    # Add bonus for exact phrase matches
    if query.lower() in document.lower():
        score += 1.0
    
    # Add bonus for multiple word matches
    query_words = set(query_tokens)
    doc_words = set(doc_tokens)
    word_overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
    score += word_overlap * 0.5
    
    return score

def retrieve_top_chunks(model, query: str, chunks: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve top k chunks based on BM25 similarity to query.
    Simulates the neural retrieval approach from the provided code.
    
    Args:
        query: Query text (corresponds to queries in neural-cherche example)
        chunks: List of chunk dictionaries (corresponds to documents in neural-cherche example)
        k: Number of top chunks to retrieve
    
    Returns:
        List of top k chunks with similarity scores (corresponds to scores in neural-cherche example)
    """
    if not chunks:
        return []
    # SPLADE V3
    
    retriever = retrieve.Splade(
    key="id",
    on=["text"],
    model=model
    )

    documents_embeddings = retriever.encode_documents(
        documents=chunks,
        batch_size=batch_size,
    )

    retriever.add(
        documents_embeddings=documents_embeddings,
    )

    queries = [query]

    queries_embeddings = retriever.encode_queries(
        queries=queries,
        batch_size=batch_size,
    )

    similarities  = retriever(
        queries_embeddings=queries_embeddings,
        k= k,
    )
    #similarities = [
    #[   {'id': 'chunk_1', 'similarity': np.float32(2.0331366)},
    #    {'id': 'chunk_0', 'similarity': np.float32(0.11928999)}]]
    

    chunk_scores = []

    # Flatten similarity results in case they’re nested
    for sim_group in similarities:
        for sim_item in sim_group:
            sim_id = sim_item["id"]
            similarity = sim_item["similarity"]

            # find the corresponding chunk by id
            for chunk in chunks:
                if chunk["id"] == sim_id:
                    chunk_scores.append({
                        "id": chunk["id"],
                        "title": chunk["title"],
                        "text": chunk["text"],
                        "similarity": float(similarity),
                        "start_pos": chunk["start_pos"],
                        "end_pos": chunk["end_pos"]
                    })
                    break
    
    # Sort by similarity in descending order (simulating retriever(queries_embeddings, k=3))
    chunk_scores.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top k chunks
    return chunk_scores#[:k]

def process_single_entry_retrieval(model, question: str, articles_text: str, chunk_size: int = 500, overlap: int = 20, top_k: int = 3) -> List[str]:
    """
    Process a single entry with retrieval-based context selection.
    This function simulates the workflow from the neural-cherche example.
    
    Args:
        question: The question to answer
        articles_text: Combined article text
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        top_k: Number of top chunks to retrieve
    
    Returns:
        List of top chunk texts for context
    """
    
    # Step 1: Create chunks (equivalent to documents in neural-cherche example)
    chunks = chunk_text_with_overlap(articles_text, chunk_size, overlap)
    
    if not chunks:
        return [articles_text[:4000]]  # Fallback
    
    # Step 2: Retrieve top chunks (equivalent to neural-cherche retrieval pipeline)
    # queries = [question]  # In neural-cherche example
    # documents = chunks    # Chunked article_plain_text with serial id
    #print(chunks)
    top_chunks = retrieve_top_chunks(model, question, chunks, top_k)
    
    # Step 3: Extract text from top chunks (equivalent to getting entries from documents)
    # Choose your separator
    separator = "\n---\n"  # could be "\n\n", " || ", etc.

    # Combine all texts with separator
    context_texts = [separator.join(chunk["text"] for chunk in top_chunks)]
    #context_texts = [chunk["text"] for chunk in top_chunks]
    
    return context_texts

def transform_json_structure_with_retrieval(model, input_data: List[Dict], chunk_size: int = 500, overlap: int = 20, top_k: int = 3) -> List[Dict]:
    """
    Transform the input JSON structure with chunking and retrieval-based context selection.
    
    Args:
        input_data: List of dictionaries with the input structure
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        top_k: Number of top chunks to retrieve for context
    
    Returns:
        List of dictionaries with the transformed structure
    """
    
    # Define the constant system message
    SYSTEM_MESSAGE = ("You are a search engine performance predictor. A user is looking for something which is often not clearly understood. Your task is to decide if a search request is oscure. You have a subset of context which you can use to accomplish this task. Given the user request, do you think you have a clearly specified search query to find exactly one answer with no contradiction? Now think of a four level indicator ranging from 1 green, 2 yellow, 3 orange, 4 red. Green means the answer can only have one exact answer, yellow means the question could have two specific questions or answeres, orange means three question-answer pairs possible, red means extreme ambigious query which has no exact answer or can be multiple topics. Your task is not to provide answer, only give one digit value ranging from 1 to 4. Again, DO NOT ATTEMPT TO ANSWER!")
    
    transformed_data = []
    
    # Process each entry in a loop (as requested)
    for item in input_data:
        # Extract required fields
        question = item.get("question", "")
        annotations = item.get("annotations", [])
        articles_html_text = item.get("articles_html_text", [])
        articles_plain_text = item.get("articles_plain_text", [])
        
        # Combine all article texts
        combined_html_text = " ".join(articles_html_text)
        combined_plain_text = " ".join(articles_plain_text)
        
        # Use plain text for better retrieval, fallback to HTML text
        text_for_chunking = combined_plain_text if combined_plain_text.strip() else combined_html_text
        
        # Process with retrieval (simulating the neural-cherche workflow)
        context_texts = process_single_entry_retrieval(model, question, text_for_chunking, chunk_size, overlap, top_k)
        
        # Join top chunks for context
        context = " ".join(context_texts)
        
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

def determine_gpt_value(annotations: List[Dict]) -> int:
    """
    Determine the GPT value based on annotations structure.
    
    Args:
        annotations: List of annotation dictionaries
    
    Returns:
        Integer value (1, 2, 3, or 4)
    """
    # If annotations has only one element, return "1"
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

def validate_output_json(transformed_data: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate all fields in the output JSON structure.
    """
    validation_errors = []
    
    for i, item in enumerate(transformed_data):
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
                # Validate GPT value
                if len(conversations) == 2:
                    gpt_value = conversations[1].get('value')
                    if gpt_value not in ['1', '2', '3', '4']:
                        validation_errors.append(f"Item {i}: GPT value should be '1', '2', '3', or '4', got '{gpt_value}'")
    
    is_valid = len(validation_errors) == 0
    return is_valid, validation_errors

def generate_gpt_value_statistics(transformed_data: List[Dict]) -> Dict:
    """
    Generate statistics of GPT 'value' fields distribution.
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

def print_statistics(statistics: Dict):
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


if __name__ == "__main__":
    # Run comprehensive test
    
#    print("\n" + "=" * 70)
#    print("📄 USAGE INSTRUCTIONS")
#    print("=" * 70)
    print("""
To use this script with your own JSON data:

1. Load your JSON data:
   with open('your_data.json', 'r') as f:
       input_data = json.load(f)

2. Transform with retrieval:
   transformed = transform_json_structure_with_retrieval(
       input_data, 
       chunk_size=500,  # Adjust as needed
       overlap=20,      # Adjust as needed
       top_k=3          # Number of chunks to retrieve
   )

3. Validate and save:
   is_valid, errors = validate_output_json(transformed)
   if is_valid:
       with open('transformed_output.json', 'w') as f:
           json.dump(transformed, f, indent=2)
   
4. Generate statistics:
   stats = generate_gpt_value_statistics(transformed)
   print_statistics(stats)
""")
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64

model = models.Splade(
    model_name_or_path="/mnt/scratch/users/40645696/neural-cherche-sparse-embed",
    device=device,
)

#Load your JSON data:
with open('/mnt/scratch/users/40645696/dev_with_evidence_articles.json', 'r') as f:
       input_data = json.load(f)


#Transform with retrieval:
transformed = transform_json_structure_with_retrieval(
       model,
       input_data, 
       chunk_size=748,  # Adjust as needed
       overlap= 42,      # Adjust as needed
       top_k=6         # Number of chunks to retrieve
   )
#print(transformed)

is_valid, errors = validate_output_json(transformed)
if is_valid:
    with open('/mnt/scratch/users/40645696/LLaMA-Factory/data/ambigQA_dev_SPLADE3.json', 'w') as f:
        json.dump(transformed, f, indent=2)
