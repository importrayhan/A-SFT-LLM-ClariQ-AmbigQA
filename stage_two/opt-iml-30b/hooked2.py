import json
import os
import gc
import psutil
import torch
import transformer_lens
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def print_memory_usage(stage=""):
    """Print detailed memory usage information"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_free = gpu_total - gpu_memory
        
        print(f"\n{'='*50}")
        print(f"MEMORY USAGE - {stage}")
        print(f"{'='*50}")
        print(f"🔥 GPU Memory:")
        print(f"   Allocated: {gpu_memory:.2f} GB / {gpu_total:.2f} GB ({gpu_memory/gpu_total*100:.1f}%)")
        print(f"   Reserved:  {gpu_reserved:.2f} GB")
        print(f"   Free:      {gpu_free:.2f} GB")
    
    # CPU Memory
    cpu_memory = psutil.virtual_memory()
    cpu_used = cpu_memory.used / 1024**3
    cpu_total = cpu_memory.total / 1024**3
    cpu_available = cpu_memory.available / 1024**3
    
    print(f"💻 CPU Memory:")
    print(f"   Used:      {cpu_used:.2f} GB / {cpu_total:.2f} GB ({cpu_used/cpu_total*100:.1f}%)")
    print(f"   Available: {cpu_available:.2f} GB")
    print(f"{'='*50}\n")

def clear_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# Configuration
HF_MODEL_PATH = "/mnt/scratch/users/40645696/qwen2.5-72b-clariq/Qwen/hub/Qwen/Qwen2.5-72B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Set environment variables for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print_memory_usage("Initial")

# Load your dataset (this part is fine as-is)
json_path = "/mnt/scratch/users/40645696/LLaMA-Factory/data/claric_sip_dev.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

list_1_2 = []
list_3_4 = []

for entry in data:
    conversations = entry.get("conversations", [])
    if len(conversations) < 2:
        continue
    human_msg = conversations[0]
    gpt_msg = conversations[1]
    if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
        continue
    human_text = human_msg.get("value", "").strip()
    gpt_text = gpt_msg.get("value", "").strip()
    if gpt_text in {"1", "2"}:
        list_1_2.append(human_text)
    elif gpt_text in {"3", "4"}:
        list_3_4.append(human_text)

print(f"Configuration:")
print(f"HuggingFace model path: {HF_MODEL_PATH}")
print(f"Model architecture: {MODEL_NAME}")

# Load tokenizer first (minimal memory usage)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_PATH,
    local_files_only=True
)
print("✅ Tokenizer loaded")

print_memory_usage("After Tokenizer")

# CRITICAL: Don't load HF model here - let TransformerLens handle it
try:
    print("Loading model with TransformerLens...")
    
    # Use CPU-first strategy for large models
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME,
        hf_model=None,  # Let it load internally with proper memory management
        tokenizer=tokenizer,
        device="cpu",  # Start on CPU
        torch_dtype=torch.float16,
        # Additional memory optimization parameters
        fold_ln=True,  # Fold layer norm weights to save memory
        center_writing_weights=False,  # Don't center weights (saves memory)
        fold_value_biases=True,  # Fold value biases
        #low_cpu_mem_usage=True,  # Enable low CPU memory usage
        #device_map="auto",  # Auto device mapping
    )
    
    print(f"\n📊 Model loaded successfully!")
    print(f"Model name: {model.cfg.model_name}")
    print(f"Number of parameters: {model.cfg.n_params:,}")
    print(f"Device: {next(model.parameters()).device}")
    
    print_memory_usage("After Model Loading")
    
except Exception as e:
    print(f"\n⚠️ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    print_memory_usage("After Error")

def verify_model_memory_efficient(model, test_text="Hello, world!"):
    """Memory-efficient model verification"""
    try:
        print(f"🧪 Testing model with input: '{test_text}'")
        print("-" * 50)
        
        print_memory_usage("Before Verification")
        
        # Basic model info
        print(f"Model architecture: {model.cfg.model_name}")
        print(f"Number of layers: {model.cfg.n_layers}")
        print(f"Hidden size: {model.cfg.d_model}")
        print(f"Vocabulary size: {model.cfg.d_vocab}")
        print(f"Context length: {model.cfg.n_ctx}")
        
        # Tokenize input
        tokens = model.to_tokens(test_text)
        print(f"\nTokenized input: {tokens}")
        print(f"Token shape: {tokens.shape}")
        
        print_memory_usage("After Tokenization")
        
        # Forward pass with gradient disabled and memory optimization
        print("\nRunning forward pass...")
        with torch.no_grad():
            # Move only the tokens to the same device as the model
            device = next(model.parameters()).device
            tokens = tokens.to(device)
            
            # Run inference
            logits = model(tokens)
            
            print_memory_usage("During Forward Pass")
        
        print(f"Output logits shape: {logits.shape}")
        print(f"Output logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        # Generate predictions
        next_token_logits = logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        top_token_id = torch.argmax(next_token_probs).item()
        top_token = model.to_string(top_token_id)
        
        print(f"\n🎯 Predicted next token: '{top_token}' (ID: {top_token_id})")
        print(f"Confidence: {next_token_probs[top_token_id].item():.3%}")
        
        # Show top 5 predictions
        top_5_indices = torch.topk(next_token_probs, 5).indices
        print("\n🏆 Top 5 predictions:")
        for i, token_id in enumerate(top_5_indices):
            token_str = model.to_string(token_id.item())
            prob = next_token_probs[token_id].item()
            print(f"  {i+1}. '{token_str}' ({prob:.3%})")
        
        print("\n✅ Model verification successful!")
        print_memory_usage("After Verification")
        
        # Clean up
        del logits, next_token_logits, next_token_probs, top_5_indices
        clear_memory()
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        print_memory_usage("After Verification Error")
        import traceback
        traceback.print_exc()

# Run verification if model loaded successfully
if 'model' in locals():
    verify_model_memory_efficient(model, "Is that your baby? I didn't know that you are already a")

print_memory_usage("Final")

