import json
import os
import warnings
import torch
import transformer_lens
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path

#os.environ ['HF_HOME'] ="/tmp/users/40645696/.cache/huggingface"
#os.environ ['HF_HUB_CACHE'] ="/mnt/scratch/users/40645696/qwen2.5-72b-clariq/Qwen/hub"
#os.environ["HF_HUB_HOME"] = "/mnt/scratch/users/40645696/qwen2.5-72b-clariq/Qwen"

# Path to your Hugging Face JSON dataset
json_path = "/mnt/scratch/users/40645696/LLaMA-Factory/data/claric_sip_dev.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

list_1_2 = []
list_3_4 = []

for entry in data:
    conversations = entry.get("conversations", [])
    if len(conversations) < 2:
        continue  # skip malformed entries
    
    human_msg = conversations[0]
    gpt_msg = conversations[1]
    
    if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
        continue  # skip if format is wrong
    
    human_text = human_msg.get("value", "").strip()
    gpt_text = gpt_msg.get("value", "").strip()

    # Condition: gpt value "1" or "2" → first list; else second list
    if gpt_text in {"1", "2"}:
        list_1_2.append(human_text)
    elif gpt_text in {"3", "4"}:
        list_3_4.append(human_text)


# Set environment variables to force offline mode
#os.environ['HF_HUB_OFFLINE'] = '1'
#os.environ['TRANSFORMERS_OFFLINE'] = '1'
#os.environ['HF_DATASETS_OFFLINE'] = '1'

# Suppress warnings for cleaner output
#warnings.filterwarnings('ignore')

#print("Environment configured for offline operation:")
#print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
#print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
#print(f"HF_DATASETS_OFFLINE: {os.environ.get('HF_DATASETS_OFFLINE')}")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# UPDATE THESE PATHS TO YOUR LOCAL MODEL DIRECTORIES
# Example paths - modify these to match your local setup

# Path to HuggingFace-format model directory
# Should contain: config.json, pytorch_model.bin (or model.safetensors), tokenizer files
HF_MODEL_PATH = "/mnt/scratch/users/40645696/qwen2.5-72b-clariq/Qwen/hub/Qwen/Qwen2.5-72B-Instruct"  # UPDATE THIS

# Model name/identifier (for TransformerLens config)
# This should match a model that TransformerLens knows about (e.g., "gpt2", "llama-7b", etc.)
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"  # UPDATE THIS to match your model architecture

print("Configuration:")
print(f"HuggingFace model path: {HF_MODEL_PATH}")
#print(f"TransformerLens state dict path: {TL_STATE_DICT_PATH}")
print(f"Model architecture: {MODEL_NAME}")

# Load model + tokenizer from local dir
hf_model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_PATH,
    local_files_only=True,      # ensures no download attempt
    torch_dtype=torch.float16,  # or "auto"
    device_map= None           # or {"": "cuda:0"}
).to("cpu")

tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_PATH,
    local_files_only=True
)

print("✅ Loaded Qwen model from:", HF_MODEL_PATH)

try:
    print("Allocated Before:", torch.cuda.memory_allocated() / 1024**2, "MB")
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME,   # name must exist in TL configs
        hf_model  = hf_model,
        tokenizer = tokenizer,
        device="cpu",
        torch_dtype=torch.float16)
    print(f"\n📊 Model loaded successfully!")
    print(f"Model name: {model.cfg.model_name}")
    print(f"Number of parameters: {model.cfg.n_params:,}")
    print(f"Device: {next(model.parameters()).device}")
except Exception as e:
    print(f"\n⚠️  Expected error (update paths first): {e}")


def verify_model(model, test_text="Hello, world!"):
    """
    Verify that the loaded model works correctly by running a forward pass.
    
    Args:
        model: HookedTransformer model to test
        test_text: Text to use for testing
    """
    try:
        print(f"🧪 Testing model with input: '{test_text}'")
        print("-" * 50)
        
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
        
        # Forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():
            logits = model(tokens)
        
        print(f"Output logits shape: {logits.shape}")
        print(f"Output logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        # Generate next token prediction
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
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        import traceback
        traceback.print_exc()

# This will only run if a model was successfully loaded above
# Uncomment the appropriate line once you have a working model

#For HuggingFace format model:
if 'model' in locals():
    verify_model(model, "Is that your baby? I didn't know that you are already a")

# For TransformerLens state dict model:
# if 'tl_model' in locals():
#     verify_model(tl_model, "The quick brown fox")

print("Verification functions defined. Uncomment the appropriate lines once models are loaded.")

