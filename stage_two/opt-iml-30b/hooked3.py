import os
import gc
import time
import torch
from transformers import AutoTokenizer
import transformer_lens
from transformer_lens import HookedTransformer
import json
#-------------------------------------------------------

class AsyncPipelineTransformer(HookedTransformer):
    """
    Asynchronous pipeline with overlapping computation
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streams = []
        self.setup_async_pipeline()
    
    def setup_async_pipeline(self):
        """Setup async streams for overlapping computation"""
        n_gpus = torch.cuda.device_count()
        
        # Create separate streams for each GPU
        for i in range(n_gpus):
            stream = torch.cuda.Stream(device=f'cuda:{i}')
            self.streams.append(stream)
    
    def forward_async(self, input, **kwargs):
        """
        Asynchronous forward pass with overlapping transfers
        """
        if isinstance(input, (str, list)):
            tokens = self.to_tokens(input)
        else:
            tokens = input
        
        n_gpus = len(self.streams)
        layers_per_gpu = self.cfg.n_layers // n_gpus
        
        # Start on GPU 0
        current_device = 'cuda:0'
        tokens = tokens.to(current_device)
        
        with torch.cuda.stream(self.streams[0]):
            residual = self.hook_embed(self.embed(tokens))
            if hasattr(self, 'pos_embed') and self.pos_embed is not None:
                pos_embed = self.hook_pos_embed(self.pos_embed(tokens))
                residual = residual + pos_embed
        
        # Pipeline through GPUs with async transfers
        for gpu_id in range(n_gpus):
            target_device = f'cuda:{gpu_id}'
            stream = self.streams[gpu_id]
            
            # Async transfer to target device
            with torch.cuda.stream(stream):
                if residual.device != torch.device(target_device):
                    residual = residual.to(target_device, non_blocking=True)
                
                # Process layers on this GPU
                start_layer = gpu_id * layers_per_gpu
                end_layer = min((gpu_id + 1) * layers_per_gpu, self.cfg.n_layers)
                
                for layer_idx in range(start_layer, end_layer):
                    residual = self.blocks[layer_idx](residual)
        
        # Final processing
        final_gpu = n_gpus - 1
        with torch.cuda.stream(self.streams[final_gpu]):
            if hasattr(self, 'ln_final') and self.ln_final is not None:
                residual = self.ln_final(residual)
            logits = self.unembed(residual)
        
        # Wait for all streams to complete
        for stream in self.streams:
            stream.synchronize()
        
        return logits

def batch_inference(model, texts, batch_size=4):
    """
    Process multiple texts in batches for maximum efficiency
    """
    results = []
    
    print(f"🔄 Processing {len(texts)} texts in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        batch_tokens = []
        max_len = 0
        
        for text in batch_texts:
            tokens = model.to_tokens(text)
            batch_tokens.append(tokens)
            max_len = max(max_len, tokens.shape[1])
        
        # Pad to same length
        padded_batch = torch.zeros((len(batch_tokens), max_len), dtype=torch.long)
        for j, tokens in enumerate(batch_tokens):
            padded_batch[j, :tokens.shape[1]] = tokens[0]  # Remove batch dim
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            logits = model(padded_batch)
            end_time = time.time()
            
            print(f"Batch {i//batch_size + 1}: {end_time - start_time:.2f}s for {len(batch_texts)} samples")
        
        # Process results
        for j, text in enumerate(batch_texts):
            sample_logits = logits[j]
            next_token_logits = sample_logits[-1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            top_token_id = torch.argmax(next_token_probs).item()
            top_token = model.to_string(top_token_id)
            
            results.append({
                'input': text,
                'next_token': top_token,
                'confidence': next_token_probs[top_token_id].item()
            })
    
    return results
def load_fast_multi_gpu_model(model_name, hf_model_path, tokenizer):
    """
    Load model with optimized multi-GPU setup
    """
    
    def print_memory_status():
        print("\n" + "="*50)
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")
        print("="*50)
    
    # Clear GPU memory
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    
    print_memory_status()
    
    # Load model on CPU first
    print("🔄 Loading model on CPU...")
    model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=None,
        tokenizer=tokenizer,
        device="cpu",
        torch_dtype=torch.float16,
        fold_ln=True,
        center_writing_weights=False,
        fold_value_biases=True,
    )
    
    # Convert to fast multi-GPU model
    print("🔄 Converting to fast multi-GPU model...")
    fast_model = FastMultiGPUHookedTransformer(model.cfg, tokenizer=tokenizer)
    
    # Copy weights efficiently
    print("🔄 Copying model weights...")
    with torch.no_grad():
        fast_model.load_state_dict(model.state_dict(), strict=False)
    
    # Clear original model from memory
    del model
    gc.collect()
    
    # Distribute across GPUs
    fast_model.distribute_model()
    
    print_memory_status()
    
    return fast_model
def batch_inference(model, texts, batch_size=4):
    """
    Process multiple texts in batches for maximum efficiency
    """
    results = []
    
    print(f"🔄 Processing {len(texts)} texts in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        batch_tokens = []
        max_len = 0
        
        for text in batch_texts:
            tokens = model.to_tokens(text)
            batch_tokens.append(tokens)
            max_len = max(max_len, tokens.shape[1])
        
        # Pad to same length
        padded_batch = torch.zeros((len(batch_tokens), max_len), dtype=torch.long)
        for j, tokens in enumerate(batch_tokens):
            padded_batch[j, :tokens.shape[1]] = tokens[0]  # Remove batch dim
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            logits = model(padded_batch)
            end_time = time.time()
            
            print(f"Batch {i//batch_size + 1}: {end_time - start_time:.2f}s for {len(batch_texts)} samples")
        
        # Process results
        for j, text in enumerate(batch_texts):
            sample_logits = logits[j]
            next_token_logits = sample_logits[-1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            top_token_id = torch.argmax(next_token_probs).item()
            top_token = model.to_string(top_token_id)
            
            results.append({
                'input': text,
                'next_token': top_token,
                'confidence': next_token_probs[top_token_id].item()
            })
    
    return results
def load_fast_multi_gpu_model(model_name, hf_model_path, tokenizer):
    """
    Load model with optimized multi-GPU setup
    """
    
    def print_memory_status():
        print("\n" + "="*50)
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")
        print("="*50)
    
    # Clear GPU memory
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    
    print_memory_status()
    
    # Load model on CPU first
    print("🔄 Loading model on CPU...")
    model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=None,
        tokenizer=tokenizer,
        device="cpu",
        torch_dtype=torch.float16,
        fold_ln=True,
        center_writing_weights=False,
        fold_value_biases=True,
    )
    
    # Convert to fast multi-GPU model
    print("🔄 Converting to fast multi-GPU model...")
    fast_model = FastMultiGPUHookedTransformer(model.cfg, tokenizer=tokenizer)
    
    # Copy weights efficiently
    print("🔄 Copying model weights...")
    with torch.no_grad():
        fast_model.load_state_dict(model.state_dict(), strict=False)
    
    # Clear original model from memory
    del model
    gc.collect()
    
    # Distribute across GPUs
    fast_model.distribute_model()
    
    print_memory_status()
    
    return fast_model

class FastMultiGPUHookedTransformer(HookedTransformer):
    """
    High-performance multi-GPU HookedTransformer with optimized forward pass
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_map = {}
        self.layer_devices = []
        self.pipeline_stages = []
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup pipeline stages for efficient processing"""
        n_layers = self.cfg.n_layers
        n_gpus = torch.cuda.device_count()
        layers_per_gpu = n_layers // n_gpus
        
        # Create pipeline stages
        for gpu_id in range(n_gpus):
            start_layer = gpu_id * layers_per_gpu
            end_layer = min((gpu_id + 1) * layers_per_gpu, n_layers)
            self.pipeline_stages.append({
                'device': f'cuda:{gpu_id}',
                'start_layer': start_layer,
                'end_layer': end_layer,
                'layers': list(range(start_layer, end_layer))
            })
        
        # Store layer-to-device mapping for quick lookup
        for stage in self.pipeline_stages:
            for layer_idx in stage['layers']:
                self.layer_devices.append(stage['device'])
    
    def forward(self, input, **kwargs):
        """
        Optimized forward pass with minimal device transfers
        """
        # Tokenize input
        if isinstance(input, (str, list)):
            tokens = self.to_tokens(input)
        else:
            tokens = input
        
        batch_size, seq_len = tokens.shape
        
        # Start timing
        import time
        start_time = time.time()
        
        # Stage 1: Embeddings (GPU 0)
        embed_device = self.pipeline_stages[0]['device']
        tokens = tokens.to(embed_device)
        
        with torch.cuda.device(embed_device):
            residual = self.hook_embed(self.embed(tokens))
            
            # Add positional embeddings if they exist
            if hasattr(self, 'pos_embed') and self.pos_embed is not None:
                pos_embed = self.hook_pos_embed(self.pos_embed(tokens))
                residual = residual + pos_embed
        
        # Pipeline through transformer layers
        for stage_idx, stage in enumerate(self.pipeline_stages):
            stage_device = stage['device']
            
            # Transfer to stage device (only once per stage)
            if residual.device.type != stage_device.split(':')[0] or residual.device.index != int(stage_device.split(':')[1]):
                residual = residual.to(stage_device, non_blocking=True)
            
            # Process all layers in this stage on the same device
            with torch.cuda.device(stage_device):
                for layer_idx in range(stage['start_layer'], stage['end_layer']):
                    residual = self.blocks[layer_idx](residual)
                    
                # Synchronize after processing stage
                torch.cuda.current_stream().synchronize()
        
        # Final processing on last GPU
        final_device = self.pipeline_stages[-1]['device']
        if residual.device != torch.device(final_device):
            residual = residual.to(final_device, non_blocking=True)
        
        with torch.cuda.device(final_device):
            # Final layer norm
            if hasattr(self, 'ln_final') and self.ln_final is not None:
                residual = self.ln_final(residual)
            
            # Unembedding
            logits = self.unembed(residual)
        
        end_time = time.time()
        print(f"⚡ Forward pass completed in {end_time - start_time:.2f} seconds")
        
        return logits

    def distribute_model(self):
        """
        Distribute model across GPUs with optimal placement
        """
        print("🔄 Distributing model across GPUs...")
        
        # Move embeddings to GPU 0
        embed_device = self.pipeline_stages[0]['device']
        self.embed = self.embed.to(embed_device)
        if hasattr(self, 'pos_embed') and self.pos_embed is not None:
            self.pos_embed = self.pos_embed.to(embed_device)
        
        # Distribute transformer blocks
        for stage in self.pipeline_stages:
            stage_device = stage['device']
            print(f"Moving layers {stage['start_layer']}-{stage['end_layer']-1} to {stage_device}")
            
            for layer_idx in range(stage['start_layer'], stage['end_layer']):
                self.blocks[layer_idx] = self.blocks[layer_idx].to(stage_device)
        
        # Move final layers to last GPU
        final_device = self.pipeline_stages[-1]['device']
        if hasattr(self, 'ln_final') and self.ln_final is not None:
            self.ln_final = self.ln_final.to(final_device)
        self.unembed = self.unembed.to(final_device)
        
        print("✅ Model distribution completed")

#------------------------------------------------------
# Optimize CUDA settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configuration
HF_MODEL_PATH = "/mnt/scratch/users/40645696/qwen2.5-72b-clariq/Qwen/hub/Qwen/Qwen2.5-72B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

def main():
    """Main execution with fast multi-GPU setup"""
    
    # Load your dataset
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
    
    # Load tokenizer
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, local_files_only=True)
    print("✅ Tokenizer loaded")
    
    # Load fast multi-GPU model
    try:
        print("🚀 Loading fast multi-GPU model...")
        model = load_fast_multi_gpu_model(MODEL_NAME, HF_MODEL_PATH, tokenizer)
        
        print(f"✅ Fast multi-GPU model loaded!")
        print(f"Model: {model.cfg.model_name}")
        print(f"Parameters: {model.cfg.n_params:,}")
        
        # Test single inference
        print("\n🧪 Testing single inference...")
        test_text = "Is that your baby? I didn't know that you are already a"
        
        start_time = time.time()
        with torch.no_grad():
            logits = model(test_text)
            
            # Get prediction
            next_token_logits = logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            top_token_id = torch.argmax(next_token_probs).item()
            top_token = model.to_string(top_token_id)
        
        end_time = time.time()
        
        print(f"Input: {test_text}")
        print(f"Predicted: '{top_token}'")
        print(f"⚡ Single inference time: {end_time - start_time:.2f} seconds")
        
        # Test batch inference
        print("\n🧪 Testing batch inference...")
        test_batch = [
            "Hello, how are you?",
            "What is the weather like?",
            "Is that your baby? I didn't know that you are already a",
            "The quick brown fox jumps over"
        ]
        
        batch_results = batch_inference(model, test_batch, batch_size=2)
        
        print("\n📊 Batch Results:")
        for result in batch_results:
            print(f"'{result['input']}' → '{result['next_token']}' ({result['confidence']:.3%})")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

