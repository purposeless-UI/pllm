import os
import sys
import torch
import math  
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root directory to system path so Python finds the src folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# UPDATED: Importing your specific custom project layout classes directly
from src.model.model import PaniniConfig, PaniniTransformer
from src.dataset_pipeline.dataset import create_pllm_dataloader

@torch.no_grad()
def run_evaluation_metrics():
    print("🔬 [EVALUATION] Starting custom Panini token validation check...")
    
    # 1. Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"💻 [EVALUATION] Utilizing computation node: {device}")

    # 2. Match your custom architecture setup dimensions perfectly
    # FIX: Swapped dim, n_layers, and n_heads to match your train_cluster.py settings
    config = PaniniConfig(
        vocab_size=64000,
        dim=1024,            # Fixed config mismatch
        n_layers=12,         # Fixed config mismatch
        n_heads=16,          # Fixed config mismatch
        n_kv_heads=2,
        max_seq_len=32
    )

    # 3. Reconstruct model and load saved snapshot weights
    # UPDATED: Instantiating your dedicated custom project architecture class
    model = PaniniTransformer(config)
    # FIX: Point checkpoint filename to the actual saved model path
    model_path = "models/panini_sanskrit_final.pt"
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        
        # FIX: Safe guard to unpack wrapped state dictionary structures seamlessly if present
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        print(f"✅ [EVALUATION] Pre-trained weights restored from: {model_path}")
    else:
        print(f"⚠️ [EVALUATION] No weights found at '{model_path}'. Running evaluation on random parameters.")
        
    model.to(device)
    model.eval()

    # 4. Initialize Data Loader
    try:
        dataloader = create_pllm_dataloader(
            raw_data_dir="data/raw",
            tokenizer_path="tokenizer.json",
            batch_size=2,
            seq_len=config.max_seq_len
        )
    except Exception as e:
        print(f"❌ [EVALUATION] Data stream error: {e}")
        return

    # 5. Core Metric Calculation Loop
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0

    print("跑 [EVALUATION] Processing evaluation batches...")
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        
        # Reshape to token layout: (Batch * Seq, Vocab)
        flat_logits = logits.view(-1, config.vocab_size)
        flat_targets = targets.contiguous().view(-1)

        # Calculate raw loss sum
        loss = loss_fn(flat_logits, flat_targets)
        total_loss += loss.item()
        
        # Calculate strict word ordering matching accuracy
        predictions = torch.argmax(flat_logits, dim=-1)
        correct_predictions += (predictions == flat_targets).sum().item()
        total_tokens += flat_targets.numel()

    if total_tokens == 0:
        print("⚠️ [EVALUATION] Dataset too small to calculate meaningful statistics.")
        return

    # Calculate final scores
    avg_loss = total_loss / total_tokens
    accuracy_percentage = (correct_predictions / total_tokens) * 100
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    # UPDATED: Rebranded the final output reports for your project namespace
    print("\n==============================================================")
    print("📊 PANINI CORE TRANSFORMER PERFORMANCE REVIEWS")
    print("==============================================================")
    print(f"🎯 Next-Word Token Match Accuracy: {accuracy_percentage:.2f}%")
    print(f"📉 Cross-Entropy Evaluation Loss: {avg_loss:.4f}")
    print(f"🌀 System Model Perplexity Score: {perplexity:.2f}")
    print("==============================================================")

if __name__ == "__main__":
    run_evaluation_metrics()
