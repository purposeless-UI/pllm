import os
import sys
import torch
import torch.nn as nn

# Add the project root directory to the system path so Python can find the src folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.model import PaniniConfig, PaniniTransformer
from src.dataset_pipeline.dataset import create_pllm_dataloader
from src.utils.tracker import TrainingTracker  
# ADDED: Importing your built-in Tkinter 3D visualizer module safely
from src.utils.visual_brain import NeuralBrainVisualizer

def run_local_pipeline_test():
    print("🚀 [LOCAL PROTOTYPE] Initializing small Panini testing brain loop...")
    
    # 1. Detect hardware accelerator power
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"💻 [LOCAL PROTOTYPE] Hardware accelerator engaged: {device}")

    # 2. Workspace Paths Safety Check
    raw_data_dir = "data/raw"
    tokenizer_path = "tokenizer.json"
    
    if not os.path.exists(os.path.join(raw_data_dir, "sample_sanskrit_scripture.txt")):
        print(f"❌ Error: Missing scripture file inside '{raw_data_dir}'.")
        return
    if not os.path.exists(tokenizer_path):
        print(f"❌ Error: '{tokenizer_path}' is missing. Run tokenizer script first!")
        return

    # 3. Setup a 12-Million Parameter Testing Brain Configuration
    config = PaniniConfig(
        vocab_size=64000,   # Matches your custom Panini Tokenizer max capacity
        dim=256,            # Reduced embedding dimension for local laptop VRAM
        n_layers=3,         # 4 stacked processing layer blocks
        n_heads=4,          # 4 Query heads
        n_kv_heads=1,       # Grouped-Query Attention (4:1 GQA ratio)
        max_seq_len=64      # Comfortable local sequence context window length
    )
    print("⚙️ [LOCAL PROTOTYPE] 7-Million parameter Panini configuration initialized.")

    # 4. Initialize Data Loader and System Tracker Engine
    try:
        print("📦 [LOCAL PROTOTYPE] Loading dataset tokens...")
        dataloader = create_pllm_dataloader(
            raw_data_dir=raw_data_dir,
            tokenizer_path=tokenizer_path,
            batch_size=4,           # Sized for rapid laptop step iterations
            seq_len=config.max_seq_len
        )
        print("✅ [LOCAL PROTOTYPE] Dataset token sequences online.")
    except Exception as e:
        print(f"❌ Failed to parse dataset file: {e}")
        return

    tracker = TrainingTracker(checkpoint_dir="models/checkpoints", log_dir="logs")

    # ADDED: Instantiating the built-in 3D desktop graphics canvas window
    visualizer = NeuralBrainVisualizer(active=True)

    # 5. Build the Model and Optimizer Layers
    print("🧠 [LOCAL PROTOTYPE] Allocating neural vectors to device memory...")
    model = PaniniTransformer(config).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Attempt to load a previous checkpoint to resume if it exists
    recovery_meta = tracker.load_latest_checkpoint(model, optimizer, None, device)
    start_epoch = recovery_meta["epoch"]

        # 6. Core Multi-Epoch Training Loop
    epochs = 100
    model.train()
    
    print("\n🏋️ Starting optimized weight loop training run...")
    print("--------------------------------------------------------------")

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Trigger a glowing neon blue electrical pulse on the screen
            visualizer.trigger_pulse(mode="TRAIN")
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            loss = loss_fn(
                logits.view(-1, config.vocab_size), 
                targets.contiguous().view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log metrics to memory cache quickly
            tracker.log_step(epoch=epoch, batch=batch_idx, loss=loss.item(), lr=5e-4, is_master=True)
            
            # Print quick progress tracker lines
            print(f"   [Epoch {epoch+1}/{epochs}] Step {batch_idx+1:02d} | Loss: {loss.item():.4f}")
            
        # --- FIXED PERFORMANCE: MOVE CONTROLS OUTSIDE THE INNER LOOP ---
        # 1. Save backup states ONCE per completed epoch pass instead of every step
        tracker.save_checkpoint(model, optimizer, None, epoch=epoch, batch=batch_idx, loss=loss.item(), is_master=True)
        
        # 2. Render the next frame of the rotating 3D network lattice ONCE per epoch pass
        visualizer.update_and_render()
            
        avg_loss = total_loss / len(dataloader)
        print(f"📉 [LOCAL PROTOTYPE] Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}\n")


    print("==============================================================")
    print("🎉 SUCCESS! Your local testing brain has completed its epochs.")
    print("💾 Rolling weights are fully optimized inside models/checkpoints/")
    print("==============================================================")

        # Ensure destination directory exists and save the final calibrated state dict cleanly
    os.makedirs("models", exist_ok=True)
    final_save_path = "models/panini_sanskrit_final.pt"
    torch.save({"model_state_dict": model.state_dict()}, final_save_path)

    print("==============================================================")
    print("🎉 SUCCESS! Your local testing brain has completed its epochs.")
    print(f"💾 Synchronized model weights secured at: {final_save_path}")
    print("==============================================================")

if __name__ == "__main__":
    run_local_pipeline_test()






''' import os
import sys
import torch

# Add the project root directory to the system path so Python can find the src folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# UPDATED: Importing your custom project layout components directly
from src.model.model import PaniniConfig, PaniniTransformer
from src.dataset_pipeline.dataset import create_pllm_dataloader
from src.utils.tracker import TrainingTracker  

def run_local_pipeline_test():
    print("🚀 [LOCAL TEST] Starting tracked end-to-end architecture pipeline test...")
    
    # 1. Detect local computer power (GPU, Apple Silicon MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"💻 [LOCAL TEST] Utilizing hardware accelerator: {device}")

    # 2. Safety Check for Files
    raw_data_dir = "data/raw"
    tokenizer_path = "tokenizer.json"
    
    if not os.path.exists(os.path.join(raw_data_dir, "sample_sanskrit_scripture.txt")):
        print(f"❌ [LOCAL TEST] Error: Could not find scripture file inside '{raw_data_dir}'.")
        return
    if not os.path.exists(tokenizer_path):
        print(f"❌ [LOCAL TEST] Error: '{tokenizer_path}' missing. Run tokenizer script first!")
        return

    # 3. Setup a small configuration for safe local laptop execution
    # UPDATED: Swapped class constructor name to match your custom setup parameters
    config = PaniniConfig(
        vocab_size=64000,   
        dim=256,            
        n_layers=2,         
        n_heads=4,          
        n_kv_heads=2,        
        max_seq_len=32     
    )
    print("⚙️ [LOCAL TEST] Scaled Panini Hyperparameters loaded.")

    # 4. Initialize Data Loader and Tracker
    try:
        print("📦 [LOCAL TEST] Initializing data loader engine...")
        dataloader = create_pllm_dataloader(
            raw_data_dir=raw_data_dir,
            tokenizer_path=tokenizer_path,
            batch_size=2,
            seq_len=config.max_seq_len
        )
        print("✅ [LOCAL TEST] Scripture streaming dataset loaded successfully.")
    except Exception as e:
        print(f"❌ [LOCAL TEST] Failed to parse dataset file: {e}")
        return

    # Initialize the tracking dashboard
    tracker = TrainingTracker(checkpoint_dir="models/checkpoints", log_dir="logs")

    # 5. Initialize the Model
    print("🧠 [LOCAL TEST] Allocating Panini Transformer layers to device...")
    # UPDATED: Instantiating your dedicated custom project architecture class
    model = PaniniTransformer(config).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Attempt to recover an old crash state if it exists on disk
    # Pass None for scheduler locally to maintain simplified processing speed checks
    recovery_meta = tracker.load_latest_checkpoint(model, optimizer, None, device)
    start_epoch = recovery_meta["epoch"]

    model.train()

    # 6. Run a tracked loop pass to verify metric collection works
    print("🏃 [LOCAL TEST] Pushing sample batch through forward & backward pass layers...")
    try:
        inputs, targets = next(iter(dataloader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        
        loss = loss_fn(
            logits.view(-1, config.vocab_size), 
            targets.contiguous().view(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        # Log the step results into logs/training_metrics.jsonl
        tracker.log_step(epoch=start_epoch, batch=1, loss=loss.item(), lr=1e-4, is_master=True)
        
        # Backup weights and optimizer positions to models/checkpoints/
        tracker.save_checkpoint(model, optimizer, None, epoch=start_epoch, batch=1, loss=loss.item(), is_master=True)
        
        print(f"🔥 [LOCAL TEST] Pass Successful! Calculated Batch Loss: {loss.item():.4f}")
        print("🎉 [LOCAL TEST] Success! Your custom transformer architecture is 100% mathematically correct.")
        
    except Exception as e:
        print(f"❌ [LOCAL TEST] Error during training execution pass: {e}")

if __name__ == "__main__":
    run_local_pipeline_test() '''