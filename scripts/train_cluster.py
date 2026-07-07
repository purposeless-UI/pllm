import os
import sys
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Imported the PyTorch native automatic mixed precision engine
from torch.amp import autocast, GradScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pointing to your custom project architecture definitions directly
from src.model.model import PaniniConfig, PaniniTransformer
from src.dataset_pipeline.dataset import PaniniStreamingDataset
from src.utils.tracker import TrainingTracker

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def run_cluster_training():
    local_rank = setup_distributed()
    is_master = (local_rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    
    # Swapped to your custom setup parameter constructor
    config = PaniniConfig(
        vocab_size=64000,    
        dim=1024,            
        n_layers=12,         
        n_heads=16,
        n_kv_heads=4,        
        max_seq_len=2048     
    )

    try:
        dataset = PaniniStreamingDataset(
            raw_data_dir="data/raw",
            tokenizer_path="tokenizer.json",
            seq_len=config.max_seq_len
        )
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset, 
            batch_size=4, # Lower batch size per GPU to avoid Out-Of-Memory errors
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
    except Exception as e:
        if is_master: print(f"❌ [CLUSTER] Dataset Error: {e}")
        cleanup_distributed()
        return

    # Instantiating your dedicated custom project architecture class
    model = PaniniTransformer(config).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # --- UPGRADE 1: Initialize System Tracker & Scaler for 16-bit Mixed Precision ---
    tracker = TrainingTracker(checkpoint_dir="models/checkpoints", log_dir="logs")
    scaler = GradScaler(device="cuda") # Prevents small gradients from underflowing to 0

    # --- UPGRADE 2: Gradient Accumulation Settings ---
    accumulation_steps = 4 # Updates weights once every 4 batches (Simulates batch size 16)

    epochs = 5
    model.train()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # --- UPGRADE 3: Autocast Mixed Precision Forward Pass ---
            # Runs calculations in rapid float16/bfloat16 instead of heavy float32
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = loss_fn(
                    logits.view(-1, config.vocab_size), 
                    targets.contiguous().view(-1)
                )
                # Scale the loss down based on our accumulation step requirements
                loss = loss / accumulation_steps

            # Scale gradients and run backward pass safely
            scaler.scale(loss).backward()
            
            # Trigger weight updates only after accumulating enough steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Unscale gradients before clipping them to check for overflows
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step using scaled precision
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Document real-time metrics through our unified tracker file
                if is_master:
                    current_raw_loss = loss.item() * accumulation_steps
                    tracker.log_step(epoch=epoch, batch=batch_idx, loss=current_raw_loss, lr=2e-4, is_master=True)
                    
                    if batch_idx % 20 == 0:
                        print(f"   [Epoch {epoch+1}] Batch {batch_idx:03d} | Optimized Loss: {current_raw_loss:.4f}")
                        # FIXED: Added None in the 3rd position to match your tracker's expected layout arguments
                        tracker.save_checkpoint(model, optimizer, None, epoch, batch_idx, current_raw_loss, is_master=True)

            total_loss += loss.item() * accumulation_steps
        
        if is_master:
            avg_loss = total_loss / len(dataloader)
            print(f"📉 [CLUSTER] Epoch {epoch+1} Complete. Average Combined Loss: {avg_loss:.4f}")

    if is_master:
        # FIX: Ensure destination directory exists and save state_dict from the raw module (.module) to unwrap DDP
        os.makedirs("models", exist_ok=True)
        final_save_path = "models/panini_sanskrit_final.pt"
        
        # Saving state dict wrapped inside a dictionary matching what generate.py and eval_accuracy.py expect
        torch.save({"model_state_dict": model.module.state_dict()}, final_save_path)

        print(f"💾 [CLUSTER] Master training complete! Global weights secured at: {final_save_path}")

    cleanup_distributed()

if __name__ == "__main__":
    run_cluster_training()
