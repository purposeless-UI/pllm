import os
import json
import torch
import time
from typing import Dict, Any

class TrainingTracker:
    """
    Handles training metrics tracking, system logging, and fault-tolerant 
    model weight checkpoint saving/loading for local and cluster environments.
    """
    def __init__(self, checkpoint_dir: str = "models/checkpoints", log_dir: str = "logs"):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.start_time = time.time()
        
        # Ensure workspace structural directories exist safely
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, "training_metrics.jsonl")

    def log_step(self, epoch: int, batch: int, loss: float, lr: float, is_master: bool = True):
        """Logs real-time processing performance metrics to a JSON lines file."""
        if not is_master:
            return
            
        elapsed = time.time() - self.start_time
        metrics = {
            "epoch": epoch,
            "batch": batch,
            "loss": round(loss, 4),
            "learning_rate": float(f"{lr:.8f}"), # Formats scientific learning rate notation cleanly
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Append historical records in structured JSONL format
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def save_checkpoint(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: Any,                  # Added to track scheduler positions during recovery
        epoch: int, 
        batch: int, 
        loss: float,
        is_master: bool = True
    ):
        """Saves weights, optimization gradients, and metadata for crash-recovery."""
        if not is_master:
            return
            
        # Extract model from distributed wrapper container if DDP is active
        model_to_save = model.module if hasattr(model, "module") else model
        
        checkpoint_state = {
            "epoch": epoch,
            "batch": batch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "loss": loss,
        }
        
        rolling_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint_state, rolling_path)

    def load_latest_checkpoint(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: Any,
        device: torch.device
    ) -> Dict[str, Any]:
        """Scans recovery directories and reloads the latest state if found."""
        rolling_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        
        if not os.path.exists(rolling_path):
            print("💡 [TRACKER] No existing checkpoint history discovered. Starting clean execution.")
            return {"epoch": 0, "batch": 0, "loss": float("inf")}
            
        print(f"🔄 [TRACKER] Recovery target detected. Restoring execution states from: {rolling_path}")
        checkpoint = torch.load(rolling_path, map_location=device)
        
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        print(f"✅ [TRACKER] Resuming training from Epoch {checkpoint['epoch'] + 1}, Batch {checkpoint['batch']}.")
        return checkpoint
