import os
import json
import matplotlib.pyplot as plt

def generate_training_plot():
    log_file = "logs/training_metrics.jsonl"
    output_image = "logs/loss_curve.png"
    
    if not os.path.exists(log_file):
        print(f"❌ Error: Cannot find log file at '{log_file}'. Run training first!")
        return
        
    batches = []
    losses = []
    lrs = []
    
    # Read the JSON Lines records row by row
    with open(log_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            batches.append(idx)
            losses.append(data["loss"])
            lrs.append(data["learning_rate"])
            
    if not losses:
        print("⚠️ Log file is empty. No steps recorded yet.")
        return

    # Initialize a dual-axis metric chart layout
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot 1: Loss values (Left Axis)
    color = "tab:red"
    ax1.set_xlabel("Training Optimization Steps")
    ax1.set_ylabel("Cross Entropy Loss", color=color)
    ax1.plot(batches, losses, color=color, alpha=0.6, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Cosine Learning Rate Warmup values (Right Axis)
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Learning Rate Scale", color=color)
    ax2.plot(batches, lrs, color=color, linestyle=":", linewidth=2, label="LR Schedule")
    ax2.tick_params(axis="y", labelcolor=color)

    # UPDATED: Rebranded the title text block to feature your custom project identity
    fig.suptitle("pLLM Sanskrit Panini Core Architecture Optimization Metrics", fontsize=12)
    fig.tight_layout()
    
    # Save chart image directly into logs/ folder layout
    plt.savefig(output_image, dpi=150)
    print(f"📈 [VISUAL] Metrics analyzed successfully. Training curve saved to: {output_image}")

if __name__ == "__main__":
    generate_training_plot()
