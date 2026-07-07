import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_panini_tokenizer():
    print("🪙 Initializing pLLM Automated Sanskrit Tokenizer Engine...")
    
    # Matching your exact file system architecture
    raw_data_dir = "data/raw"
    save_path = "tokenizer.json"
    
    if not os.path.exists(raw_data_dir):
        print(f"❌ Error: The folder '{raw_data_dir}' does not exist in your current terminal view!")
        return

    text_files = [
        os.path.join(raw_data_dir, f) 
        for f in os.listdir(raw_data_dir) 
        if f.endswith(".txt")
    ]
    
    if not text_files:
        print(f"❌ Error: No text files found inside '{raw_data_dir}'. Make sure your sample_sanskrit_scripture.txt file is not empty!")
        return
        
    # Initialize a clean Byte-Pair Encoding model shell
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure the trainer with special token guardrails matching Llama 3
    trainer = BpeTrainer(
        vocab_size=64000,
        special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]", "[MASK]"],
        show_progress=True
    )
    
    print(f"🔄 Training automated vocabulary across: {text_files}")
    tokenizer.train(text_files, trainer)
    
    # Save the completed vocabulary codebook to your Desktop root folder
    tokenizer.save(save_path)
    print(f"✅ Success! Your tokenizer file was generated at: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    train_panini_tokenizer()
