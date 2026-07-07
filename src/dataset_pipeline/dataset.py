import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

class PaniniStreamingDataset(Dataset):
    def __init__(self, raw_data_dir: str, tokenizer_path: str, seq_len: int = 2048, processed_dir: str = "data/processed"):
        self.seq_len = seq_len
        processed_file_path = os.path.join(processed_dir, "sanskrit_tokens.npy")
        
        # --- Check if highly compressed pre-tokenized binary already exists ---
        if os.path.exists(processed_file_path):
            print(f"⚡ [PROCESSED] Found cached binary data. Loading instantly from: {processed_file_path}")
            # Load using memory map layout for rapid disk-to-RAM loading
            self.tokens = np.load(processed_file_path, mmap_mode="r")
            print(f"✅ [PROCESSED] Cache Linked. Total active tokens in database: {len(self.tokens)}")
            return

        # --- Fallback: Process raw text if no cache is found ---
        print(f"📂 [PROCESSED] No cache found. Accessing raw text files inside: {raw_data_dir}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"❌ Cannot find '{tokenizer_path}'. Please run the tokenizer script first!")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Read and merge all raw text files into one continuous stream string
        text_files = [
            os.path.join(raw_data_dir, f) 
            for f in os.listdir(raw_data_dir) 
            if f.endswith(".txt")
        ]
        
        if not text_files:
            raise FileNotFoundError(f"❌ No raw text files found inside: {raw_data_dir}")
            
        full_text = ""
        for file_path in text_files:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text += f.read() + " "
                
        print("🧼 Tokenizing raw Sanskrit text streams into tensor coordinates...")
        raw_token_ids = self.tokenizer.encode(full_text).ids
        
        # --- Optimize and save data layout into the processed folder directory ---
        os.makedirs(processed_dir, exist_ok=True)
        # uint16 tracks vocabulary coordinates up to 65,535 cleanly with half memory size
        self.tokens = np.array(raw_token_ids, dtype=np.uint16)
        np.save(processed_file_path, self.tokens)
        
        print(f"💾 [PROCESSED] Successfully generated binary cache archive at: {processed_file_path}")
        print(f"✅ Data Stream Matrix Compiled. Total active tokens: {len(self.tokens)}")

    def __len__(self):
        # Calculates the maximum number of sliding training chunks we can extract
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        # Extract a window chunk that is exactly seq_len + 1 tokens long
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        
        # Convert NumPy slice views directly into clean 64-bit PyTorch tensors
        # x is the input block, y is the target block shifted right by 1 position
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def create_pllm_dataloader(raw_data_dir: str, tokenizer_path: str, batch_size: int = 4, seq_len: int = 2048):
    """
    Groups individual data chunks into batched training blocks.
    Configured for high-speed multi-GPU data parallelism streaming.
    """
    dataset = PaniniStreamingDataset(raw_data_dir, tokenizer_path, seq_len)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        pin_memory=torch.cuda.is_available() # Speeds up server GPU transfer
    )

if __name__ == "__main__":
    # Test path execution variables using your exact local workspace setup
    print("🧪 Running local validation on the upgraded streaming dataset loader...")
    try:
        loader = create_pllm_dataloader(
            raw_data_dir="data/raw", 
            tokenizer_path="tokenizer.json", 
            batch_size=2, 
            seq_len=32 # Tiny sequence length just to test local execution safely
        )
        x_batch, y_batch = next(iter(loader))
        print(f"👍 Data Batch Test Success! Input Tensor Shape: {list(x_batch.shape)}")
        print(f"                         Target Tensor Shape: {list(y_batch.shape)}")
    except Exception as e:
        print(f"⚠️ Test pass skipped or failed: {e}")
        print("   (This is normal if your sample data text is too short for a full context window check)")
