import os
import re
import sys
import numpy as np
from tokenizers import Tokenizer

# Add the project root directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean_scripture_text(text_data: str) -> str:
    """
    Strips textbook-style formatting markup that isn't real Sanskrit content, so the
    model doesn't learn to reproduce it verbatim.

    BUG FIX: the raw source file contains section dividers/headers like:
        ==========================
        लट् लकारः
        ==========================
    These were previously tokenized as-is along with everything else, so the (very
    small, easily-overfit) model learned them as valid continuations and would
    regurgitate them verbatim during generation (e.g. "==========================",
    "व्यवहारिकसंवादाः"). Removing them here means future retraining runs won't learn
    this artifact. Note: this alone does NOT fix an already-trained checkpoint —
    data needs to be reprocessed and the model retrained for this to take effect.
    """
    # Remove "divider \n title \n divider" blocks (section headings between rule lines)
    text_data = re.sub(r'={5,}\s*\n.*?\n={5,}\s*\n?', '', text_data)

    # Remove any remaining standalone divider lines (======...)
    text_data = re.sub(r'^\s*={5,}\s*$\n?', '', text_data, flags=re.MULTILINE)

    # Collapse the resulting excess blank lines left behind
    text_data = re.sub(r'\n{3,}', '\n\n', text_data)

    return text_data.strip()


def pre_tokenize_scripture():
    print("🧹 Starting text pre-processing and binary token packing...")

    raw_file = "data/raw/sample_sanskrit_scripture.txt"
    tokenizer_path = "tokenizer.json"
    output_bin_file = "data/processed/sanskrit_tokens.npy"

    # 1. Verification Guardrails
    if not os.path.exists(raw_file):
        print(f"❌ Error: Cannot find raw source file at {raw_file}")
        return
    if not os.path.exists(tokenizer_path):
        print(f"❌ Error: Missing {tokenizer_path}. Run the tokenizer script first.")
        return

    # 2. Initialize the generated codebook
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 3. Read the text asset
    with open(raw_file, "r", encoding="utf-8") as f:
        text_data = f.read()

    # 4. Strip textbook markup (section dividers/headings) before tokenizing
    original_len = len(text_data)
    text_data = clean_scripture_text(text_data)
    print(f"🧽 Cleaned raw text: {original_len} -> {len(text_data)} characters "
          f"({original_len - len(text_data)} chars of markup removed)")

    print("🔢 Converting text matrix into numerical IDs...")
    token_ids = tokenizer.encode(text_data).ids

    # 5. Pack into an ultra-fast NumPy 16-bit binary matrix array
    # uint16 handles vocabulary numbers up to 65,535 cleanly, keeping files small
    token_array = np.array(token_ids, dtype=np.uint16)

    # 6. Save directly into your processed folder layout
    os.makedirs(os.path.dirname(output_bin_file), exist_ok=True)
    np.save(output_bin_file, token_array)

    print(f"💾 Success! Compressed binary saved to: {output_bin_file}")
    print(f"📦 Total tokens packed: {len(token_array)}")


if __name__ == "__main__":
    pre_tokenize_scripture()