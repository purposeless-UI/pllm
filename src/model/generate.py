import os
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import re

# Resolve the absolute path of the project root directory ('pllm') and inject it into sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))               # src/model
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))           # pllm root folder
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.model import PaniniConfig, PaniniTransformer
from src.utils.visual_brain import NeuralBrainVisualizer


class PaniniGenerator:
    """
    Handles autoregressive text generation for the custom Panini architecture.
    """
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config: PaniniConfig,
        device: torch.device,
        visualize: bool = True,
    ):
        self.device = device

        # Retain your configuration as-is; it must match the checkpoint being loaded.
        self.config = config

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"❌ Tokenizer configuration missing at: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Optional visualizer, safe on headless environments
        self.visualizer = None
        if visualize:
            try:
                self.visualizer = NeuralBrainVisualizer(width=1000, height=600, active=True)
            except Exception as e:
                print(f"⚠️ [VISUALIZER] Could not start graphic canvas ({e}). Continuing without it.")
                self.visualizer = None

        print("🧠 Allocating Panini Transformer weights into inference memory fields...")
        self.model = PaniniTransformer(self.config)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)

            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                self.model.load_state_dict(state_dict["model_state_dict"])
            else:
                self.model.load_state_dict(state_dict)
            print(f"✅ Successfully loaded model weights from: {model_path}")
        else:
            print(f"⚠️ Warning: Target file '{model_path}' not found. Running with random initialization.")

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 50, 
        temperature: float = 0.85, 
        top_k: int = 50,
        repetition_penalty: float = 1.2  # 👈 Added penalty parameter (values between 1.1 and 1.3 work best)
    ) -> str:
        if not prompt or not prompt.strip():
            return "⚠️ Prompt cannot be empty."

        try:
            encoded = self.tokenizer.encode(prompt)
            prompt_token_ids = encoded.ids if hasattr(encoded, "ids") else encoded
        except Exception as e:
            print(f"⚠️ Tokenizer encoding error: {str(e)}")
            return "❌ Tokenizer parsing pipeline breakdown."

        bos_id = self.tokenizer.token_to_id("[BOS]")
        eos_id = self.tokenizer.token_to_id("[EOS]")

        if bos_id is not None and (len(prompt_token_ids) == 0 or prompt_token_ids[0] != bos_id):
            prompt_token_ids = [bos_id] + prompt_token_ids

        input_tensor = torch.tensor([prompt_token_ids], dtype=torch.long, device=self.device)
        new_generated_ids = []

        for _ in range(max_new_tokens):
            if input_tensor.shape[1] > self.config.max_seq_len:
                input_tensor = input_tensor[:, -self.config.max_seq_len:]

            logits = self.model(input_tensor)
            next_token_logits = logits[:, -1, :].clone()  # Clone to alter probabilities safely

            # 🛠️ APPLY REPETITION PENALTY
            # Lowers the probability scores of tokens that have already been generated in this session
            if repetition_penalty != 1.0 and len(new_generated_ids) > 0:
                for token_id in set(new_generated_ids):
                    # If logit is positive, divide it to reduce confidence. If negative, multiply it to push it lower.
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_tensor = torch.cat((input_tensor, next_token), dim=1)

            token_val = next_token.item()
            new_generated_ids.append(token_val)

            if self.visualizer is not None:
                try:
                    token_str = self.tokenizer.id_to_token(token_val)
                    if token_str:
                        self.visualizer.inject_live_token(token_str, phase="INFERENCE")
                except Exception:
                    pass

            if eos_id is not None and token_val == eos_id:
                break

        # --- NATIVE DECODER SAFELANE ---
        try:
            final_text = self.tokenizer.decode(new_generated_ids)
        except Exception as e:
            print(f"⚠️ Tokenizer decoding error: {str(e)}")
            return "❌ Tokenizer rendering pipeline breakdown."

        for special_tag in ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]:
            final_text = final_text.replace(special_tag, "")

        final_text = re.sub(r'(matrices\.\.\.|output\s*vector|vector\s*matrices\.\.\.)', '', final_text, flags=re.IGNORECASE)

        final_text = re.sub(r'\s+([्ािीुूृेैोौंः])', r'\1', final_text)
        final_text = re.sub(r'\s+', ' ', final_text)
        final_text = re.sub(r'\s+([।॥])', r'\1', final_text)

        return final_text.strip()
