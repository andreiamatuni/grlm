import torch
from tokenizers import Tokenizer

from model import (
    GrassmannianLanguageModel,
    generate_text_bpe,
    load_checkpoint,
)

tokenizer, model, optimizer_state, config = load_checkpoint(
    load_dir="checkpoints/grlm",
    model_class=GrassmannianLanguageModel,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

seq_len = 16
device = "cpu"

prompt = "The history of natural language models"
generated = generate_text_bpe(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    seq_len=seq_len,
    max_new_tokens=50,
    temperature=0.9,
    top_k=40,
    device=device,
)

print(generated)
