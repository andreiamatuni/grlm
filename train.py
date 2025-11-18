import argparse
import os.path as osp
import random
from collections import Counter

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, Dataset

from data import BPEWikiTextWindowDataset
from model import (
    GrassmannianLanguageModel,
    save_checkpoint,
)

parser = argparse.ArgumentParser(description="Train a Grassmannian language model")
parser.add_argument(
    "--num_epochs", type=int, default=5, help="number of epochs to train"
)
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--seq_len", type=int, default=16, help="sequence length")
parser.add_argument(
    "--max_samples", type=int, default=100000, help="maximum number of samples"
)
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument(
    "--save_path", type=str, default="model.pt", help="path to save model"
)
parser.add_argument(
    "--n", type=int, default=512, help="ambient dimension of the Grassmannian"
)
parser.add_argument(
    "--k", type=int, default=128, help="dimension of the Grassmannian subspace"
)
parser.add_argument("--d_model", type=int, default=256, help="embedding dimension")
args = parser.parse_args()


tokenizer = Tokenizer.from_file("data/tokenizers/bpe_tokenizer_wt103.json")
vocab_size = tokenizer.get_vocab_size()

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_texts = dataset["train"]["text"]

train_dataset = BPEWikiTextWindowDataset(
    train_texts, tokenizer, seq_len=args.seq_len, max_samples=args.max_samples
)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


config = {
    "vocab_size": vocab_size,
    "d_model": args.d_model,
    "n": args.n,
    "k": args.k,
    "seq_len": args.seq_len,
}

print("config: ", config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: ", device)

model = GrassmannianLanguageModel(
    vocab_size=vocab_size, d_model=args.d_model, n=args.n, k=args.k
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


for epoch in range(args.num_epochs):
    model.train()
    total_loss = 0.0
    for step, (input_ids, target_ids) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, loss = model(input_ids, target_ids, lambda_ortho=1e-3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / 50
            print(f"Epoch {epoch + 1}, step {step + 1}, avg loss {avg_loss:.4f}")
            total_loss = 0.0

save_checkpoint(
    save_dir="checkpoints/grlm",
    model=model,
    optimizer=optimizer,
    tokenizer=tokenizer,
    config=config,
)

# save_grassmannian_model(
#     model=model,
#     optimizer=optimizer,
#     config=config,
#     path="checkpoints/grass_model.pt",
# )

# model, optimizer_state, itos, stoi, config = load_grassmannian_model(
#     osp.join("checkpoints", args.save_path), model_class=GrassmannianLanguageModel
# )
