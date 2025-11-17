import argparse
import os.path as osp
import random
from collections import Counter

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from model import (
    GrassmannianLanguageModel,
    # generate_text,
    load_grassmannian_model,
    save_grassmannian_model,
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

# Load WikiText-103 (raw)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_texts = dataset["train"]["text"]


# Simple word-level tokenization
def tokenize(text):
    return text.strip().split()


# Build vocab from training texts
counter = Counter()
for line in train_texts:
    tokens = tokenize(line)
    counter.update(tokens)


# Keep top N words to keep it small
max_vocab_size = 10000
most_common = counter.most_common(max_vocab_size - 2)  # reserve for <pad>, <unk>

itos = ["<pad>", "<unk>"] + [w for w, _ in most_common]
stoi = {w: i for i, w in enumerate(itos)}

pad_id = stoi["<pad>"]
unk_id = stoi["<unk>"]
vocab_size = len(itos)
print("Vocab size:", vocab_size)


def encode(tokens):
    return [stoi.get(t, unk_id) for t in tokens]


class WikiTextWindowDataset(Dataset):
    def __init__(self, texts, seq_len=16, max_samples=50000):
        self.seq_len = seq_len
        self.samples = []

        for line in texts:
            tokens = tokenize(line)
            ids = encode(tokens)
            if len(ids) < seq_len + 1:
                continue
            # sliding windows within this line
            for i in range(len(ids) - seq_len):
                x = ids[i : i + seq_len]
                y = ids[i + seq_len]
                self.samples.append((x, y))
                if len(self.samples) >= max_samples:
                    break
            if len(self.samples) >= max_samples:
                break

        print(f"Created {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


train_dataset = WikiTextWindowDataset(
    train_texts, seq_len=args.seq_len, max_samples=args.max_samples
)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

config = {
    "vocab_size": vocab_size,
    "d_model": args.d_model,
    "n": args.n,
    "k": args.k,
    "seq_len": args.seq_len,
}

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

save_grassmannian_model(
    model=model,
    optimizer=optimizer,
    itos=itos,
    stoi=stoi,
    config=config,
    path="checkpoints/grass_model.pt",
)

model, optimizer_state, itos, stoi, config = load_grassmannian_model(
    osp.join("checkpoints", args.save_path), model_class=GrassmannianLanguageModel
)
