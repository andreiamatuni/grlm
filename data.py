import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset


def setup_bpe():
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_texts = dataset["train"]["text"]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    trainer = BpeTrainer(
        vocab_size=16000,
        min_frequency=2,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(batch_iterator(train_texts), trainer=trainer)

    print("Vocab size:", tokenizer.get_vocab_size())
    pad_id = tokenizer.token_to_id("[PAD]")
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    unk_id = tokenizer.token_to_id("[UNK]")

    # Optional: save tokenizer
    tokenizer.save("data/tokenizers/bpe_tokenizer_wt103.json")


def batch_iterator(train_texts):
    # You *could* subsample here if you want faster training of tokenizer
    for line in train_texts:
        line = line.strip()
        if line:
            yield line


class BPEWikiTextWindowDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=32, max_samples=200_000):
        self.seq_len = seq_len
        self.samples = []
        self.tokenizer = tokenizer

        for line in texts:
            line = line.strip()
            if not line:
                continue

            ids = tokenizer.encode(line).ids

            if len(ids) < seq_len + 1:
                continue

            for i in range(len(ids) - seq_len):
                x = ids[i : i + seq_len]
                y = ids[i + seq_len]
                self.samples.append((x, y))
                if len(self.samples) >= max_samples:
                    break
            if len(self.samples) >= max_samples:
                break

        print(f"Created {len(self.samples)} samples with BPE on WikiText-103.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


if __name__ == "__main__":
    setup_bpe()
