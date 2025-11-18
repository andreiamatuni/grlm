from tokenizers import Tokenizer

from model import GrassmannianLanguageModel, generate_text_bpe, load_grassmannian_model


# Simple word-level tokenization
def tokenize(text):
    return text.strip().split()


def encode(tokens):
    return [stoi.get(t, unk_id) for t in tokens]


model, optimizer_state, itos, stoi, config = load_grassmannian_model(
    "checkpoints/grass_model.pt", model_class=GrassmannianLanguageModel
)

unk_id = stoi["<unk>"]

seq_len = 16

device = "cpu"

tokenizer = Tokenizer.from_file("data/tokenizers/bpe_tokenizer_wt103.json")
vocab_size = tokenizer.get_vocab_size()

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
