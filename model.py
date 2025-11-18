import torch
import torch.nn as nn
import torch.nn.functional as F


class GRULM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, num_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, input_ids, target_ids=None, lambda_ortho=1e-3):
        pass


class GrassmannianLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,  # context embedding dim
        n: int = 1024,  # ambient dimension for Grassmannian
        k: int = 256,  # subspace dimension
    ):
        super().__init__()
        assert k <= n, "Subspace dimension k must be <= n"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n = n
        self.k = k

        # 1) Context encoder: token embeddings -> mean pooled context vector
        self.token_inp_emb = nn.Embedding(vocab_size, d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # 2) Map context vector -> A \in R^{n x k} (will approximate a basis for subspace)
        self.ctx_to_A = nn.Linear(d_model, n * k)

        # 3) Token embeddings in the Grassmannian ambient space R^n
        self.token_geom_emb = nn.Embedding(vocab_size, n)

    def forward(self, input_ids, target_ids=None, lambda_ortho=1e-3):
        """
        input_ids: (batch_size, seq_len)  - context tokens
        target_ids: (batch_size,)         - next-token targets (optional, for training)

        Returns:
            logits: (batch_size, vocab_size)
            loss: scalar (if target_ids provided), else None
        """
        B, L = input_ids.shape

        # Encode context
        # (B, L, d_model)
        inp_emb = self.token_inp_emb(input_ids)
        output, h_n = self.gru(inp_emb)
        ctx_vec = h_n[-1]

        # Map context -> A (approx Grassmannian point)
        # (B, n*k)
        A_flat = self.ctx_to_A(ctx_vec)
        # (B, n, k)
        A = A_flat.view(B, self.n, self.k)

        # use A directly and add an orthogonality penalty
        U = A  # (B, n, k)

        # Build projection matrices P = U U^T
        # (B, n, n)
        P = torch.bmm(U, U.transpose(1, 2))

        # Score each vocab token by how much it lives in the subspace
        # token_geom_emb.weight: (V, n)
        E = self.token_geom_emb.weight  # (V, n)

        # We want scores s_{b,v} = e_v^T P_b e_v
        # Compute tmp_{b,v,m} = (P_b e_v)_m then dot with e_v
        # P: (B, n, n), E: (V, n)
        # tmp: (B, V, n)
        tmp = torch.einsum("bmn,vn->bvm", P, E)
        # scores: (B, V)
        scores = (tmp * E.unsqueeze(0)).sum(dim=-1)

        logits = scores  # (B, V)

        loss = None
        if target_ids is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(logits, target_ids)

            # Orthogonality penalty: encourage U^T U ≈ I_k
            # U^T U: (B, k, k)
            UtU = torch.bmm(U.transpose(1, 2), U)
            I = torch.eye(self.k, device=U.device).unsqueeze(0).expand(B, -1, -1)
            ortho_loss = ((UtU - I) ** 2).mean()

            loss = ce_loss + lambda_ortho * ortho_loss

        return logits, loss


def save_grassmannian_model(
    model, optimizer, itos, stoi, config, path="grassmannian_model.pt"
):
    """
    Saves:
      - model state dict
      - optimizer state dict
      - vocab (itos, stoi)
      - config dict (hyperparameters)
    """
    save_dict = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "itos": itos,
        "stoi": stoi,
        "config": config,
    }
    torch.save(save_dict, path)
    print(f"Saved Grassmannian model to {path}")


def load_grassmannian_model(path, model_class):
    """
    Loads a saved Grassmannian language model.

    Args:
        path: path to the .pt file saved earlier
        model_class: the class definition of GrassmannianLanguageModel

    Returns:
        model, optimizer_state_dict, itos, stoi, config
    """
    checkpoint = torch.load(path, map_location="cpu")

    config = checkpoint["config"]

    # Reconstruct model from config
    model = model_class(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n=config["n"],
        k=config["k"],
    )

    model.load_state_dict(checkpoint["model_state"])

    optimizer_state = checkpoint["optimizer_state"]
    itos = checkpoint["itos"]
    stoi = checkpoint["stoi"]

    print(f"Loaded Grassmannian model from {path}")

    return model, optimizer_state, itos, stoi, config


#########


def sample_from_logits(logits, temperature=1.0, top_k=None):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature

    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, k=top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask[indices] = values
        logits = mask

    probs = F.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()
    return token_id


def generate_text_bpe(
    model,
    tokenizer,
    prompt: str,
    seq_len: int = 32,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str | torch.device = "cpu",
) -> str:
    model.eval()
    device = torch.device(device)

    # encode prompt
    prompt_ids = tokenizer.encode(prompt).ids
    if len(prompt_ids) == 0:
        prompt_ids = [tokenizer.token_to_id("[UNK]")]

    generated_ids = prompt_ids[:]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = generated_ids[-seq_len:]
            if len(context_ids) < seq_len:
                pad_id = tokenizer.token_to_id("[PAD]")
                context_ids = [pad_id] * (seq_len - len(context_ids)) + context_ids

            x = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)

            logits, _ = model(x, target_ids=None)
            logits = logits[0]

            next_id = sample_from_logits(logits, temperature=temperature, top_k=top_k)
            generated_ids.append(next_id)

    text = tokenizer.decode(generated_ids)
    return text
