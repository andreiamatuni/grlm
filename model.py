import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # ----- 1) Encode context -----
        # (B, L, d_model)
        inp_emb = self.token_inp_emb(input_ids)
        # mean-pool over sequence: (B, d_model)
        ctx_vec = inp_emb.mean(dim=1)

        # ----- 2) Map context -> A (approx Grassmannian point) -----
        # (B, n*k)
        A_flat = self.ctx_to_A(ctx_vec)
        # (B, n, k)
        A = A_flat.view(B, self.n, self.k)

        # Option 1 (simple): use A directly and add an orthogonality penalty
        U = A  # (B, n, k)

        # If you prefer strict orthonormal columns, you could do QR per batch:
        # U_list = []
        # for b in range(B):
        #     q, r = torch.linalg.qr(A[b])
        #     U_list.append(q)
        # U = torch.stack(U_list, dim=0)  # (B, n, k)

        # ----- 3) Build projection matrices P = U U^T -----
        # (B, n, n)
        P = torch.bmm(U, U.transpose(1, 2))

        # ----- 4) Score each vocab token by how much it lives in the subspace -----
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


def sample_from_logits(logits, temperature=1.0, top_k=None):
    """
    logits: (vocab_size,)
    returns: int token_id
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        # keep only top_k tokens
        values, indices = torch.topk(logits, k=top_k)
        # set others to -inf
        mask = torch.full_like(logits, float("-inf"))
        mask[indices] = values
        logits = mask

    probs = F.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()
    return token_id


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


def generate_text(
    model,
    prompt: str,
    stoi,
    itos,
    tokenize,
    encode,
    seq_len: int = 16,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str | torch.device = "cpu",
) -> str:
    """
    Autoregressive word-level generation using the GrassmannianLanguageModel.
    """
    model.eval()
    device = torch.device(device)

    # 1) Tokenize & encode prompt
    prompt_tokens = tokenize(prompt)
    prompt_ids = encode(prompt_tokens)

    # if empty, seed with <unk> or similar
    if len(prompt_ids) == 0:
        prompt_ids = [stoi.get("<unk>", 0)]

    # work in a mutable list
    generated_ids = prompt_ids[:]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # take the last seq_len tokens as context
            context_ids = generated_ids[-seq_len:]
            # pad on the left if needed
            if len(context_ids) < seq_len:
                pad_id = stoi.get("<pad>", 0)
                context_ids = [pad_id] * (seq_len - len(context_ids)) + context_ids

            x = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(
                0
            )  # (1, seq_len)

            logits, _ = model(x, target_ids=None)
            # logits: (1, vocab_size)
            logits = logits[0]  # (vocab_size,)

            next_id = sample_from_logits(logits, temperature=temperature, top_k=top_k)

            generated_ids.append(next_id)

    # Decode tokens back to text (skip leading pads if any)
    # Here we just join with spaces; you can fancy this up if you like.
    words = [itos[t] for t in generated_ids]
    text = " ".join(words)
    return text


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
