import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperbolicMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(HyperbolicMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # 1. Proiecții liniare
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. Reshape pentru multiple capete
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Calculul atenției hiperbolice
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        dist = torch.acosh(torch.clamp(torch.einsum('bhqd,bhkd->bhqk', Q_norm, K_norm), min=1.0))
        dist_weights = torch.exp(-dist)

        # 4. Aplicarea ponderilor atenției
        output = torch.einsum('bhqk,bhkd->bhqd', dist_weights, V)

        # 5. Reconstrucția formei originale
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(output)

# Testare cod
d_model = 64
num_heads = 8
batch_size = 4
seq_len = 10

hyp_mha = HyperbolicMultiheadAttention(d_model, num_heads)
x = torch.randn(batch_size, seq_len, d_model)
output = hyp_mha(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
