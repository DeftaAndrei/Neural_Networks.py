import torch
import torch.nn as nn

class HyperbolicLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(HyperbolicLayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Transformare din spațiul Poincare în Lorentz
        x_tang = self.poincare_to_lorentz(x)
        
        # Calcularea mediei și deviației standard
        mean = x_tang.mean(dim=-1, keepdim=True)
        std = x_tang.std(dim=-1, keepdim=True)
        
        # Normalizarea
        x_norm = (x_tang - mean) / (std + self.eps)
        
        # Aplicarea parametrilor trainabili
        x_norm = self.a * x_norm + self.b
        
        # Reîntoarcerea la spațiul Poincare
        return self.lorentz_to_poincare(x_norm)
    
    def poincare_to_lorentz(self, x):
        x_sq_norm = torch.sum(x**2, dim=-1, keepdim=True)
        return 2 * x / (1 - x_sq_norm)
    
    def lorentz_to_poincare(self, x):
        return x / (1 + torch.sqrt(1 + torch.sum(x**2, dim=-1, keepdim=True)))

# Dimensiunea modelului
d_model = 64
batch_size = 2
seq_len = 10

# Inițializarea stratului de normalizare
hyp_layer_norm = HyperbolicLayerNorm(d_model)

# Tenzor de intrare
x = torch.randn(batch_size, seq_len, d_model)

# Aplicarea normalizării
x_norm = hyp_layer_norm(x)

# Afișarea rezultatelor
print(f"Input shape: {x.shape}")
print(f"Output shape: {x_norm.shape}")
print(f"Input norm: {torch.norm(x[0, 0])}")
print(f"Output norm: {torch.norm(x_norm[0, 0])}")
