import torch
import torch.nn as nn

class HyperbolicDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(HyperbolicDecoderLayer, self).__init__()
        self.self_attn = HyperbolicMultiHeadAttention(d_model, num_heads)
        self.cross_attn = HyperbolicMultiHeadAttention(d_model, num_heads)
        self.ffn = HyperbolicFFN(d_model, d_ff)
        self.norm1 = HyperbolicLayerNorm(d_model)
        self.norm2 = HyperbolicLayerNorm(d_model)
        self.norm3 = HyperbolicLayerNorm(d_model)

    def forward(self, x, encoder_output):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output)
        x = self.norm2(x + cross_attn_output)
        
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x

# Example usage
d_model = 64
num_heads = 8
d_ff = 256
batch_size = 2
seq_len = 10

decoder_layer = HyperbolicDecoderLayer(d_model, num_heads, d_ff)
x = torch.randn(batch_size, seq_len, d_model)
encoder_output = torch.randn(batch_size, seq_len, d_model)
output = decoder_layer(x, encoder_output)

print(f"Input shape: {x.shape}")
print(f"Encoder output shape: {encoder_output.shape}")
print(f"Output shape: {output.shape}")
