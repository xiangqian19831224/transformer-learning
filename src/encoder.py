import torch
from PIL.Image import ENCODERS
from torch import nn

from src.attention import MultiHeadAttention
from src.embedding import TransformerEmbedding
from src.layernorm import LayerNorm


class PositionalwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1, device=None):
        super(PositionalwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden).to(device)
        self.fc2 = nn.Linear(hidden, d_model).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncodeLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1, device=None):
        super(EncodeLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model).to(device)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionalwiseFeedForward(d_model, ffn_hidden, dropout).to(device)

        self.norm2 = LayerNorm(d_model).to(device)

        """
        在 PyTorch 中，nn.Dropout 是一个模块（Module），它的作用是在训练过程中随机将输入张量的某些元素置为零，以实现dropout正则化。nn.Dropout 
        本身并不存储任何张量数据，它只是一个操作，因此不需要被移动到特定的设备上。
        当你调用 nn.Dropout 实例时，它会直接作用于输入的张量，而这个张量可以位于 CPU 或 GPU 上。nn.Dropout 会根据输入张量所在的设备执行相应的操作。
        这意味着，只要你确保所有的输入张量都在正确的设备上，nn.Dropout 就会在正确的设备上执行操作，而不需要你显式地调用 .to(device)。
        """
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, dropout, device).to(device)
        self.layers = nn.ModuleList(
            [
                EncodeLayer(d_model, ffn_hidden, n_head, dropout).to(device) for _ in range(n_layer)
            ]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)

        return x
