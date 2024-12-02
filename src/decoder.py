from torch import nn

from attention import MultiHeadAttention
from src.embedding import TransformerEmbedding
from src.encoder import PositionalwiseFeedForward
from src.layernorm import LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden,n_head,drop_prob,device=None):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head).to(device)
        self.norm1 = LayerNorm(d_model).to(device)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadAttention(d_model, n_head).to(device)
        self.norm2=LayerNorm(d_model).to(device)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionalwiseFeedForward(d_model,ffn_hidden,drop_prob,device).to(device)
        self.norm3 = LayerNorm(d_model).to(device)
        self.dropout3=nn.Dropout(drop_prob)

    def forward(self,dec,enc,t_mask,s_mask):
        _x=dec
        x=self.attention1(dec,dec,dec,t_mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)

        _x=x
        x=self.cross_attention(x,enc,enc,s_mask)
        x=self.dropout2(x)
        x=self.norm2(x + _x)

        _x=x
        x=self.ffn(x)
        x=self.dropout3(x)
        x=self.norm3(x+_x)
        return x


class Decoder(nn.Module):
    def __init__(self,dec_voc_size, max_len,d_model,ffn_hidden, n_head, n_layer,drop_prob,device):
        super(Decoder,self).__init__()
        self.embedding = TransformerEmbedding(
            dec_voc_size,d_model,max_len,drop_prob,device
        ).to(device)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, ffn_hidden, n_head, drop_prob).to(device)
                for _ in range(n_layer)
            ]
        )

        self.fc = nn.Linear(d_model,dec_voc_size).to(device)

    def forward(self, dec,enc,t_mask,s_mask):
        dec=self.embedding(dec)
        for layer in self.layers:
            dec=layer(dec,enc,t_mask,s_mask)
        dec=self.fc(dec)
        return dec





















