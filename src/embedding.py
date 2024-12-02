import torch
from torch import nn

random_torch = torch.rand(4, 4)
print(random_torch)


# 将输入的词汇索引转换为制定的维度的Embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model,device):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
        self.to(device)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()
        # 位置编码矩阵 （长度，编码向量）
        self.encoding = torch.zeros(max_len, d_model, device=device)
        # 位置编码不会被更新
        self.encoding.requires_grad = False

        # 定义序列
        pos = torch.arange(0, max_len, device=device)
        # 转为二维度张量
        pos = pos.float().unsqueeze(1)
        # 生成0-d_model的序列
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 位置编码
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        print("encoding:")
        print(self.encoding)

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :].to(x.device)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model,device)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        tok_emb = tok_emb.to(x.device)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
