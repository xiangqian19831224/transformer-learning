import torch
from torch import nn

from encoder import Encoder
from src.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 d_model,
                 max_len,
                 n_heads,
                 ffn_hidden,
                 n_layers,
                 drop_prob,
                 device,
                 ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            enc_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device
        ).to(device)
        self.decoder = Decoder(
            dec_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device
        ).to(device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        """
            在 Transformer 模型中，make_pad_mask 函数用于生成掩码，这个掩码用于在自注意力（Self-Attention）机制中忽略填充（padding）元素。
        以下是 make_pad_mask 函数的详细解释：

        函数参数：
        q 和 k：分别是查询（Query）和键（Key）张量，它们来自于 Transformer 的输入序列。在自注意力机制中，查询、键和值通常来自于同一个地方（自注意力）
        或者不同的地方（编码器-解码器注意力）。pad_idx_q 和 pad_idx_k：分别是查询和键序列中的填充索引。在处理序列数据时，为了将不同长度的序列统一到
        相同的长度，通常会用一个特定的值（如0）来填充较短的序列。
        函数步骤：
        获取序列长度：

        len_q 和 len_k 分别获取查询和键张量的序列长度。
        生成掩码：

        q.ne(pad_idx_q)：生成一个布尔张量，当 q 中的元素不等于 pad_idx_q 时为 True，否则为 False。
        unsqueeze(1) 和 unsqueeze(3)：在适当的维度上增加维度，以便进行广播操作。这里 unsqueeze(1) 在第二个维度（索引为1）上增加一个维度，
        unsqueeze(3) 在第四个维度（索引为3）上增加一个维度。
        repeat(1,1,1,len_k)：将布尔张量 q 沿着最后一个维度重复 len_k 次，以匹配键张量 k 的序列长度。
        对键张量 k 执行类似的操作：

        k.ne(pad_idx_k)：生成一个布尔张量，当 k 中的元素不等于 pad_idx_k 时为 True，否则为 False。
        unsqueeze(1) 和 unsqueeze(2)：在适当的维度上增加维度，以便进行广播操作。
        repeat(1,1,len_q,1)：将布尔张量 k 沿着第三个维度重复 len_q 次，以匹配查询张量 q 的序列长度。
        计算最终掩码：

        mask = q & k：通过逻辑与操作（&），计算两个布尔张量的逐元素与，得到最终的掩码。在两个张量都为 True 的位置，掩码为 True；否则为 False。
        掩码的作用：
        在自注意力机制中，这个掩码用于将填充元素的影响设置为零。具体来说，掩码会与注意力得分相乘，将掩码为 False 的位置的得分设置为一个非常小的值
        （通常是负无穷），这样在应用 softmax 函数时，这些位置的权重就会接近于零，从而忽略填充元素。
        """
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        mask = q & k

        return mask.to(self.device)

    def make_casual_mask(self, q, k):
        """
        函数 make_casual_mask 目的是创建一个因果掩码（也称为下三角掩码），这种掩码在解码器的自注意力层中使用，以确保每个位置只能关注它之前的位置，
        而不是之后的位置。这在序列生成任务中非常重要，比如机器翻译或文本摘要，因为它可以帮助模型在生成下一个词时只能使用已经生成的词。
        :param q:
        :param k:
        :return:
        """
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_casual_mask(trg, trg)
        enc = self.encoder(src, src_mask).to(self.device)
        out = self.decoder(trg, src, trg_mask, src_mask).to(self.device)
