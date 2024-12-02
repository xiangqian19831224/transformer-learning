import torch
from transformer import Transformer


def make_pad_mask(q, k, pad_idx_q, pad_idx_k):
    print("**************************************")
    print("q.ne(pad_idx_q)")
    print(q.ne(pad_idx_q))
    print("q.ne(pad_idx_q).unsqueeze(1)")
    print(q.ne(pad_idx_q).unsqueeze(1))
    print("q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)")
    print(q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3))

    len_q, len_k = q.size(1), k.size(1)
    q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
    q = q.repeat(1, 1, 1, len_k)

    print("q")
    print(q)
    print("**************************************")

    print("k.ne(pad_idx_k)")
    print(k.ne(pad_idx_k))
    print("k.ne(pad_idx_k).unsqueeze(1)")
    print(k.ne(pad_idx_k).unsqueeze(1))
    print("k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)")
    print(k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2))

    k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
    k = k.repeat(1, 1, len_q, 1)

    print("k")
    print(k)
    print("**************************************")

    mask = q & k

    print("mask")
    print(mask)
    return mask


def make_casual_mask(q, k, device):
    len_q, len_k = q.size(1), k.size(1)
    mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(device)
    return mask


def testMakePadMask():
    # 测试代码
    src_pad_idx = 0  # 源语言的填充索引
    trg_pad_idx = 0  # 目标语言的填充索引
    enc_voc_size = 1000  # 编码器词汇表大小
    dec_voc_size = 1000  # 解码器词汇表大小
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

    # 生成一些随机数据作为输入
    batch_size = 2  # 批次大小
    seq_len_q = 5  # 查询序列长度
    seq_len_k = 3  # 键序列长度
    q = torch.randint(1, enc_voc_size, (batch_size, seq_len_q), device=device)  # 查询序列，不包含填充索引
    k = torch.randint(1, dec_voc_size, (batch_size, seq_len_k), device=device)  # 键序列，不包含填充索引
    # 在第二维度（dim=1）填充两个0
    # 首先创建一个形状为 (2, 2) 的零张量，然后将其与原始张量拼接
    padding = torch.zeros((q.size(0), 2))
    q = torch.cat((q, padding), dim=1)

    k = torch.cat((k, padding), dim=1)

    print(q)
    print(k)

    # 生成掩码
    mask = make_pad_mask(q=q, k=k, pad_idx_q=src_pad_idx, pad_idx_k=trg_pad_idx)

    print("Mask shape:", mask.shape)
    print("Mask:", mask)


def testMakeCasualMask():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    q = torch.randn(2, 10, 512)  # 假设的查询张量
    k = torch.randn(2, 8, 512)  # 假设的键张量

    # 生成因果掩码
    mask = make_casual_mask(q=q, k=k, device=device)

    print("Mask shape:", mask.shape)
    print("Mask:", mask)


# 测试样例
def test_transformer():
    # 设置参数
    src_pad_ix = 0
    trg_pad_idx = 0
    enc_voc_size = 1000
    dec_voc_size = 1000
    d_model = 512
    max_len = 100
    n_heads = 8
    ffn_hidden = 2048
    n_layers = 6
    drop_prob = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建 Transformer 实例
    transformer = Transformer(src_pad_ix, trg_pad_idx, enc_voc_size, dec_voc_size, d_model, max_len, n_heads,
                              ffn_hidden, n_layers, drop_prob, device)

    # 创建假数据
    src = torch.randint(0, enc_voc_size, (32, 50)).to(device) # 32个样本，每个样本50个单词
    trg = torch.randint(0, dec_voc_size, (32, 50)).to(device)  # 32个样本，每个样本50个单词

    # 前向传播
    output = transformer(src, trg)

    print("Output shape:", output.shape)


# 运行测试样例
# testMakePadMask()
# testMakeCasualMask()
test_transformer()
