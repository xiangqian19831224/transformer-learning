from encoder import EncodeLayer,Encoder
import torch


def testEncodeLayer():
    # 测试代码
    d_model = 8  # 模型的维度
    ffn_hidden = 16  # 前馈网络的隐藏层维度
    n_head = 2  # 多头注意力的头数
    dropout = 0.1  # Dropout 概率

    # 创建 EncodeLayer 实例
    encode_layer = EncodeLayer(d_model, ffn_hidden, n_head, dropout)

    # 生成一些随机数据作为输入
    # 假设我们有一个批次大小为 2，序列长度为 5 的输入
    input_data = torch.randn(2, 5, d_model)

    # 通过 EncodeLayer
    output_data = encode_layer(input_data)

    print("Output shape:", output_data.shape)
    print("Output data:", output_data)

def testEncoder():
    # 测试代码
    enc_voc_size = 1000  # 编码器词汇表大小
    max_len = 5  # 序列的最大长度
    d_model = 8  # 模型的维度
    ffn_hidden = 128  # 前馈网络的隐藏层维度
    n_head = 2  # 多头注意力的头数
    n_layer = 6  # 层数
    dropout = 0.1  # Dropout 概率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

    # 创建 Encoder 实例
    encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout, device)

    # 生成一些随机数据作为输入
    batch=2
    seq_len=5
    input_data = torch.randint(0, enc_voc_size, (batch, seq_len)).to(device)  # 假设有一个批次大小为 32，序列长度为 max_len 的输入

    # 通过 Encoder
    output_data = encoder(input_data, None)  # s_mask 在这个简化的例子中不使用

    print("Output shape:", output_data.shape)


testEncoder()