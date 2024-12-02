import torch
from decoder import Decoder

# 测试代码
dec_voc_size = 16  # 解码器词汇表大小
max_len = 5  # 序列的最大长度
d_model = 8  # 模型的维度
ffn_hidden = 128  # 前馈网络的隐藏层维度
n_head = 2  # 多头注意力的头数
n_layer = 6  # 层数
drop_prob = 0.1  # Dropout 概率
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

# 创建 Decoder 实例
decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device)

# 生成一些随机数据作为输入
batch=2
seq_len=5
dec = torch.randint(0, dec_voc_size, (batch, seq_len)).to(device)  # 解码器输入
enc = torch.randint(0, dec_voc_size, (batch, seq_len)).to(device)  # 编码器输出
t_mask = None  # 目标掩码（在这个简化的例子中不使用）
s_mask = None  # 源掩码（在这个简化的例子中不使用）

# 模拟下编码输出结果
enc = decoder.embedding(enc)

# 通过 Decoder
output = decoder(dec, enc, t_mask, s_mask)

print("Output shape:", output.shape)