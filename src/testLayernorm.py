from layernorm import LayerNorm
import torch

# 测试代码
d_model = 8  # 假设模型的维度是 512
layer_norm = LayerNorm(d_model)

# 生成一些随机数据作为输入
# 假设我们有一个批次大小为 2，序列长度为 5，特征维度为 d_model 的输入
input_data = torch.randn(2, 5, d_model)

# 通过 LayerNorm 层
output_data = layer_norm(input_data)

print("Output shape:", output_data.shape)
print("Output data:", output_data)