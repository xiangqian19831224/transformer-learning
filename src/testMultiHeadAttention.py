from attention import MultiHeadAttention
import torch

# 创建多头注意力实例
x=torch.rand(2,3,8)
d_model=8
n_head=2
attention = MultiHeadAttention(d_model, n_head)
out=attention(x,x,x)
print(out)
