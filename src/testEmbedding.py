import torch

from embedding import PositionalEmbedding, TokenEmbedding
from src.embedding import TransformerEmbedding

def testPositionalEmbedding():
    # 测试代码
    d_model = 8  # 例如，模型的维度是 512
    max_len = 5  # 例如，序列的最大长度是 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

    # 创建 PositionalEmbedding 实例
    positional_embedding = PositionalEmbedding(d_model, max_len, device)

    # 输入
    x = torch.zeros(4, 5).to(device)  # 假设有一个批次大小为 4，序列长度为 5 的输入

    #获取位置编码
    encoding = positional_embedding(x)
    print(encoding)


def testTokenEmbedding():
    # 测试代码
    vocab_size = 1000  # 假设词汇表大小为 1000
    d_model = 4  # 假设模型的维度是 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

    # 创建 TokenEmbedding 实例
    token_embedding = TokenEmbedding(vocab_size, d_model,device)

    # 生成一个随机的索引张量，模拟一批输入数据
    # 假设我们有一个批次大小为 2，序列长度为 5 的输入
    input_indices = torch.randint(0, vocab_size, (2, 5)).to(device)

    # 使用 TokenEmbedding 来获取输入索引对应的嵌入向量
    embedded_tokens = token_embedding(input_indices)

    print("Shape of embedded tokens:", embedded_tokens.shape)
    print("Embedded tokens:", embedded_tokens)

def testTransformerEmbedding():
    # 测试代码
    vocab_size = 1000  # 词汇表大小
    d_model = 8  # 模型的维度
    max_len = 6  # 序列的最大长度
    drop_prob = 0.1  # Dropout 概率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备

    # 创建 TransformerEmbedding 实例
    transformer_embedding = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)

    # 生成一些随机数据作为输入
    batch=2
    seq_len=5
    input_data = torch.randint(0, vocab_size, (batch, seq_len)).to(device)  # 假设有一个批次大小为 32，序列长度为 50 的输入

    # 通过 TransformerEmbedding
    output_data = transformer_embedding(input_data)

    print("Output shape:", output_data.shape)
    print("Output data:", output_data)

testPositionalEmbedding()
testTokenEmbedding()
testTransformerEmbedding()