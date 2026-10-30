import torch
import torch.nn as nn

# 假设我们有 3 个词（比如：地下、连续墙、渗水）
# 在模型里，它们已经被转换成了向量（这里假设每个词是一个长度为 4 的向量）
# 维度含义：(batch_size=1, sequence_length=3, embedding_dim=4)
input_tensor = torch.tensor([
    [
        [1.0, 0.1, 0.1, 0.1], # 词1（地下）
        [0.1, 1.0, 0.1, 0.1], # 词2（连续墙）
        [0.1, 0.1, 1.0, 0.5]  # 词3（渗水）
    ]
])

print(f"输入张量形状: {input_tensor.shape} -> (批次, 词数, 词向量维度)")

# 1. 直接调用 PyTorch 的多头注意力机制！
# embed_dim=4: 词向量长度
# num_heads=2: 分成 2 个头去理解（每个头看 2 维）
multihead_attn = nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)

# 2. 前向传播（计算 Attention）
# 注意：在 Self-Attention 中，Q、K、V 都是同一个输入！
attn_output, attn_weights = multihead_attn(input_tensor, input_tensor, input_tensor)

print(f"\n输出张量形状: {attn_output.shape}")
print(f"\n注意力权重矩阵 (每行代表一个词对其他词的关注度):")
print(attn_weights[0].detach().numpy())

# 看看词1 到底对谁关注度最高
print(f"\n词1（地下）对 [地下, 连续墙, 渗水] 的关注度分别是: {attn_weights[0][0].tolist()}")