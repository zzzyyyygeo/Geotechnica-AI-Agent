import torch
import torch.nn.functional as F

# 假设模型预测下一个词，输出了 4 个候选词的原始得分（Logits）
# 得分越高，模型越觉得应该选它。
# 候选词假设为：0:"渗水" 1:"坍塌" 2:"完工" 3:"下雨"
logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])

# 1. 把原始得分变成概率（Softmax 魔法）
probs = F.softmax(logits, dim=-1)
print("正常情况下的概率分布：")
print(f"渗水: {probs[0][0]:.2f}, 坍塌: {probs[0][1]:.2f}, 完工: {probs[0][2]:.2f}, 下雨: {probs[0][3]:.2f}")
# 你会发现，渗水的概率最高，但其他词也有微小的概率。

# 2. 加上 Temperature (温度参数) 看看会发生什么！
def apply_temperature(logits, temperature):
    # 温度越高，概率分布越平缓（容易瞎编）
    # 温度越低，概率分布越尖锐（极度保守）
    return F.softmax(logits / temperature, dim=-1)

print("\n温度设为 0.1 (极度冷静保守):")
probs_cold = apply_temperature(logits, 0.1)
print(f"渗水: {probs_cold[0][0]:.4f}, 坍塌: {probs_cold[0][1]:.4f}, 完工: {probs_cold[0][2]:.4f}")

print("\n温度设为 2.0 (脑洞大开，创造力强):")
probs_hot = apply_temperature(logits, 2.0)
print(f"渗水: {probs_hot[0][0]:.2f}, 坍塌: {probs_hot[0][1]:.2f}, 完工: {probs_hot[0][2]:.2f}")