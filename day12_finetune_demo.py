import json
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. 自动生成微调数据集（避免手动建文件导致路径错误）
# ==========================================
print("准备教材文件 dataset_example.jsonl...")
dataset_content = [
    {
        "messages": [
            {"role": "system", "content": "你是一个严谨的地下工程专家。"},
            {"role": "user", "content": "基坑开挖深度超过5米属于什么工程？"},
            {"role": "assistant", "content": "属于超过一定规模的危险性较大的分部分项工程，需要组织专家论证。"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个严谨的地下工程专家。"},
            {"role": "user", "content": "地下连续墙渗水怎么处理？"},
            {"role": "assistant", "content": "应立即采用聚氨酯或水泥-水玻璃双液浆进行封堵，并排查是否存在槽段接头夹泥现象。"}
        ]
    }
]

with open("dataset_example.jsonl", "w", encoding="utf-8") as f:
    for item in dataset_content:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("✅ 数据集生成成功！")

# ==========================================
# 2. 加载模型和 Tokenizer（Qwen 5亿参数版本）
# ==========================================
model_id = "Qwen/Qwen2.5-0.5B"
print("\n正在加载模型（因为之前下载过，这次会直接从本地缓存秒开）...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 给 tokenizer 加上 pad_token，防止报错
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

# ==========================================
# 3. 配置 LoRA “便利贴”
# ==========================================
lora_config = LoraConfig(
    r=8, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("\nLoRA 配置成功，当前模型需要训练的参数如下：")
model.print_trainable_parameters() 

# ==========================================
# 4. 读取数据集并格式化
# ==========================================
print("\n正在读取教材并转换为 Token...")
dataset = load_dataset("json", data_files="dataset_example.jsonl", split="train")

def format_and_tokenize(example):
    # 将 messages 拼接成纯文本
    text = ""
    for msg in example["messages"]:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    
    # 转换成模型能看懂的 Token ID
    result = tokenizer(
        text, 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    # 训练时，labels 就是 input_ids
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = dataset.map(format_and_tokenize)

# ==========================================
# 5. 开始训练
# ==========================================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,           # 数据少，我们多训练几轮
    per_device_train_batch_size=1,# 每次只喂 1 条数据（保护显存）
    logging_steps=1,              # 每训练 1 步打印一次日志
    save_steps=10,                # 每 10 步保存一次
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

print("\n🚀 开始微调（体验版）...")
trainer.train()
print("\n🎉 微调完成！你的模型已经初步吸收了地下工程知识。")