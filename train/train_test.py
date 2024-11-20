from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# 加载预训练模型和分词器
model_name = "defog/llama-3-sqlcoder-8b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载微调数据集
from datasets import load_dataset
dataset = load_dataset("train_test.json")

train_model_name = "../train_model/kpaas_train_model"

# 定义训练参数
training_args = TrainingArguments(
    output_dir=train_model_name,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True  # 如果有 GPU，开启混合精度训练
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 开始微调
trainer.train()

# 保存微调后的模型
model.save_pretrained(train_model_name)
tokenizer.save_pretrained(train_model_name)
