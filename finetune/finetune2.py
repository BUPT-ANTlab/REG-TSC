from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch

df = pd.read_json('/root/autodl-tmp/datas/finetune2_3.jsonl', lines=True)
ds = Dataset.from_pandas(df)

print(ds[:3])

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/finetune_models/jinan1_8_13_finetune_2_2_merge', use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = "<|pad|>"
# tokenizer.eos_token = "<|eos|>"

def process_func(example):
    MAX_LENGTH = 2560
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a Traffic Signal Control Agent.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['response']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)


model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/models/finetune_models/jinan1_8_13_finetune_2_2_merge', device_map="auto",torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,

)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="/root/autodl-tmp/models/finetune_models/jinan1_8_13_finetune_2_3",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=800,
    save_total_limit=3,
    # learning_rate=3e-4,
    learning_rate=5e-5,
    save_on_each_node=True,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()