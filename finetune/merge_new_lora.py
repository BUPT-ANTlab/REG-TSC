import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 离线设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 使用绝对路径
mode_path = '/root/autodl-tmp/models/finetune_models/jinan1_8_13_finetune_2_2_merge'
lora_path = '/root/autodl-tmp/models/finetune_models/jinan1_8_13_finetune_2_3/checkpoint-189'
save_path = '/root/autodl-tmp/models/finetune_models/jinan1_8_13_finetune_2_3_merge'

tokenizer = AutoTokenizer.from_pretrained(
    mode_path, trust_remote_code=True, local_files_only=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    mode_path,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto",
    torch_dtype="auto"
).eval()

lora_model = PeftModel.from_pretrained(
    base_model,
    model_id=lora_path,
    local_files_only=True
)

merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Merged model saved at:", save_path)
