import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

import os
from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

model_name = "mistralai/Mistral-7B-v0.1"
dataset_name = "monology/VMware-open-instruct-higgsfield"

attention_only = False
layer_config = "attention" if attention_only else "all"
cluster = "narval"

# Load dataset from local path
if cluster == "narval":
    dataset_path = "/scratch/alif/monology___v_mware-open-instruct-higgsfield/default/0.0.0/622a7cf65a222fcb"
    train_path = os.path.join(dataset_path, 'v_mware-open-instruct-higgsfield-train.arrow')
    train_dataset = Dataset.from_file(train_path)
elif cluster == "cedar":
    train_dataset = load_dataset(dataset_name, split="train")

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

output_dir = "./results"
num_train_epochs = 1
fp16 = True
bf16 = False
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001

optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True

save_steps = 0
logging_steps = 25
max_seq_length = 512  # Adjust as needed
packing = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
        fp16 = False
        bf16 = True

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# By default, LoRA is only applied to attention layers
# https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py#L49
# The QLoRA paper suggests that LoRA should be applied to all linear layers
# https://github.com/huggingface/peft/issues/735
if attention_only:
    print(f'Configuring LoRA to be applied to attention layers...')
    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    print(f'Configuring LoRA to be applied to all transformer linear layers...')
    target_modules = [name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules # Applying LoRA to all linear layers
    )

# Print trainable parameters for this PEFT config
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

peft_model = get_peft_model(model, peft_config)
print_trainable_parameters(peft_model)


# Preprocess function
def preprocess_function(examples):
    # Concatenate 'prompt' and 'completion' fields
    texts = [prompt + " " + completion for prompt, completion in zip(examples['prompt'], examples['completion'])]
    return {'text': texts}

hf_dataset = train_dataset.map(preprocess_function, batched=True)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to=None
)

print(f'Training single LoRA...')

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()

new_model = f"mistral-7b-instruct-lora-{layer_config}_r{lora_r}_a{lora_alpha}"
trainer.model.save_pretrained(new_model)
