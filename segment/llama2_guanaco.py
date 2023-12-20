import os
import torch
import torch.nn as nn

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

model_name = "meta-llama/Llama-2-7b-hf"
dataset_name = "timdettmers/openassistant-guanaco"


attention_only = False
layer_config = "attention" if attention_only else "all"
new_model = f"llama-2-7b-guanaco-lora-{layer_config}"

lora_r = 64
lora_alpha = 128
lora_dropout = 0.1

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

report_to = "wandb"
output_dir = "./results"
num_train_epochs = 1
fp16 = True
bf16 = False
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
# weight_decay = 0.001
weight_decay = 0.0

# Use these if the dataset has a test or validation split
# ValueError: Trainer: evaluation requires an eval_dataset.
# evaluation_strategy = "steps"
# eval_steps = 187
# per_device_eval_batch_size = 1

optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
group_by_length = True

save_steps = 0
logging_steps = 10
max_seq_length = None
packing = False
# device_map = {"": 0}

dataset_path = "/scratch/alif/timdettmers___json/timdettmers--openassistant-guanaco-c93588435bc90172/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/"
train_path = os.path.join(dataset_path, 'json-train.arrow')
test_path = os.path.join(dataset_path, 'json-test.arrow')

train_dataset = Dataset.from_file(train_path)
test_dataset = Dataset.from_file(test_path)

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
    # device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

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

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    # num_train_epochs=num_train_epochs,
    max_steps = 1875,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to=report_to
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

# %load_ext tensorboard
# %tensorboard --logdir results/runs

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

