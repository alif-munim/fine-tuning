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

# model_name = "meta-llama/Llama-2-7b-hf"
# dataset_name = "timdettmers/openassistant-guanaco"

model_name = "mistralai/Mistral-7B-v0.1"
dataset_name = "monology/VMware-open-instruct-higgsfield"

# Working on narval: uncomment each line and cache locally
# dataset = load_dataset(dataset_name, split="train")
# model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

