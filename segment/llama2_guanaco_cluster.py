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

model_name = "meta-llama/Llama-2-7b-hf"
dataset_name = "timdettmers/openassistant-guanaco"


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
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
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

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
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
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to=None
)



#### CLUSTERING #########


def cluster_dataset(dataset, num_clusters):    
    # Extract text data (adjust the key 'text' if your dataset has a different text field)
    texts = [entry['text'] for entry in dataset]

    # Data Preparation and Feature Extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    # Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)

    # Assigning cluster labels to each text
    cluster_labels = kmeans.labels_

    # Creating a DataFrame for easier visualization
    clustered_data = pd.DataFrame({'text': texts, 'cluster': cluster_labels})

    return clustered_data



num_clusters = 4  # Adjust the number of clusters as needed
clustered_data = cluster_dataset(train_dataset, num_clusters)
unique_clusters = clustered_data['cluster'].unique()
cluster_datasets = {}

# Loop through each cluster
for cluster_label in unique_clusters:

    cluster_df = clustered_data[clustered_data['cluster'] == cluster_label]
    cluster_datasets[f"cluster_{cluster_label}"] = Dataset.from_pandas(cluster_df)
    

for cluster_label, cluster_dataset in cluster_datasets.items()[1:]:
    print(f'Training cluster {cluster_label}...')
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config
    )

    # Create a trainer for this cluster
    trainer = SFTTrainer(
        model=model,
        train_dataset=cluster_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Train the model for this cluster
    trainer.train()

    # Save the trained model
    new_model_name = f"llama-2-7b-guanaco-{num_clusters}_{cluster_label}"
    trainer.model.save_pretrained(new_model_name)




