import os
import math
import torch
import torch.nn as nn

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    TrainerCallback,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

import os
from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import islice

def preprocess_instruct(examples):
    # Concatenate 'prompt' and 'completion' fields
    texts = [prompt + " " + completion for prompt, completion in zip(examples['prompt'], examples['completion'])]
    return {'text': texts}

model_name = "meta-llama/Llama-2-7b-hf" # Also try "mistralai/Mistral-7B-v0.1"
dataset = "instruct"
cluster = "cedar"

num_clusters = 2  # Adjust the number of clusters as needed
cluster_strategy = "embeddings"
resume_from_cluster = 0

if cluster == "cedar":
    if dataset == "guanaco":
        dataset_name = "timdettmers/openassistant-guanaco"
    elif dataset == "instruct":
        dataset_name = "monology/VMware-open-instruct-higgsfield"
    train_dataset = load_dataset(dataset_name, split="train")
    train_dataset = train_dataset.map(preprocess_instruct, batched=True)
    print(f"Training dataset set to {dataset_name} from hugging face")

if cluster == "narval":
    if dataset == "guanaco":
        dataset_path = "/scratch/alif/timdettmers___json/timdettmers--openassistant-guanaco-c93588435bc90172/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-train.arrow"
    elif dataset == "instruct":
        dataset_path = '/scratch/alif/monology___v_mware-open-instruct-higgsfield/default/0.0.0/622a7cf65a222fcb/v_mware-open-instruct-higgsfield-train.arrow'
    train_dataset = Dataset.from_file(dataset_path)
    train_dataset = train_dataset.map(preprocess_instruct, batched=True)
    print(f"Training dataset set to: {dataset} from local path: {dataset_path}")

#### CLUSTERING #########

def cluster_embeddings(dataset, num_clusters, embeddings):    
    # Extract text data
    texts = [entry['text'] for entry in dataset]
    prompts = [entry['prompt'] for entry in dataset]
    completions = [entry['completion'] for entry in dataset]

    # Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)

    # Assigning cluster labels to each text
    cluster_labels = kmeans.labels_

    # Creating a DataFrame for easier visualization
    clustered_data = pd.DataFrame({'text': texts, 'prompt': prompts, 'completion': completions, 'cluster': cluster_labels})

    return clustered_data

def cluster_tfid(dataset, num_clusters):    
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

if cluster_strategy == "embeddings":
    embeddings = np.load(f'embeddings/{dataset}_embeddings.npy')
    clustered_data = cluster_embeddings(train_dataset, num_clusters, embeddings)
    print(f'Created {num_clusters} clusters from precomputed embeddings file: {dataset}_embeddings.npy')
elif cluster_strategy == "tfid":
    clustered_data = cluster_tfid(train_dataset, num_clusters)

unique_clusters = clustered_data['cluster'].unique()
cluster_datasets = {}

# Loop through each cluster and create datasets
for cluster_label in unique_clusters:
    cluster_df = clustered_data[clustered_data['cluster'] == cluster_label]
    cluster_datasets[f"cluster_{cluster_label}"] = Dataset.from_pandas(cluster_df)
    
# Sort the cluster numerically
cluster_datasets = dict(sorted(cluster_datasets.items()))  

count = 1
total = len(cluster_datasets.items())

for cluster_label, cluster_dataset in islice(cluster_datasets.items(), resume_from_cluster, total, 1):   
    
    print(f'({count}/{total}) Saving {cluster_label} for {dataset} dataset...')
    print(cluster_dataset[0])
    dataset_length = len(cluster_dataset)    
    dataset_name = f"data/{dataset}_{num_clusters}_{cluster_label}_{dataset_length}.json"
    cluster_dataset.to_json(dataset_name)
    count += 1



