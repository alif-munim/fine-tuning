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
from tqdm.auto import tqdm

from sklearn.decomposition import IncrementalPCA
from joblib import dump, load

def preprocess_instruct(examples):
    # Concatenate 'prompt' and 'completion' fields
    texts = [prompt + " " + completion for prompt, completion in zip(examples['prompt'], examples['completion'])]
    return {'text': texts}

model_name = "meta-llama/Llama-2-7b-hf" # Also try "mistralai/Mistral-7B-v0.1"
dataset = "guanaco"
cluster = "cedar"

num_clusters = 2  # Adjust the number of clusters as needed
cluster_strategy = "gradients"
resume_from_cluster = 0

if cluster == "cedar":
    if dataset == "guanaco":
        dataset_name = "timdettmers/openassistant-guanaco"
        train_dataset = load_dataset(dataset_name, split="train")
    elif dataset == "instruct":
        dataset_name = "monology/VMware-open-instruct-higgsfield"
        train_dataset = load_dataset(dataset_name, split="train")
        train_dataset = train_dataset.map(preprocess_instruct, batched=True)
    print(f"Training dataset set to {dataset_name} from hugging face")

if cluster == "narval":
    if dataset == "guanaco":
        dataset_path = "/scratch/alif/timdettmers___json/timdettmers--openassistant-guanaco-c93588435bc90172/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-train.arrow"
        train_dataset = Dataset.from_file(dataset_path)
    elif dataset == "instruct":
        dataset_path = '/scratch/alif/monology___v_mware-open-instruct-higgsfield/default/0.0.0/622a7cf65a222fcb/v_mware-open-instruct-higgsfield-train.arrow'
        train_dataset = Dataset.from_file(dataset_path)
        train_dataset = train_dataset.map(preprocess_instruct, batched=True)
    print(f"Training dataset set to: {dataset} from local path: {dataset_path}")
    
    
### GRADIENTS ###

def aggregate_gradients_for_batch(batch, tokenizer, model, max_length=None):
    # Check if tokenizer has a padding token, if not, set it to the eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the batch
    inputs = tokenizer(batch, return_tensors='pt', padding=True, max_length=max_length, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    model.zero_grad()
    # Forward pass with batched inputs
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    # Backward pass for the entire batch
    loss.backward()

    # Collect gradients for the entire batch from the last layer
    batch_gradients = []
    for name, parameter in model.named_parameters():
        if "lm_head" in name and parameter.requires_grad and parameter.grad is not None:
            # Reshape gradients to (batch_size, -1), we do not detach and move to CPU yet
            batch_grads_float32 = parameter.grad.view(inputs['input_ids'].shape[0], -1).to(dtype=torch.float32) # leads to stacking along batch dimension
            # batch_grads_float32 = parameter.grad.view(-1).to(dtype=torch.float32) # leads to collapsing into 1 long tensor
            batch_gradients.append(batch_grads_float32)
    
    # Concatenate the gradients for the batch along the second dimension
    # (batch_size, sum_of_all_gradients_dimensions)
    # all_gradients = torch.cat(batch_gradients, dim=1).detach().cpu().numpy()
    all_gradients = torch.stack(batch_gradients).detach().cpu().numpy()
    
    # Return a numpy array of shape (batch_size, num_gradients_per_example)
    return all_gradients


def stack_gradients_for_batch(batch, tokenizer, model, max_length=None):
    # Check if tokenizer has a padding token, if not, set it to the eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the batch
    inputs = tokenizer(batch, return_tensors='pt', padding=True, max_length=max_length, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Forward pass in a loop for each example to get individual gradients
    gradients_list = []
    for i in range(inputs['input_ids'].shape[0]):
        model.zero_grad()
        outputs = model(**{k: v[i:i+1] for k, v in inputs.items()}, labels=inputs['input_ids'][i:i+1])
        loss = outputs.loss
        loss.backward()

        # Collect gradients for the current example from the last layer
        example_gradients = []
        for name, parameter in model.named_parameters():
            if "lm_head" in name and parameter.requires_grad and parameter.grad is not None:
                # Note: We're not detaching and moving to CPU here
                example_grads_float32 = parameter.grad.view(-1).to(dtype=torch.float32)
                example_gradients.append(example_grads_float32)
        # Concatenate the gradients for the current example and keep on GPU
        gradients_list.append(torch.cat(example_gradients))

    # Stack the gradients, detach and move the entire tensor to CPU at once
    all_gradients = torch.stack(gradients_list).detach().cpu().numpy()

    # Return a numpy array of shape (batch_size, num_gradients_per_example)
    return all_gradients

def manage_ipca_checkpoints(ipca_model_name, step_count, ipca, max_checkpoints=10):
    # Pattern to match the checkpoint files
    checkpoint_pattern = f"{ipca_model_name}_*.joblib"
    
    # Find all existing checkpoint files
    existing_checkpoints = glob.glob(checkpoint_pattern)
    
    # Sort the files by their step count, extracted from the filename
    sorted_checkpoints = sorted(existing_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # If we have more checkpoints than the max allowed, remove the oldest ones
    while len(sorted_checkpoints) >= max_checkpoints:
        os.remove(sorted_checkpoints.pop(0))  # Remove the oldest checkpoint
    
    # Save the new checkpoint
    last_saved_pca = f"{ipca_model_name}_{step_count}.joblib"
    dump(ipca, last_saved_pca)
    
    # Add the new checkpoint to the list
    sorted_checkpoints.append(last_saved_pca)
    
    print(f"{step_count}: saving PCA at {last_saved_pca}. There are {len(sorted_checkpoints)} checkpoints saved.")

#### CLUSTERING #########

def cluster_gradients(dataset, num_clusters, batch_size, max_length):

    # Load the model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"  # replace with your model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Set the model to evaluation mode and to the device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    model.eval()
    
    # For PCA, n_components must be less or equal to batch_size
    batch_size = 16
    n_components = batch_size

    # Initialize IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    gradient_buffer = []
    gradient_buffer_size = 1  # The desired buffer size
    expected_batch_size = batch_size
    step_count = 0
    total_steps = len(dataset['text']) // batch_size

    compute_pca = True
    ipca_save_step = 5
    
    load_ipca_checkpoint = False
    ipca_model_name = "b1_lmhead_ipca" # ["ipca_model", "b1_lmhead_ipca"]
    ipca_checkpoint_step = 616
    ipca_checkpoint = f"{ipca_model_name}_{ipca_checkpoint_step}.joblib"   
    
    end_idx = min(0 + batch_size, len(dataset['text']))
    batch_texts = dataset['text'][0:end_idx]
    batch_gradients = stack_gradients_for_batch(batch_texts, tokenizer, model, max_length)
    gradient_buffer.append(batch_gradients)
    gradient_list = np.vstack(gradient_buffer)
    print(f"gradient_list (batch): {gradient_list.shape}")
    ipca.partial_fit(gradient_list)
    gradient_buffer = [] 
    
    example_batch_size = 1
    end_idx = min(0 + example_batch_size, len(dataset['text']))
    example_batch_texts = dataset['text'][0:end_idx]
    example_gradients = stack_gradients_for_batch(example_batch_texts, tokenizer, model, max_length)
    gradient_buffer.append(example_gradients)
    gradient_list = np.vstack(gradient_buffer)
    print(f"gradient_list (reduced example): {gradient_list.shape}")
    reduced_gradients_batch = ipca.transform(gradient_list)
    gradient_buffer = [] 
    
    
    print(f"""
        Starting Incremental PCA training with the following configs
        
        Compute IPCA: {compute_pca}
        Load IPCA Checkpoint: {load_ipca_checkpoint}
        IPCA Checkpoint Path: {ipca_checkpoint}
        IPCA Save Steps: {ipca_save_step}
        Num Components: {n_components}
        Batch Size: {batch_size}
        Gradient Shape: {batch_gradients.shape}
        Reduced Gradient Shape: {reduced_gradients_batch.shape}
    """)
    
    
    if load_ipca_checkpoint:
        ipca = load(ipca_checkpoint)
      
    if compute_pca:
        # Process gradients in batches and accumulate in buffer
        for i in tqdm(range(0, len(dataset['text']), batch_size), desc='Estimating PCA model from gradients...'):

            step_count += 1
            if (load_ipca_checkpoint and step_count > ipca_checkpoint_step) or not load_ipca_checkpoint:
                if step_count % ipca_save_step == 0:
                    manage_ipca_checkpoints(ipca_model_name, step_count, ipca)

                end_idx = min(i + batch_size, len(dataset['text']))
                batch_texts = dataset['text'][i:end_idx]

                if len(batch_texts) < expected_batch_size:
                    print(f"Skipping a batch with size {len(batch_texts)} which is smaller than the expected size {expected_batch_size}.")
                    continue
                else:
                    batch_gradients = stack_gradients_for_batch(batch_texts, tokenizer, model, max_length)
                    gradient_buffer.append(batch_gradients) # Append batch_size (16) gradients to the buffer

                    # If buffer has enough elements, perform partial_fit
                    if len(gradient_buffer) == gradient_buffer_size:
                        print(f"Gradient buffer size: {len(gradient_buffer)}")
                        # Why isn't the buffer clearing properly?
                        gradient_list = np.vstack(gradient_buffer) # Unable to allocate 125. GiB for an array with shape (256, 131072000) and data type float32
                        ipca.partial_fit(gradient_list) # NOTE: accumulate more gradients before fit to speed up computation
                        gradient_buffer = []  # Clear the buffer
                        del gradient_list  # Delete the variable holding the tensor
                        del gradient_buffer
                        torch.cuda.empty_cache()  # Release GPU memory
                        gc.collect()

        print(f"{step_count}/{total_steps}: saving PCA at ipca_model_{step_count}.joblib")
        last_saved_pca = f"{ipca_model_name}_{step_count}.joblib"
        dump(ipca, last_saved_pca)

    # Convert the list of batch gradient arrays into a single array
    # gradient_features = np.vstack(gradient_features)

    gradient_buffer = []
    reduced_gradients = []
    example_batch_size = 1
    
    for i in tqdm(range(0, len(train_dataset), batch_size), desc='Reducing gradients with IPCA transform'):
        end_idx = min(0 + example_batch_size, len(dataset['text']))
        example_batch_texts = dataset['text'][0:end_idx]
        example_gradients = stack_gradients_for_batch(example_batch_texts, tokenizer, model, max_length)
        gradient_buffer.append(example_gradients)
        gradient_list = np.vstack(gradient_buffer)
        reduced_gradients_batch = ipca.transform(gradient_list)
        reduced_gradients.append(reduced_gradients_batch)
        gradient_buffer = []
        
    stacked_gradients = np.stack(arrays_list, axis=0)
    np.save('reduced_gradients_stack.npy', stacked_gradients)
    print(f"Saved reduced gradients with shape {stacked_gradients.shape} to reduced_gradients_stack.npy")
    
    # Cluster the gradient features
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(reduced_gradients)
    cluster_labels = kmeans.labels_

    # Create the clustered dataset
    clustered_data = pd.DataFrame({'text': train_dataset['text'], 'cluster': cluster_labels})
    return clustered_data

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
elif cluster_strategy == "gradients":
    batch_size = 1
    max_length = 512
    clustered_data = cluster_gradients(train_dataset, num_clusters, batch_size, max_length)
    save_path = f"gradient_clusters_{num_clusters}.csv" 
    clustered_data.to_csv(save_path)
    print(f"Clustered data using model gradients and saved dataframe to {save_path}")

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
