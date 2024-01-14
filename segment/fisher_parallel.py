import os
import re
from scipy.sparse.linalg import LinearOperator, cg as conjugate_gradient # for conjugate gradient methods

from safetensors import safe_open # for checkpoint loading
from safetensors.torch import load_model, save_model, load_file
from safetensors.torch import load as load_safetensors

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, PeftModel

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm 
from itertools import islice
import numpy as np
from functools import partial

# FISHER MERGING METHODS
    
def compute_diag_fisher(model, param_regex):
    example_fisher = {}
    for param_name, param in model.named_parameters():
        if re.fullmatch(param_regex, param_name) and param.requires_grad:
            grad = param.grad if param.grad is not None else None
            if grad is not None:
                if param_name not in example_fisher:
                    example_fisher[param_name] = torch.zeros_like(param.data)
                example_fisher[param_name] += torch.square(grad)
    return example_fisher
    

def get_model_fisher(pretrained_model, checkpoint_path, dataset, tokenizer, fisher_path):

    model = load_peft_model(pretrained_model, checkpoint_path)
        
    for param in model.parameters():
        param.requires_grad = True
    
    req_grad_count = 0
    lora_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} does not require grad.")
            req_grad_count += 1
        if 'lora' in name:
            print(f"Parameter {name} is a LoRA parameter.")
            lora_count += 1
            
    if req_grad_count == 0:
        print("All parameters require grad.")        
    if lora_count == 0:
        print("All LoRA parameters have been merged.")
        
    model.eval()
    stored_fisher = {}
    
    def update_fisher(example_fisher):
        for param_name, value in example_fisher.items():
            if param_name not in stored_fisher:
                stored_fisher[param_name] = value
            else:
                stored_fisher[param_name] += value
    rank = 0
    batch_size = 1 
    
    dataloader = get_tokenized_dataloader(dataset, tokenizer, batch_size)
    
    num_samples = 0
    param_regex = ".*"
    
    print(f'Calculating losses for {checkpoint_path}...')
    for batch in tqdm(islice(dataloader, 1000), total=1000):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        # Fisher is the gradient of the log likelihood (which is the negative loss of the log prob)
        log_prob = -loss
        log_prob.backward()

        # Compute the per-example fisher and update total fisher
        with torch.no_grad():
            example_fisher = compute_diag_fisher(model, param_regex)
            update_fisher(example_fisher)

        num_samples += 1
        model.zero_grad()
            
    with torch.no_grad():
        stored_fisher = normalize_metadata(stored_fisher, num_samples)
        
    for param_name, param in model.named_parameters():
        print(f"mean value of {param_name}: {torch.mean(stored_fisher[param_name])}")
        
    torch.save(detach_metadata(stored_fisher), fisher_path)
    

def conjugate_gradient_forward(
    sum_fisher_matrices, 
    sum_fisher_times_weight, 
    init_model,
    all_param_names,
    num_iters
):
    final_model = {}
    
    pbar = tqdm(total=len(all_param_names))    
    for param_name in all_param_names:
        weight_shape = sum_fisher_times_weight[param_name].shape
        
        def matrix_vector_product(v):
            v_weight_torch = torch.from_numpy(v).reshape(weight_shape).float()
            matrix_vector = torch.mul(sum_fisher_matrices[param_name], v_weight_torch)
            return matrix_vector.flatten().cpu().numpy()
        
        b = sum_fisher_times_weight[param_name].detach().flatten()
        A = LinearOperator(
            (weight_shape.numel(), weight_shape.numel()), matvec=matrix_vector_product
        )
        
        if init_model is not None:
            init_x0 = init_model[param_name].detach().cpu().numpy().flatten()
            x_final, exit_code = conjugate_gradient(A, b, x0=init_x0, maxiter=num_iters)
            initial_error = np.linalg.norm(matrix_vector_product(init_x0) - b.numpy())
        else:
            x_final, exit_code = conjugate_gradient(A, b, maxiter=num_iters)
        
        final_error = np.linalg.norm(matrix_vector_product(x_final) - b.numpy())
        
        final_weight = torch.tensor(x_final).reshape(weight_shape)
        final_model[param_name] = final_weight
        pbar.update(1)
    
    pbar.close()
    return final_model
    

    
# DATASET AND CHECKPOINT LOADING

def load_peft_model(base_model, adapter_model):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto", # Parallelize across GPUs
    )
    model = PeftModel.from_pretrained(base_model, adapter_model)
    return model

def get_tokenized_dataloader(tokenized_dataset, tokenizer, batch_size):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to False for CLM
    )
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator 
    )
    
    return dataloader

def load_checkpoints(pretrained_model, num_clusters, device, target_epoch=None):
    checkpoint_dict = {}
    base_path = "./"

    for cluster_idx in range(num_clusters):
        cluster_name = f"{pretrained_model}_cluster_{cluster_idx}"
        cluster_path = os.path.join(base_path, cluster_name)

        epochs = [d for d in os.listdir(cluster_path) if d.startswith('epoch')]
        if not epochs:
            raise FileNotFoundError(f"No epochs found in {cluster_path}")

        latest_epoch = sorted(epochs, key=lambda x: int(x.split('_')[1]), reverse=True)[0]
        epoch = latest_epoch if target_epoch is None else target_epoch
        
        checkpoint_file = os.path.join(cluster_path, 
                                       epoch, 
                                       # 'model/'
                                      )
        checkpoint_dict[cluster_name] = checkpoint_file

    return checkpoint_dict



# REGEX FUNCTIONS

def get_cluster(checkpoint):
    pattern = re.compile(r"cluster_\d+")
    matches = pattern.findall(checkpoint)
    return matches[-1] if matches else None


# MODEL OPERATIONS

def debug_params(model_label, model_params, verbose=False):
    bug_counter = 0
    nan_params = []
    non_tensor_params = []
    for param_name, param_value in model_params.items():
        if isinstance(param_value, tuple):
            if verbose: print(f"WARNING: Tuple found in {model_label} in parameter: {param_name}")
            if verbose: print(param_value)
            bug_counter += 1
            non_tensor_params.append(param_name)
        elif not isinstance(param_value, torch.Tensor):
            if verbose: print(f"WARNING: Non-tensor found in {model_label} in parameter: {param_name}: {type(param_value)}")
            bug_counter += 1
            non_tensor_params.append(param_name)
        elif torch.isnan(param_value).any():
            if verbose: print(f"WARNING: parameter {param_name} contains NaN")
            bug_counter += 1
            nan_params.append(param_name)
    if bug_counter == 0:
        print(f"SUCCESS: {model_label} contains no NaN or non-tensor parameter values")
    else:
        if len(nan_params) > 0:
            print(f"WARNING: NaN param found in {model_label} in parameters:")
            print_parameter_dict(model_params, indent=4)
        elif len(non_tensor_params) > 0:
            print(f"WARNING: Non-tensor param found in {model_label} in parameters:")
            print_parameter_dict(model_params, indent=4)
            
def param_map(model_params, map_fn, select):
    new_params = {}
    for param_name, param_value in model_params.items():
        if select is not None and all(substring in param_name for substring in select):
            new_params[param_name] = map_fn(param_value)
        else:
            new_params[param_name] = param_value
    return new_params

def scale(model_params, scaler, select):
    scale_fn = lambda x: x * scaler
    scaled_model = param_map(model_params, scale_fn, select)
    return scaled_model

def reduce_params(model_params, reduce_fn, select):
    param_values = zip(*list(map(lambda x: x.values(), model_params)))
    param_names = model_params[0].keys()
    
    new_params = {}
    for param_name, param_value in zip(*[param_names, param_values]):
        if select is not None and all(substring in param_name for substring in select):
            new_params[param_name] = reduce_fn(torch.stack(list(param_value), dim=0))
        else:
            new_params[param_name] = param_value[0]
    return new_params

def scale_and_sum(model_params, model_lambda, select):
    sum_fn = lambda parameters: torch.sum(parameters * model_lambda, dim=0)
    summed_model = reduce_params(model_params, sum_fn, select)
    return summed_model

def pairwise_param_map(params_a, params_b, map_fn, select):
    a_params = params_a.keys()
    b_params = params_b.keys()
    
    new_params = {}
    
    for param_name in a_params:
        if select is not None and all(substring in param_name for substring in select):
            # Ensure both parameter sets have the parameter by name before mapping
            if param_name in b_params:
                new_params[param_name] = map_fn(params_a[param_name], params_b[param_name])
            else:
                raise KeyError(f"Parameter {param_name} not found in the second set of parameters.")
        else:
            new_params[param_name] = params_a[param_name]
    return new_params

def divide(params_a, params_b, select, epsilon=1e-8):
    divide_fn = lambda x, y: x / (y + epsilon * (y == 0).float())
    divide_model = pairwise_param_map(
        params_a, params_b, divide_fn, select
    )
    return divide_model

def element_wise_multiply(params_a, params_b, select):
    element_wise_mul = lambda x, y: torch.mul(x, y)
    element_wise_mul_model = pairwise_param_map(params_a, params_b, element_wise_mul, select)
    return element_wise_mul_model

def set_minimum(model_parameters, epsilon):
    """
    Set the minimum of the parameters to be epsilon. For any value less than epsilon,
    replace with epsilon
    """
    new_modelParameters = {}
    for parameter_name, parameter in model_parameters.items():
        new_parameter = parameter.clone()
        new_parameter[new_parameter < epsilon] = epsilon
        new_modelParameters[parameter_name] = new_parameter
    return new_modelParameters

def normalize_metadata(stored_metadata, count):
    normalized_metadata = {}
    for parameter_name, parameter in stored_metadata.items():
        normalized_metadata[parameter_name] = parameter / count
    return normalized_metadata

def detach_metadata(stored_metadata):
    detached_metadata = {}
    for parameter_name, parameter in stored_metadata.items():
        detached_metadata[parameter_name] = parameter.detach().contiguous().cpu()
    return detached_metadata


# HUGGING FACE DATA PRE-PROCESSING
# Partially borrowed from https://github.com/ovh/ai-training-examples/blob/main/notebooks/natural-language-processing/llm/miniconda/llama2-fine-tuning/llama_2_finetuning.ipynb

def preprocess_instruct(examples):
    # Concatenate 'prompt' and 'completion' fields
    texts = [prompt + " " + completion for prompt, completion in zip(examples['prompt'], examples['completion'])]
    return {'text': texts}

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

# Tokenize the dataset
def tokenize_function(example, max_length):
    return tokenizer(
        example["prompt"],
        example["completion"],
        truncation=True,       # truncate to the model's max length
        max_length=max_length,        # max length for the tokens
        padding="max_length",  # add padding to the tokens
        return_tensors="pt"    # return PyTorch tensors
    )

def shift_labels_right(examples):
    examples['labels'] = examples['input_ids'].copy()
    examples['labels'] = [x[1:] + [-100] for x in examples['labels']]
    return examples


if __name__ == "__main__":
    
    # COMPUTE FISHERS
    # https://github.com/r-three/mats/blob/main/src/merging/save_metadata/save_fisher.py
    
    dataset = "instruct"
    cluster = "cedar"
    
    pretrained_model = "llama-2-7b-instruct_lora-att-d0-r32-a16-2"
    num_clusters = 2
    
    compute_fishers = False
    debug_mode = True
    select_params = ['lora', 'attn']
    model_lambda_factor = 3
    model_lambda = 0.1 * model_lambda_factor
    epoch_num = 2
    
    use_conjugate_gradient = True
    initialization = "average"
    num_iters = 100
    
    print(
        f"""
        Beginning model merging process with the following configs:
        Model: {pretrained_model}
        Dataset: {dataset}
        Epoch: {epoch_num}
        Model Lambda: {model_lambda:.2f}
        Target Parameters: {select_params}
        Conjugate Gradient: {use_conjugate_gradient}
        Initialization: {initialization}
        Compute Fishers: {compute_fishers}
        Debug Mode: {debug_mode}
        """
    )

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
        
    model_name = "meta-llama/Llama-2-7b-hf"
    if model_name != "meta-llama/Llama-2-7b-hf":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto", # Parallelize across GPUs
        )
        max_length = get_max_length(model) # Returns 4096
    else:
        max_length = 1024
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Pre-process and tokenize dataset for loss calculations
    tokenized_dataset = train_dataset.map(
        partial(tokenize_function, max_length=max_length), 
        batched=True,
    )
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {'input_ids': examples['input_ids'], 'attention_mask': examples['attention_mask']},
        batched=True,
        remove_columns=tokenized_dataset.column_names  # This removes all columns except the ones specified above
    )
    tokenized_dataset = tokenized_dataset.map(shift_labels_right, batched=True)
    print(f"Tokenized dataset length: {len(tokenized_dataset)}")
    
    # Load checkpoints
    checkpoint_dict = load_checkpoints(pretrained_model, num_clusters, torch.device("cpu"), target_epoch=f'epoch_{epoch_num}')
    
    if compute_fishers:
        for model_folder in checkpoint_dict.keys():
            print(f'Calculating fisher for {model_folder}...')
            fisher_name = f"{model_folder}_fisher_ep{epoch_num}.pt"
            fisher_path = os.path.join('fishers/', fisher_name)
            # model = load_file(checkpoint_dict[model_folder])
            checkpoint_path = checkpoint_dict[model_folder]
            get_model_fisher(model_name, checkpoint_path, tokenized_dataset, tokenizer, fisher_path)
            print(f'Saved fisher for {model_folder} at {fisher_path}')
        
        
    # MERGE FISHERS
    # https://github.com/r-three/mats/blob/main/src/merging/diagonal_fisherMerging.py
    
    if not use_conjugate_gradient:
        # Load fishers    
        loaded_fishers = {}
        for model_folder in checkpoint_dict.keys():
            fisher_name = f"{model_folder}_fisher_ep{epoch_num}.pt"
            fisher_path = os.path.join('fishers/', fisher_name)
            fisher = torch.load(fisher_path, torch.device("cpu"))
            loaded_fishers[model_folder] = fisher
        if debug_mode: print(f"keys for loaded_fishers: {loaded_fishers.keys()}")

        # The original implementation merges checkpoints by dataset
        # For adapters, merge checkpoints by cluster
        checkpoint_fisher_matrices = {}

        for model_folder, checkpoint_path in checkpoint_dict.items():
            cluster = get_cluster(model_folder)
            print(f"Adding checkpoint path for {model_folder} and {cluster}: {checkpoint_path}")
            checkpoint_fisher_matrices[cluster] = {"checkpoint": checkpoint_path}

        for model_folder, loaded_fisher in loaded_fishers.items():
            cluster = get_cluster(model_folder)
            if cluster not in checkpoint_fisher_matrices:
                raise ValueError(f"Cluster key {cluster} not found in checkpoint_fisher_matrices")
            checkpoint_fisher_matrices[cluster].update({"fisher": loaded_fisher})

        weighted_checkpoint_list = []
        fisher_list = []

        for cluster, checkpoint_fisher_matrix in checkpoint_fisher_matrices.items():
            if debug_mode: print(f"cluster name: {cluster}")
            if debug_mode: print(f"matrix keys: {checkpoint_fisher_matrix.keys()}")      
            checkpoint_path = checkpoint_fisher_matrix["checkpoint"]
            
            print(f"loading peft model {model_name} with adapter {checkpoint_path}")
            checkpoint_model = load_peft_model(model_name, checkpoint_path)
            checkpoint = checkpoint_model.state_dict()
            checkpoint = {key: value.to('cpu') for key, value in checkpoint.items()}

            print(f'Loaded checkpoint for {cluster}')
            if debug_mode: debug_params(checkpoint_path, checkpoint)

            fisher = checkpoint_fisher_matrix["fisher"]
            fisher = set_minimum(fisher, 1e-8)

            print(f'Loaded fisher for {cluster}')
            if debug_mode: debug_params("fisher", fisher)

            # Scale fishers
            if len(checkpoint_dict) == 2:
                if len(fisher_list) == 0:
                    fisher = scale(fisher, model_lambda, select=select_params)
                    if debug_mode: debug_params("scaled fisher", fisher)
                else:
                    assert len(fisher_list) == 1
                    fisher = scale(fisher, (1 - model_lambda), select=select_params)
                    if debug_mode: debug_params("scaled fisher", fisher)

            weighted_cp = pairwise_param_map(
                checkpoint, fisher, lambda x, y: x * y, select=select_params
            )
            if debug_mode: debug_params("weighted_cp", weighted_cp)

            weighted_checkpoint_list.append(weighted_cp)
            fisher_list.append(fisher)

        weighted_cp_sum = scale_and_sum(weighted_checkpoint_list, 1, select=select_params)
        if debug_mode: debug_params("weighted_cp_sum", weighted_cp_sum)
        fisher_sum = scale_and_sum(fisher_list, 1, select=select_params)
        if debug_mode: debug_params("fisher_sum", fisher_sum)

        merged_model = divide(weighted_cp_sum, fisher_sum, select=select_params)

        if debug_mode: debug_params("merged_model", merged_model)
        selected_params_part = '-' + '-'.join(select_params)

        # Construct the merged filename dynamically
        merged_filename = (
            f"{pretrained_model}"
            f"{selected_params_part}"
            f"-fisher-ep{epoch_num}"
            f"-ml{int(model_lambda_factor)}" 
            ".pt"
        )
        merged_path = os.path.join('merged_models/', merged_filename)
        torch.save(merged_model, merged_path)
        print(f'Merged models and saved to {merged_path}')
    
    
    # COMPARE: CONJUGATE GRADIENTS (MaTS)
    # https://github.com/r-three/mats/blob/main/src/merging/conjugateGradient_diagonalFisher.py
       
    if use_conjugate_gradient:    
        # Load fishers    
        loaded_fishers = {}
        for model_folder in checkpoint_dict.keys():
            fisher_name = f"{model_folder}_fisher_ep{epoch_num}.pt"
            fisher_path = os.path.join('fishers/', fisher_name)
            fisher = torch.load(fisher_path, torch.device("cpu"))
            loaded_fishers[model_folder] = fisher
        if debug_mode: print(f"keys for loaded_fishers: {loaded_fishers.keys()}")

        checkpoint_gram_matrices = {}
        all_param_names = None

        for model_folder, checkpoint_path in checkpoint_dict.items():
            cluster = get_cluster(model_folder)
            print(f"Adding checkpoint path for {model_folder} and {cluster}: {checkpoint_path}")
            checkpoint_gram_matrices[cluster] = {"checkpoint": checkpoint_path}

        for model_folder, loaded_fisher in loaded_fishers.items():
            cluster = get_cluster(model_folder)
            if cluster not in checkpoint_gram_matrices:
                raise ValueError(f"Cluster key {cluster} not found in checkpoint_fisher_matrices")
            checkpoint_gram_matrices[cluster].update({"diagonal_fisher": loaded_fisher})
            all_param_names = loaded_fisher.keys()

        datasets_fishers = []
        datasets_weights = []
        datasets_fisher_times_weight = []
        datasets_nonmerged_weights = []

        for cluster, checkpoint_gram_matrix in checkpoint_gram_matrices.items():
            if debug_mode: print(f"cluster name: {cluster}")
            if debug_mode: print(f"matrix keys: {checkpoint_gram_matrix.keys()}")
            checkpoint_path = checkpoint_gram_matrix["checkpoint"]
            
            print(f"loading peft model {model_name} with adapter {checkpoint_path}")
            checkpoint_model = load_peft_model(model_name, checkpoint_path)
            checkpoint = checkpoint_model.state_dict()
            checkpoint = {key: value.to('cpu') for key, value in checkpoint.items()}

            print(f'Loaded checkpoint for {cluster}')
            if debug_mode: debug_params(checkpoint_path, checkpoint)

            fisher_matrices = {}
            weights = {}

            for module_name, diag_fisher in checkpoint_gram_matrix["diagonal_fisher"].items():
                param_name = module_name
                weights[param_name] = checkpoint[param_name]
                fisher_matrices[param_name] = diag_fisher

            datasets_fishers.append(fisher_matrices)
            datasets_weights.append(weights)

            fisher_times_weight = element_wise_multiply(fisher_matrices, weights, select_params)
            datasets_fisher_times_weight.append(fisher_times_weight)

            nonmerged_weights = {}
            for param_name, param in checkpoint.items():
                if param_name not in fisher_matrices:
                    nonmerged_weights[param_name] = param
            datasets_nonmerged_weights.append(nonmerged_weights)

        average_weights = scale_and_sum(datasets_weights, 1 / len(datasets_weights), select_params)

        sum_fisher_matrices = scale_and_sum(datasets_fishers, 1, select_params)
        sum_fisher_times_weight = scale_and_sum(datasets_fisher_times_weight, 1, select_params)

        if initialization == "average":
            init_model = average_weights
        elif initialization == "pretrained":
            init_model = pretrained_checkpoint
        else:
            if initialization is not None:
                init_model = {}
                # Transpose weights
                for param_name, param in torch.load(initialization).items():
                    init_model[param_name] = param
            else:
                init_model = None


        final_model = conjugate_gradient_forward(
            sum_fisher_matrices,
            sum_fisher_times_weight,
            init_model,
            all_param_names,
            num_iters
        )

        final_nonmerged_model = scale_and_sum(datasets_nonmerged_weights, 1 / len(datasets_nonmerged_weights), select_params)

        for param_name, param in final_nonmerged_model.items():
            final_model[param_name] = param

        # Construct the merged filename dynamically
        selected_params_part = '-' + '-'.join(select_params)
        merged_filename = (
            f"{pretrained_model}"
            f"{selected_params_part}"
            f"-cg-fisher-ep{epoch_num}"
            f"-ml{int(model_lambda_factor)}" 
            ".pt"
        )
        merged_path = os.path.join('merged_models/', merged_filename)
        torch.save(final_model, merged_path)
        print(f'Merged models and saved to {merged_path}')