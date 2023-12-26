# Trainable param regex: https://github.com/r-three/mats/blob/a7165f84cff194596465b50c49a49bcd4dbd0fbe/src/model/load_model.py#L63

# Use conjugate gradient calculation from scipy
from scipy.sparse.linalg import cg




# FISHER MERGING METHODS

def compute_diag_fisher(model, param_regex):
    example_fisher = {}
    for param_name, param in model.named_parameters():
        if (re.fullmatch(param_regex, param_name) and param.requires_grad):
            if param_name not in example_fisher:
                example_fisher[param_name] = torch.zeros_like(param.data)
            example_fisher[param_name] += torch.square(param.grad)
    return example_fisher
    

def get_model_fisher(model, dataset, device, world_size, fisher_path):
    
    model.eval()
    stored_fisher = {}
    
    def update_fisher(example_fisher):
    for param_name, value in example_fisher.items():
        if param_name not in stored_fisher:
            stored_fisher[param_name] = value
        else:
            stored_fisher[param_name] += value
    
    iterator = get_epoch_batches(
        dataset,
        batch_size=1,
        world_size=world_size,
        device=device
    )
    
    num_samples = 0
    param_regex = ".*"
    
    for batch in iterator:
        # Fisher is the gradient of the log likelihood (negative loss of log prob)
        loss, _ = model(batch)
        log_prob = -loss
        log_prob.backward()
        
        # Compute the per-example fisher and update total fisher
        with torch.no_grad():
            example_fisher = compute_diag_fisher(model, param_regex)
            update_fisher(example_fisher)
        
        num_samples += 1
        model.zero_grad()
        
        # 1000 examples is sufficient
        if num_samples >= 1000:
            break
            
    with torch.no_grad():
        stored_fisher = normalize_metadata(stored_fisher, num_samples)
        
    torch.save(detach_metadata(stored_fisher), fisher_path)
    
    
def conjugate_gradient_forward(
    sum_fisher_matrices, 
    sum_fisher_times_weight, 
    init_model,
    all_param_names,
    num_iters
):
    final_model = {}
    
    for param_name in tqdm(all_param_names):
        weight_shape = sum_fisher_times_weight[param_name].shape
        
        def matrix_vector_product(v):
            v_weight_torch = from_numpy(v).reshape(weight_shape).float()
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
        
        final_weight = torch.tensor(x_final).reshape(weight.shape)
        final_model[param_name] = final_weight
        
        return final_model
    
    

    
# DATASET AND CHECKPOINT LOADING
    
def get_dataset(filepath)

def load_checkpoints(pretrained_model, num_clusters, device)
    """
    Load model (adapter) checkpoints to merge.
    
    Args:
        pretrained_model: name of the base model (e.g. llama-2-7b)
        num_clusters: number of dataset clusters to retrieve adapter checkpoints
        device: device to load checkpoints onto (e.g. cpu or cuda)
    Return:
        checkpoint_paths: checkpoint file paths (to retrieve fishers later)
        loaded_checkpoints: adapter checkpoints to merge 
    """

def _create_dataLoader(pytorch_dataset, batch_size, should_shuffle, world_size, device):
    if is_distributedSetup(world_size):
        sampler = DistributedSampler(
            pytorch_dataset,
            num_replicas=world_size,
            rank=device,
            shuffle=should_shuffle,
        )

        data_loader = data.DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            sampler=sampler,
            collate_fn=pytorch_dataset.collate_fn,
        )
        return sampler, data_loader
    else:
        data_loader = data.DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=should_shuffle,
            collate_fn=pytorch_dataset.collate_fn,
        )

        return None, data_loader
    
def get_epoch_batches(pytorch_dataset, batch_size, world_size, device):
    _, data_loader = _create_dataLoader(
        pytorch_dataset, batch_size, False, world_size, device
    )

    for x in data_loader:
        yield x


        
        

# MODEL OPERATIONS

def map_params(model_params, map_fn):
    new_params = {}
    for param_name, param_value in model_params.items():
        new_params[param_name] = map_fn(param_value)
    return new_params

def reduce_params(model_params, reduce_fn):
    param_values = zip(*list(map(lambda x: x.values(), model_params)))
    param_names = model_params[0].keys()
    
    new_params = {}
    for param_name, param_value in zip(*[param_names, param_values]):
        new_params[param_name] = reduce_fn(torch.stack(list(param_value), dim=0))
    return new_params

def scale(model_params, scaler):
    scale_fn = lambda x: x * scaler
    scaled_model = map_params(model_params, scale_fn)
    return scaled_model

def scale_and_sum(model_params, model_lambda):
    sum fn = lambda params: torch.sum(params * model_lambda, dim=0)
    summed_model = reduce_params(model_params, sum_fn)
    return summed_model

def pairwise_param_map(params_a, params_b, map_fn):
    all_params = params_a.keys()
    new_params = {}
    
    for param_name in all_params:
        new_params[param_name] = map_fn(params_a[param_name], params_b[param_name])
    return new_params

def element_wise_multiply(params_a, params_b):
    element_wise_mul = lambda x, y: torch.mul(x, y)
    element_wise_mul_model = pairwise_param_map(params_a, params_b, element_wise_mul)
    return element_wise_mul_model





if __name__ == "__main__":
    
    # COMPUTE FISHERS
    # https://github.com/r-three/mats/blob/main/src/merging/save_metadata/save_fisher.py
    
    model_config, data_config, eval_config = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_list = ["guanaco"]
    
    for dataset in dataset_list:
        fisher_path = "model_dataset_fisher.pt"
        get_model_fisher(model, dataset, device, world_size, fisher_path)
        
        
    # MERGE FISHERS
    # https://github.com/r-three/mats/blob/main/src/merging/diagonal_fisherMerging.py
    
    # Load checkpoints
    pretrained_model = "llama-2-7b"
    num_clusters = 2
    checkpoint_paths, loaded_checkpoints = load_checkpoints(pretrained_model, num_clusters, torch.device("cpu"))
    
    # Load fishers
    loaded_fishers = {}
    for checkpoint_path in checkpoint_paths:
        fisher_path = "fisher_" + checkpoint_path
        fisher = torch.load(fisher_path, torch.device("cpu"))
        loaded_fishers[checkpoint_path] = fisher
    
    checkpoint_fisher_matrices = {}
    
    # The original implementation merges checkpoints by dataset
    # For adapters, merge checkpoints by cluster
    for checkpoint_path, loaded_checkpoint in loaded_checkpoints.items():
        dataset = get_dataset(checkpoint_path)
        checkpoint_fisher_matrices[dataset] = {"checkpoint": loaded_checkpoint}
        
    for fisher_path, loaded_fisher in loaded_fishers.items():
        dataset = get_dataset(fisher_path)
        checkpoint_fisher_matrices[dataset].update({"fisher": fisher})
    
    
    weighted_checkpoint_list = []
    fisher_list = []
    
    for dataset, checkpoint_fisher_matrix in checkpoint_fisher_matrices.items():
        checkpoint = checkpoint_fisher_matrix["checkpoint"]
        fisher = checkpoint_fisher_matrix["fisher"]
        
        fisher = set_minimum(fisher, 1e-8)
        
        # Scale fishers
        if len(loaded_checkpoints) == 2:
            if len(fisher_list) == 0:
                fisher = scale(fisher, model_lambda)
            else:
                assert len(fisher_list) == 1
                fisher = scale(fisher, (1 - model_lambda))
                
        weighted_cp = pairwise_param_map(
            checkpoint, fisher, lambda x, y: x * y
        )
        
        weighted_checkpoint_list.append(weighted_cp)
        fisher_list.append(fisher)
        
    merged_model = divide(
        scale_and_sum(weighted_checkpoint_list, 1),
        scale_and_sum(fisher_list, 1)
    )
    
    torch.save(merged_model, "merged_model.pt")
    
        
        
    # COMPARE: CONJUGATE GRADIENTS (MaTS)
    # https://github.com/r-three/mats/blob/main/src/merging/conjugateGradient_diagonalFisher.py
    
    checkpoint_gram_matrices = {}
    all_param_names = None
    
    for checkpoint_path, loaded_checkpoint in loaded_checkpoints.items():
        dataset = get_dataset(checkpoint_path)
        checkpoint_gram_matrices[dataset] = {"checkpoint": loaded_checkpoint}
        
    for fisher_path, loaded_fisher in loaded_fishers.items():
        dataset = get_dataset(fisher_path)
        checkpoint_gram_matrices[dataset].update({"fisher": fisher})
        all_param_names = loaded_fisher.keys()
        
    datasets_fishers = []
    datasets_weights = []
    datasets_fisher_times_weight = []
    datasets_nonmerged_weights = []
    
    for dataset, checkpoint_gram_matrix in checkpoint_gram_matrices.items():
        checkpoint = checkpoint_gram_matrix["checkpoint"]
        
        fisher_matrices = {}
        weights = {}
        
        for module_name, diag_fisher in checkpoint_gram_matrix["diagonal_fisher"].items():
            param_name = module_name
            weights[param_name] = checkpoint[param_name]
            fisher_matrices[param_name] = diag_fisher
            
        datasets_fishers.append(fisher_matrices)
        datasets_weights.append(weights)
        
        fisher_times_weight = element_wise_multiply(fisher_matrices, weights)
        datasets_fisher_times_weight.append(fisher_times_weight)
        
        nonmerged_weights = {}
        for param_name, param in checkpoint.items()
            if param_name not in fisher_matrices:
                nonmerged_weights[param_name] = param
        datasets_nonmerged_weights.append(nonmerged_weights)
        
        average_weights = scale_and_sum(datasets_weights, 1 / len(datasets_weights))
        
        sum_fisher_matrices = scale_and_sum(datasets_fishers, 1)
        sum_fisher_times_weight = scale_and_sum(datasets_fisher_times_weight, 1)
        
        
        
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
        
        final_nonmerged_model = scale_and_sum(datasets_nonmerged_weights, 1 / len(datasets_nonmerged_weights))
        
        for param_name, param in final_nonmerged_model.items():
            final_model[param_name] = param
            
        torch.save(final_model, "final_model.pt")
        
        