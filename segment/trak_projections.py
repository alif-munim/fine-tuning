def collect_grads(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  max_samples: Optional[int] = None):
    
    """
    Collects gradients from the model during eval and saves them to disk.
    """
    
    torch.random.manual_seed(0)
    
    model_id = 0 # used to draft random seed for projectors
    block_size = 128 # fixed block size for projectors
    
    projector_batch_size = 16
    project_interval = 16 # project every 16 batches
    save_interval = 160 # save every 160 batches
    
    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16) # torch.stack concats along new dim (2,3), whereas torch.cat concats along existing dim (6,)
        for i, projector in enumerate(projectors):
            current_projected_grads = projector.project(current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())
            
    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])
            
            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt")
            torch.save(projected_grads[dim], outfile)
            print(f"Saving {outfile}, {projected_grads[dim].shape}", flush=True) # flushes the buffer and displays immediately
            projected_grads[dim] = []
            
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device) # TODO: write this function
        
        
    projector = get_trak_projector(device) # TODO: write this function
    number_of_params = get_number_of_params(model) # TODO: write this function
    
    # initialize a projection for each target proj dim and keep track in a list
    projectors = []
    for dim in proj_dim:
        proj = projector() # TODO: create this class
        projectors.append(proj)
        
    count = 0
    
    # setup output dir for each dim and keep track in a dictionary
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)
        
    # max index for each dim (what does this do??)
    max_index = min(get_max_saved_index(output_dirs[dim], "grads") for dim in proj_dim) # TODO: write this function
    
    # projected gradients
    full_grads = []
    projected_grads = {dim: [] for dim in proj_dim} # initialize dictionary w/ entry for each dim
    
    for batch in tqdm(dataloader, total=len(dataloader))
        prepare_batch(batch) # TODO: write this function
        count += 1 
    
        if count <= max_index:
            print("skipping count", count)
            continue
            
        if gradient_type == "adam":
            if count == 1:
                print("Using Adam gradients")
            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
        elif gradient_type == "sign":
            if count == 1:
                print("Using Sign gradients") # What are these?
            vectorized_grads = obtain_sign_gradients(model, batch)
        else:
            if count == 1:
                print("Using SGD gradients")
            vectorized_grads = obtain_gradients(model, batch)
            
        full_grads.append(vectorized_grads)
        model.zero_grad()
        
        # compute projections and clear full grad buffer
        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = [] 
            
        if count % save_interval == 0:
            _save(projected_grads, output_dirs)
            
        # What does this do??
        if max_samples is not None and count == max_samples:
            break
            
    # project remaining grads
    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []
        
    for dim in proj_dim:
        _save(projected_grads, output_dirs)
        
    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_and_normalize_info(output_dir, prefix=grads) # TODO: write this function
        merge_info(output_dir, prefix="grads")
        
    print("Finished")