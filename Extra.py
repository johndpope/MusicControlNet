def evaluate_single_multiple_controls(controlnet, dataloader, device):
    controlnet.eval()
    
    for batch in dataloader:
        latents = batch["latents"].to(device)
        genre_ids = batch["genre_ids"].to(device)
        mood_ids = batch["mood_ids"].to(device)
        controls = [control.to(device) for control in batch["controls"]]
        
        # Evaluate with single controls
        for control_idx in range(len(controls)):
            single_control = [controls[control_idx]]
            with torch.no_grad():
                output = controlnet(latents, genre_ids, mood_ids, single_control)
            # Compute metrics (e.g., melody accuracy, dynamics correlation, rhythm F1)
            # ...
        
        # Evaluate with multiple controls
        with torch.no_grad():
            output = controlnet(latents, genre_ids, mood_ids, controls)
        # Compute metrics (e.g., melody accuracy, dynamics correlation, rhythm F1)
        # ...
    
    # Aggregate and log the metrics
    # ...
            

def evaluate_created_controls(controlnet, dataloader, device):
    controlnet.eval()
    
    for batch in dataloader:
        latents = batch["latents"].to(device)
        genre_ids = batch["genre_ids"].to(device)
        mood_ids = batch["mood_ids"].to(device)
        controls = [control.to(device) for control in batch["created_controls"]]
        
        with torch.no_grad():
            output = controlnet(latents, genre_ids, mood_ids, controls)
        # Compute metrics (e.g., melody accuracy, dynamics correlation, rhythm F1)
        # ...
    
    # Aggregate and log the metrics
    # ...
            

def evaluate_partially_specified_controls(controlnet, dataloader, device):
    controlnet.eval()
    
    for batch in dataloader:
        latents = batch["latents"].to(device)
        genre_ids = batch["genre_ids"].to(device)
        mood_ids = batch["mood_ids"].to(device)
        controls = [control.to(device) for control in batch["created_controls"]]
        
        # Randomly mask controls to simulate partially-specified controls
        masked_controls = []
        for control in controls:
            mask = torch.rand_like(control) < 0.5
            masked_control = control * mask
            masked_controls.append(masked_control)
        
        with torch.no_grad():
            output = controlnet(latents, genre_ids, mood_ids, masked_controls)
        # Compute metrics (e.g., melody accuracy, dynamics correlation, rhythm F1)
        # ...
    
    # Aggregate and log the metrics
    # ...
            
def evaluate_extrapolated_duration(controlnet, dataloader, device):
    controlnet.eval()
    
    for batch in dataloader:
        latents = batch["latents"].to(device)
        genre_ids = batch["genre_ids"].to(device)
        mood_ids = batch["mood_ids"].to(device)
        controls = [control.to(device) for control in batch["created_controls"]]
        
        # Generate longer outputs by repeating the controls
        extended_controls = [control.repeat(2, 1, 1, 1) for control in controls]
        
        with torch.no_grad():
            output = controlnet(latents, genre_ids, mood_ids, extended_controls)
        # Compute metrics (e.g., melody accuracy, CLAP score, FAD)
        # ...
    
    # Aggregate and log the metrics
    # ...