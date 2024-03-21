import numpy as np
import torch

def generate_sample_data(num_samples, num_genres, num_moods, num_controls):
    # Generate random latents
    latents = torch.randn(num_samples, 512, 160, 1)
    
    # Generate random genre IDs
    genre_ids = torch.randint(0, num_genres, (num_samples,))
    
    # Generate random mood IDs
    mood_ids = torch.randint(0, num_moods, (num_samples,))
    
    # Generate random control signals
    controls = []
    for _ in range(num_controls):
        control_dim = np.random.randint(1, 5)  # Random control dimension
        control_signal = torch.randn(num_samples, 512, control_dim, 1)
        controls.append(control_signal)
    
    # Create a dictionary to store the sample data
    sample_data = {
        "latents": latents,
        "genre_ids": genre_ids,
        "mood_ids": mood_ids,
        "controls": controls
    }
    
    return sample_data


num_samples = 4
num_genres = 10
num_moods = 5
num_controls = 3

sample_data = generate_sample_data(num_samples, num_genres, num_moods, num_controls)

latents = sample_data["latents"]
genre_ids = sample_data["genre_ids"]
mood_ids = sample_data["mood_ids"]
controls = sample_data["controls"]

print("Latents shape:", latents.shape)
print("Genre IDs:", genre_ids)
print("Mood IDs:", mood_ids)
print("Number of control signals:", len(controls))
print("Control signal shapes:")
for i, control_signal in enumerate(controls):
    print(f"Control {i+1} shape:", control_signal.shape)