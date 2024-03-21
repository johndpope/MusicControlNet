import os
import random
import torch
from torch.utils.data import Dataset
from MusicHelper import load_audio


class MusicDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.audio_files = self._get_audio_files()
        
    def _get_audio_files(self):
        audio_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".mp3"):
                    audio_files.append(os.path.join(root, file))
        return audio_files
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index):
        audio_path = self.audio_files[index]
        waveform, sample_rate = load_audio(audio_path)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, sample_rate

class MusicDatasetHelper:
    def __init__(self, data_dir, batch_size, num_workers=4, transform=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
    def get_dataloader(self):
        dataset = MusicDataset(self.data_dir, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return dataloader
    
    def get_random_sample(self):
        dataset = MusicDataset(self.data_dir, transform=self.transform)
        index = random.randint(0, len(dataset) - 1)
        waveform, sample_rate = dataset[index]
        return waveform, sample_rate