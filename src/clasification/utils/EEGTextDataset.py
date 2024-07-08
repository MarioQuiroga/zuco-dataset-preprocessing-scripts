from torch.utils.data import Dataset, DataLoader

# Custom dataset
class EEGTextDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]