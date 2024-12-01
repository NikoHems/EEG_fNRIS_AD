import torch
import torch.utils
import torch.utils.data
import pandas as pd
import os

from utils import HyperParameters


class DataModule(HyperParameters):
    """
    The base class of data.
    """
    def __init__(self, root='../csv_folder/Experiment1'):
        self.save_hyperparameters()

    def read_data(self):
        raise NotImplementedError

    def get_dataloader(self, train):
        raise NotImplementedError
    
    def train_dataloader(self):
        return self.get_dataloader(train=True)
    
    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)
    

class EEGData(DataModule):
    """
    The class of EEG data
    """
    def __init__(self, root='../csv_folder/Experiment1', num_train=96, batch_size=16):
        super().__init__(root)
        self.batch_size = batch_size
        self.root = root
        self.num_train = num_train

    def read_data(self, file_name):
        """
        Read Data from the csv file.

        Parameters:
        file_name: the name of the file you want to take as input data. It will add the root path of it in this function

        Returns:
        num_features: number of features of the signal
        num_classes: number of class for classification
        """
        file_path = os.path.join(self.root, file_name)
        data = pd.read_csv(file_path)
        data = data.drop(columns=[data.columns[0]])

        y = data['label']
        X = data.drop(columns=['label'])

        num_classes = data['label'].nunique()
        num_features = X.shape[1]

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

        print('-'*20)
        print(f">>> Read data: X shape: {self.X.shape}  Y shape: {self.y.shape}")

        return (num_features, num_classes)

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

    
