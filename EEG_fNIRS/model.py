import torch
import os
import matplotlib.pyplot as plt
from torch import nn

from utils import HyperParameters, ProgressBoard


class Module(nn.Module, HyperParameters):
    """
    The base class of models (may not need to read).
    """
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
    
    def loss(self, y_hat, y):
        raise NotImplementedError
    
    def forward(self, X):
        assert hasattr(self, 'net'), "Neural network is not defined"
        return self.net(X)
    
    def plot(self, key, value, train, draw_online=True, img_path=None):
        """
        Plot a point in animation.
        """
        assert hasattr(self, 'trainer'), "Trainer is not inited"
        self.board.xlabel = 'epoch'
        self.board.ylabel = key
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
            phase = 'train'
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch
            phase = 'test'

        x_value = x
        y_value = value.to(torch.device('cpu')).detach().numpy()
        label = f"{phase}_{key}"

        if draw_online:
            self.board.draw(x_value, y_value, label, every_n=int(n))
        else:
            assert img_path is not None, "img_path must be specified if draw_online is False"

            epoch = self.trainer.epoch + 1
            file_name = f"{phase}.png"
            full_path = os.path.join(img_path, file_name)

            os.makedirs(img_path, exist_ok=True)

            self.board.draw(x_value, y_value, label, every_n=int(n), img_path=full_path)
    

    def training_step(self, batch, draw_online=True, img_path=None):
        l = self.loss(self(*batch[:-1]), batch[-1]) # self(*batch[:-1]) call Module(), and calculate the prediction
        self.plot('loss', l, train=True, draw_online=draw_online, img_path=img_path)
        return l
    
    def validation_step(self, batch, draw_online=True, img_path=None):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False, draw_online=draw_online, img_path=img_path)

    def configure_optimizers(self):
        raise NotImplementedError
    

class MLPClassifier(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        fn = nn.CrossEntropyLoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class SimpleClassifier(Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def loss(self, y_hat, y):
        fn = nn.CrossEntropyLoss()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
    