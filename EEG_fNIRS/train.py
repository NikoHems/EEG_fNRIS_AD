import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

from utils import HyperParameters


class Trainer(HyperParameters):
    """
    The base class for training models with data.
    """
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0, draw_online=True, img_path=None):
        self.save_hyperparameters()
        self.draw_online = draw_online
        self.img_path = img_path
    
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
        
    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def prepare_batch(self, batch):
        return batch
    
    def fit_epoch(self):
        self.model.train()

        total_loss = 0

        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch), self.draw_online, self.img_path)
            total_loss += loss.item()
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        
        print(f"Epoch {self.epoch + 1} - Training Loss: {total_loss / len(self.train_dataloader)}")

        if self.val_dataloader is None:
            return
        
        self.model.eval()
        total_val_loss = 0

        for batch in self.val_dataloader:
            with torch.no_grad():
                loss = self.model.validation_step(self.prepare_batch(batch), self.draw_online, self.img_path)
                total_val_loss += loss
            self.val_batch_idx += 1

        print(f"Epoch {self.epoch + 1} - Validation Loss: {total_val_loss / len(self.val_dataloader)}")

    def validate(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, labels = self.prepare_batch(batch)[:-1], self.prepare_batch(batch)[-1]
                preds = self.model(*inputs)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        # Flatten predictions and labels
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Compute Accuracy
        predictions = all_preds.argmax(axis=1)
        correct = (predictions == all_labels).sum()
        total = len(all_labels)
        accuracy = correct / total

        return accuracy