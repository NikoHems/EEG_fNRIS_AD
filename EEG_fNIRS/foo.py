import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Load data
# file_path = "./debug/perfect_classification_data.csv"
file_path = "./csv_folder/Experiment1/RFECV-5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain_R-C1-C2-N1-N2-V.csv"
data = pd.read_csv(file_path)

# Prepare dataset
class ClassificationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.features = dataframe.iloc[:, 2:].values.astype(np.float32)  # Features start from the 3rd column
        self.labels = dataframe['label'].values.astype(np.int64)  # Label column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = ClassificationDataset(data)

# Split dataset into training and testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the model
class SimpleClassifier(nn.Module):
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

input_size = 151
num_classes = 3
model = SimpleClassifier(input_size, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Testing loop
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# Train and test the model
train_model(model, train_loader, criterion, optimizer)
test_model(model, test_loader)
