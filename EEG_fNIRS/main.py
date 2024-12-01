import torch

from dataloader import EEGData
from model import MLPClassifier, SimpleClassifier
from train import Trainer



def main():
    # eeg_data = EEGData(root='./csv_folder/Experiment1')
    debug_data = EEGData(root='./debug')
    # num_features, num_classes = eeg_data.read_data(file_name='RFECV-5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain_R-C1-C2-N1-N2-V.csv')
    num_features, num_classes = debug_data.read_data(file_name='perfect_classification_data.csv')

    print(f">>> Num of features: {num_features}  Num of classes: {num_classes}")

    # model = MLPClassifier(input_dim=num_features, hidden_dim=32, output_dim=num_classes, lr=0.01)
    model = SimpleClassifier(input_size=num_features, num_classes=num_classes)
    trainer = Trainer(max_epochs=10, draw_online=False, img_path='./img')
    trainer.fit(model, debug_data)
    acc = trainer.validate()
    print(acc)


if __name__ == '__main__':
    main()

