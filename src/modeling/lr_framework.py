import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.optim import Adam
from plot_results import plot_array_values_against_length

from modeling.lr_dataloaders import create_dataloaders

class LRFramework:
    def __init__(self, config:dict, model:nn.Module=LogisticRegression):
        self.model = model(input_dim=config['hyperparameters']['input_dim'], output_dim=config['hyperparameters']['output_dim'])
        self.epochs = config['hyperparameters']['epochs']
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config['hyperparameters']['lr'])
        self.config = config
        
    def fit(self, raw_data_path):
        self.train_dl, self.test_dl = create_dataloaders(config=self.config, raw_data_path=raw_data_path)

        self.criterion = torch.nn.CrossEntropyLoss(weight=self.get_class_weights(self.train_dl, self.test_dl))

        train_losses, train_accuracies, train_F1s = [], [], []
        test_losses, test_accuracies, test_F1s = [], [], []

        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
          
            train_loss, train_accuracy, train_F1 = self.train(self.train_dl)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy) 
            train_F1s.append(train_F1)
        
            test_loss, test_accuracy, test_F1 = self.validate(self.test_dl)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy) 
            test_F1s.append(test_F1)

        plot_array_values_against_length([train_losses, test_losses], "Loss vs Epochs - 'Stance Detection' - Logistic Regression")
        plot_array_values_against_length([train_accuracies, test_accuracies], "Accuracy vs Epochs - 'Stance Detection' - Logistic Regression")
        plot_array_values_against_length([train_F1s, test_F1s], "F1 Score vs Epochs - 'Stance Detection' - Logistic Regression")

    def train(self, dl):
        self.model.train()
        train_loss = 0
        total_correct = 0
        total_labels = []
        total_preds = []
        for _, batch in enumerate(dl):
            features, labels = batch
            features = np.hstack([np.ones([features.shape[0],1])*10, features])

            pred_logits = self.model(torch.tensor(features.astype(np.float32)))
            loss = self.criterion(torch.squeeze(pred_logits), labels.long())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            total_correct += sum(torch.argmax(torch.squeeze(pred_logits), axis=1) == torch.squeeze(labels.float()))
            total_preds += list(torch.argmax(torch.squeeze(pred_logits), axis=1).cpu().detach().numpy())
            total_labels += list(labels.float().cpu().numpy())

        accuracy = total_correct / len(total_preds)
        total_loss = train_loss / len(total_preds)
        F1_global = metrics.f1_score(total_labels, total_preds).item()
        print(f"loss: {total_loss}, accuracy:{accuracy}, F1:{F1_global}")
        return total_loss, accuracy, F1_global
            
    def validate(self, dl):
        self.model.eval()
        test_loss = 0
        total_correct = 0
        total_labels = []
        total_preds = []
        for _, batch in enumerate(dl):
            features, labels = batch
            features = np.hstack([np.ones([features.shape[0],1])*10, features])

            pred_logits = self.model(torch.tensor(features.astype(np.float32)))
            loss = self.criterion(torch.round(torch.squeeze(pred_logits)), labels.long())

            test_loss += loss.item()
            total_correct += sum(torch.argmax(torch.squeeze(pred_logits), axis=1) == torch.squeeze(labels.float()))
            total_preds += list(torch.argmax(torch.squeeze(pred_logits), axis=1).cpu().detach().numpy())
            total_labels += list(labels.float().cpu().numpy())
    
        accuracy = total_correct / len(total_preds)
        total_loss = test_loss / len(total_preds)
        F1_global = metrics.f1_score(total_labels, total_preds).item()
        print(f"loss: {total_loss}, accuracy:{accuracy}, F1:{F1_global}")
        return total_loss, accuracy, F1_global

    def get_class_weights(self, train_dl, test_dl, classes:int=2):
        arr = torch.zeros(classes)
        for _, labels in train_dl:
            for label in labels:
                arr[label] += 1
        for _, labels in test_dl:
            for label in labels:
                arr[label] += 1
        arrmax = arr.max().expand(classes)
        return arrmax / arr
