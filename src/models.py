from distutils.command.config import config
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = nn.Linear(input_dim, output_dim)

     def forward(self, x):
         outputs = torch.round(torch.sigmoid(self.linear(x)))
         return outputs


class LogisticRegressionExecute:
    def __init__(self, config:dict={}, optimizer:torch.optim.Optimizer=Adam, model:nn.Module=LogisticRegression):
        self.model = model(input_dim=3, output_dim=1) #TODO: fix to set from outside based on actual number of features
        self.epochs = config['hyperparameters']['epochs']
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=config["hyperparameters"]["learning_rate"])
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def fit(self, train_dl):
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            self.train(train_dl)

    def train(self, train_dl):
        train_loss = 0

        for _, batch in enumerate(train_dl):
            features, labels = batch
            self.optimizer.zero_grad()
            pred_logits = self.model(features.float())
            loss = self.criterion(torch.squeeze(pred_logits), labels.float())
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        print(train_loss)
            
            

            

    def validate(self):
        pass


    