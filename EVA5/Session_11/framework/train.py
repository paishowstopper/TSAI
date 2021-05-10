
import torch
from tqdm import tqdm

class Train():
    
    def __init__(self, dataloader, model, criterion, optimizer, scheduler, device, epochs):
        
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs

    def train(self):

        train_losses = []
        train_accuracies = []
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct_trained = 0
            total_trained = 0
            for i, (data, labels) in enumerate(self.dataloader, 0):
                # get the inputs
                inputs, labels = data.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # print statistics
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_trained += torch.sum(preds == labels.data).item()
                total_trained += len(data)
            train_losses.append(running_loss)
            train_accuracies.append(100.0*correct_trained/total_trained)
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            print("Accuracy: ", 100.0*correct_trained/total_trained) 
        return train_losses, train_accuracies