import torch

class Test():
    
    def __init__(self, dataloader, model, criterion, device):
        
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.device = device

    def test(self):

        correct = 0
        total = 0
        test_loss = 0
        running_loss = 0
        with torch.no_grad():
            for data, label in self.dataloader:
                images, labels = data.to(self.device), label.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss = self.criterion(outputs, labels) 
                running_loss += test_loss.item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))