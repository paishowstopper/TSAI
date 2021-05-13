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

        for epoch in range(self.epochs):
            running_loss = 0.0
            pbar = tqdm(self.dataloader, total=len(self.dataloader))
            for i, (data, labels) in enumerate(pbar):
                # get the inputs
                inputs, labels = data.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
            # self.scheduler.step()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))