from src.utils import plot_graph, show_incorrect_images
import torch

from tqdm.auto import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train_step(model, device, train_loader, optimizer, criterion, scheduler):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    scheduler.step()

  train_acc=100*correct/processed
  train_loss=train_loss/processed

  return train_loss, train_acc

def test_step(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0
    processed = 0
    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1)
            correct_mask = pred.eq(target)
            incorrect_indices = ~correct_mask

            test_incorrect_pred['images'].extend(data[incorrect_indices])
            test_incorrect_pred['ground_truths'].extend(target[incorrect_indices])
            test_incorrect_pred['predicted_vals'].extend(pred[incorrect_indices])

            correct += GetCorrectPredCount(output, target)
            processed += len(data)

    test_loss /= processed
    test_acc = 100. * correct / processed

    print('Test set: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, processed,
        test_acc))
    return test_loss, test_acc, test_incorrect_pred

def train(model, train_loader, test_loader, device, optimizer, epochs, criterion, scheduler):

    class_map = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    # Data to plot accuracy and loss graphs
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}, Learning Rate: {lr}')
        train_loss, train_acc = train_step(model=model, device=device, train_loader=train_loader, optimizer=optimizer, criterion=criterion, scheduler=scheduler)
        test_loss, test_acc, test_incorrect_pred = test_step(model=model, device=device, test_loader=test_loader, criterion=criterion)
        

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    plot_graph(results["train_loss"], results["test_loss"], results["train_acc"], results["test_acc"])
    show_incorrect_images(test_incorrect_pred, class_map)
    show_incorrect_images(test_incorrect_pred, class_map, grad_cam=True, model=model)

    # Return the filled results at the end of the epochs
    return results
