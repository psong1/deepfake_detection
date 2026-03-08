import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
import copy

from download_dataset import download_dataset
from plot_utils import plot_training_history
from model_utils import get_dataloaders, get_model 

def train_model(model, loaders, criterion, optimizer, num_epochs=10):
  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  train_loader = loaders['Train']
  val_loader = loaders['Validation']

  # For plotting
  history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

  use_amp = device.type == 'cuda'
  scaler = torch.amp.GradScaler('cuda') if use_amp else None

  for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['Train', 'Validation']:
      if phase == 'Train':
        model.train() # Set model to training mode
        dataloader = train_loader
      else:
          model.eval()
          dataloader = val_loader

      print(f'  {phase}ing...')
      running_loss = 0.0
      running_corrects = 0

      # Iterate over data (non_blocking=True overlaps transfer with compute when pin_memory)
      for batch_idx, (inputs, labels) in enumerate(dataloader):
        if (batch_idx + 1) % 500 == 0 or batch_idx == 0:
          print(f'    Batch {batch_idx + 1}/{len(dataloader)}')
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        with torch.set_grad_enabled(phase == 'Train'):
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            if phase == 'Train':
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / len(dataloader.dataset)
      epoch_acc = running_corrects.double() / len(dataloader.dataset)

      print(f'{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

      # Save history for plotting
      if phase == 'Train':
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
      else:
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())

      # Deep copy the mode if it's the best one yet
      if phase == 'Validation' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    
    print()

  time_elapsed = time.time() - since
  print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
  print(f'Best val Acc: {best_acc:4f}')

  # Load best model weights
  model.load_state_dict(best_model_wts)
  return model, history

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))

    # Initialize components
    download_dataset()
    loaders = get_dataloaders('./Dataset')
    deepfake_model = get_model().to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(deepfake_model.fc.parameters(), lr=0.001)

    # Train the model
    trained_model, history = train_model(deepfake_model, loaders, criterion, optimizer, num_epochs=10)

    # Final evaluation
    print('Running final evaluation on Test set...')
    trained_model.eval()
    test_corrects = 0

    with torch.no_grad():
        for inputs, labels in loaders['Test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    
    test_acc = test_corrects.double() / len(loaders['Test'].dataset)
    print(f'Test Accuracy: {test_acc:.4f}')

    # Save final result
    torch.save(trained_model.state_dict(), 'deepfake_model.pth')
    print('Model saved to deepfake_model.pth')

    plot_training_history(history)