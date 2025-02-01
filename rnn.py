import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Define sequence data creation function
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define MASE calculation function
def calculate_mase(actual, predicted, seasonal_period=1):
    mae_forecast = np.mean(np.abs(actual - predicted))
    mae_naive = np.mean(np.abs(actual[seasonal_period:] - actual[:-seasonal_period]))
    return mae_forecast / mae_naive if mae_naive != 0 else float('inf')

# Define multi-sequence prediction model with 5 layers of LTC
class MultiFeatureRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, output_size=2):
        super(MultiFeatureRNN, self).__init__()
        
        # Define a 5-layer RNN, where each layer receives the output of the previous layer
        self.rnn_layer1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rnn_layer2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.rnn_layer3 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Sequentially pass data through each RNN layer
        out, _ = self.rnn_layer1(x)
        out, _ = self.rnn_layer2(out)
        out, _ = self.rnn_layer3(out)
        
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])  
        return out


# Training function
def train_model(train_loader, val_loader, model, criterion, optimizer, device, epochs=30):
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Move data to GPU if available
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Move data to GPU if available
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Save the best model state
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    print(f"Best model achieved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_state)
    return model,end_time - start_time

# Evaluation function
def evaluate_model(test_loader, model, device):
    model.eval()
    with torch.no_grad():
        predictions, actuals = [], []
        for X_batch, y_batch in test_loader:
            # Move data to GPU if available
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            predictions.extend(output.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate MSE and MASE
    mse_temp = mean_squared_error(actuals[:, 0], predictions[:, 0])
    mse_humidity = mean_squared_error(actuals[:, 1], predictions[:, 1])
    mase_temp = calculate_mase(actuals[:, 0], predictions[:, 0])
    mase_humidity = calculate_mase(actuals[:, 1], predictions[:, 1])

    # Get current GPU memory usage
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
    else:
        allocated_memory = reserved_memory = None

    print(f"Test set evaluation metrics (on normalized data):")
    print(f"Temperature - MSE: {mse_temp:.4f}, MASE: {mase_temp:.4f}")
    print(f"Humidity - MSE: {mse_humidity:.4f}, MASE: {mase_humidity:.4f}")

    if allocated_memory is not None and reserved_memory is not None:
        print(f"Memory allocated: {allocated_memory:.2f} MB")
        print(f"Memory reserved: {reserved_memory:.2f} MB")

    return predictions, actuals, mse_temp, mse_humidity, mase_temp, mase_humidity, allocated_memory, reserved_memory

