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
import plotly.graph_objs as go

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

class LTC(nn.Module):
    def __init__(self, input_size, hidden_size, tau=1.0):
        super(LTC, self).__init__()
        self.hidden_size = hidden_size
        self.input_weights = nn.Linear(input_size, hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, hidden_size)
        self.tau = tau

    def forward(self, x, hidden_state):
        input_effect = self.input_weights(x) # x = I(t)
        hidden_effect = self.hidden_weights(hidden_state) # hidden state = X(t)
        combined = input_effect + hidden_effect # combined = A
        
        time_constant_effect = torch.sigmoid(combined) # time_constant_effect = f( x(t), I(t), t, theta )
        dynamic_time_constants = torch.clamp(self.tau / (1 + self.tau * time_constant_effect), min=0.1, max=1.0) # dynamic_time_constants = toll_sys
        
        # Calculate dx/dt
        dx_dt = time_constant_effect * combined - hidden_state / dynamic_time_constants # dx_dt = f( x(t), I(t), t, theta) * A - x(t) / toll_sys
        
        updated_hidden = hidden_state + dx_dt
        return updated_hidden, dx_dt

    def initialize_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Updated multi-layer LTC model, collecting dx/dt values
class MultiSequenceLTCModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=30, output_size=2, tau1=1.0, tau2=1.0, tau3=1.0):
        super(MultiSequenceLTCModel, self).__init__()
        self.ltc_layer1 = LTC(input_size, hidden_size, tau=tau1)
        self.ltc_layer2 = LTC(hidden_size, hidden_size, tau=tau2)
        self.ltc_layer3 = LTC(hidden_size, hidden_size, tau=tau3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hidden_state1 = self.ltc_layer1.initialize_hidden_state(batch_size).to(x.device)
        hidden_state2 = self.ltc_layer2.initialize_hidden_state(batch_size).to(x.device)
        hidden_state3 = self.ltc_layer3.initialize_hidden_state(batch_size).to(x.device)

        dx_dt_values = {'layer1': [], 'layer2': [], 'layer3': []}
        
        for t in range(seq_length):
            hidden_state1, dx_dt1 = self.ltc_layer1(x[:, t, :], hidden_state1)
            hidden_state2, dx_dt2 = self.ltc_layer2(hidden_state1, hidden_state2)
            hidden_state3, dx_dt3 = self.ltc_layer3(hidden_state2, hidden_state3)
            
            # Collect dx/dt values for each layer
            dx_dt_values['layer1'].append(dx_dt1)
            dx_dt_values['layer2'].append(dx_dt2)
            dx_dt_values['layer3'].append(dx_dt3)
        
        out = self.fc(hidden_state3)
        return out, dx_dt_values
    
class ParallelMultiScaleLTCModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=30, output_size=2, tau1=0.7, tau2=1.0, tau3=10.0, tau4=1.0):
        super(ParallelMultiScaleLTCModel, self).__init__()
        self.ltc_layer1_ltc1 = LTC(input_size, hidden_size, tau=tau1)
        self.ltc_layer1_ltc2 = LTC(input_size, hidden_size, tau=tau2)
        self.ltc_layer1_ltc3 = LTC(input_size, hidden_size, tau=tau3)
        self.ltc_layer2 = LTC(hidden_size*3, hidden_size, tau=tau4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hidden_state1 = self.ltc_layer1_ltc1.initialize_hidden_state(batch_size).to(x.device)
        hidden_state2 = self.ltc_layer1_ltc2.initialize_hidden_state(batch_size).to(x.device)
        hidden_state3 = self.ltc_layer1_ltc3.initialize_hidden_state(batch_size).to(x.device)
        hidden_state4 = self.ltc_layer2.initialize_hidden_state(batch_size).to(x.device)

        dx_dt_values = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
        
        for t in range(seq_length):
            hidden_state1, dx_dt1 = self.ltc_layer1_ltc1(x[:, t, :], hidden_state1)
            hidden_state2, dx_dt2 = self.ltc_layer1_ltc2(x[:, t, :], hidden_state2)
            hidden_state3, dx_dt3 = self.ltc_layer1_ltc3(x[:, t, :], hidden_state3)
            hidden_state4, dx_dt4 = self.ltc_layer2(torch.cat((hidden_state1, hidden_state2, hidden_state3), dim=1), hidden_state4)
            

            # Collect dx/dt values for each layer
            dx_dt_values['layer1'].append(dx_dt1)
            dx_dt_values['layer2'].append(dx_dt2)
            dx_dt_values['layer3'].append(dx_dt3)
            dx_dt_values['layer4'].append(dx_dt4)

        out = self.fc(hidden_state4)
        return out, dx_dt_values
    
class ParallelMultiScaleLTCModel2(nn.Module):
    def __init__(self, input_size=2, hidden_size=30, output_size=2, tau1=0.7, tau2=1.0, tau3=10.0, tau4=1.0):
        super(ParallelMultiScaleLTCModel2, self).__init__()
        self.ltc_layer1_ltc1 = LTC(input_size, hidden_size, tau=tau1)
        self.ltc_layer1_ltc2 = LTC(input_size, hidden_size, tau=tau2)
        self.ltc_layer1_ltc3 = LTC(input_size, hidden_size, tau=tau3)
        self.linearCat_layer = nn.Linear(hidden_size*3, hidden_size)
        self.ltc_layer2 = LTC(hidden_size, hidden_size, tau=tau4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hidden_state1 = self.ltc_layer1_ltc1.initialize_hidden_state(batch_size).to(x.device)
        hidden_state2 = self.ltc_layer1_ltc2.initialize_hidden_state(batch_size).to(x.device)
        hidden_state3 = self.ltc_layer1_ltc3.initialize_hidden_state(batch_size).to(x.device)
        hidden_state4 = self.ltc_layer2.initialize_hidden_state(batch_size).to(x.device)

        dx_dt_values = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
        
        for t in range(seq_length):
            hidden_state1, dx_dt1 = self.ltc_layer1_ltc1(x[:, t, :], hidden_state1)
            hidden_state2, dx_dt2 = self.ltc_layer1_ltc2(x[:, t, :], hidden_state2)
            hidden_state3, dx_dt3 = self.ltc_layer1_ltc3(x[:, t, :], hidden_state3)
            linearCat_out = self.linearCat_layer(torch.cat((hidden_state1, hidden_state2, hidden_state3), dim=1))
            hidden_state4, dx_dt4 = self.ltc_layer2(linearCat_out , hidden_state4)
            

            # Collect dx/dt values for each layer
            dx_dt_values['layer1'].append(dx_dt1)
            dx_dt_values['layer2'].append(dx_dt2)
            dx_dt_values['layer3'].append(dx_dt3)
            dx_dt_values['layer4'].append(dx_dt4)

        out = self.fc(hidden_state4)
        return out, dx_dt_values
    
def train_LTC_model(train_loader, val_loader, model, criterion, optimizer, device, epochs=30):
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    epoch_dx_dt_history = []  # Records the average dx/dt values for each epoch

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        epoch_dx_dt_values = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to GPU
            optimizer.zero_grad()
            output, dx_dt_values = model(X_batch)  # Assuming the model returns output and dx_dt_values
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Collect the average dx/dt values for each batch
            for layer, values in dx_dt_values.items():
                epoch_dx_dt_values[layer].append(torch.stack(values).mean().item())
        
        # Calculate and record the average dx/dt values for each layer across the entire epoch
        avg_dx_dt = {layer: sum(values) / len(values) for layer, values in epoch_dx_dt_values.items()}
        
        # Calculate the average of the total dx/dt values across the three layers
        total_avg_dx_dt = sum(avg_dx_dt.values()) / len(avg_dx_dt)
        
        epoch_dx_dt_history.append({'total_avg_dx_dt': total_avg_dx_dt})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)  
                output, _ = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        print(f'Average dx/dt values for each layer: {avg_dx_dt}, Average dx/dt value across three layers: {total_avg_dx_dt:.4f}')

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    print(f"Best model at epoch {best_epoch}, Validation Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_state)
    
    return model, end_time - start_time, epoch_dx_dt_history


def evaluate_LTC_model(test_loader, model, device):
    model.eval()
    with torch.no_grad():
        predictions, actuals = [], []
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  
            output, _ = model(X_batch)
            predictions.extend(output.cpu().numpy()) 
            actuals.extend(y_batch.cpu().numpy())  

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate MSE and MASE
    mse_temp = mean_squared_error(actuals[:, 0], predictions[:, 0])
    mse_humidity = mean_squared_error(actuals[:, 1], predictions[:, 1])
    mase_temp = calculate_mase(actuals[:, 0], predictions[:, 0])
    mase_humidity = calculate_mase(actuals[:, 1], predictions[:, 1])

    # Get the current GPU memory usage
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)  
    else:
        allocated_memory = reserved_memory = None

    print("Test set evaluation metrics (on normalized data):")
    print(f"Temperature - MSE: {mse_temp:.4f}, MASE: {mase_temp:.4f}")
    print(f"Humidity - MSE: {mse_humidity:.4f}, MASE: {mase_humidity:.4f}")

    if allocated_memory is not None and reserved_memory is not None:
        print(f"Memory allocated: {allocated_memory:.2f} MB")
        print(f"Memory reserved: {reserved_memory:.2f} MB")

    return predictions, actuals, mse_temp, mse_humidity, mase_temp, mase_humidity, allocated_memory, reserved_memory

