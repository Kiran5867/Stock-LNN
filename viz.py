import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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

def visualize_model_results(results_dict):

    model_names = list(results_dict.keys())
    
    metrics_to_visualize = ['train_time', 'allocated_memory', 'reserved_memory', 'epoch_dx_dt_history',
                            'mse_open', 'mse_close', 'mase_open', 'mase_close']
    
    # Create subplots with 2 columns and one metric per subplot
    num_metrics = len(metrics_to_visualize)
    fig = make_subplots(rows=(num_metrics + 1) // 2, cols=2, 
                        subplot_titles=[metric.replace('_', ' ').title() for metric in metrics_to_visualize])

    for i, metric in enumerate(metrics_to_visualize):
        # Determine the current subplot position
        row = i // 2 + 1
        col = i % 2 + 1

        for model in model_names:
            metrics = results_dict[model]

            if metric in ['train_time', 'allocated_memory', 'reserved_memory', 'mse_open', 'mse_close', 'mase_open', 'mase_close']:
                value = metrics.get(metric)
                if value is not None:

                    fig.add_trace(go.Bar(
                        x=[model], 
                        y=[value], 
                        name=metric.replace('_', ' ').title(),
                        text=[f"{value:.4f}" if metric != 'epoch_dx_dt_history' else ""],  
                        texttemplate='%{text}',
                        textposition='outside'
                    ), row=row, col=col)

            elif metric == 'epoch_dx_dt_history' and model != 'rnn': 
                dx_dt_values = [entry['total_avg_dx_dt'] for entry in metrics['epoch_dx_dt_history']]
                epochs = list(range(1, len(dx_dt_values) + 1))  

                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=dx_dt_values, 
                    mode='lines+markers',
                    name=model,
                    text=[f"{val:.6f}" for val in dx_dt_values],
                    textposition='top center'
                ), row=row, col=col)

        fig.update_xaxes(title_text="Models" if metric != 'epoch_dx_dt_history' else "Epochs", row=row, col=col)
        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=row, col=col)

    fig.update_layout(
        title="Model Performance Metrics",
        legend_title="Metrics",
        height=1600,  
        showlegend=True  
    )

    fig.show()

# Usage example
# visualize_model_results(results_dict)


# Plot results function
def plot_results(predictions, actuals, title="Comparison of Predicted and Actual Values (Normalized)"):
    results = pd.DataFrame({
        'Predicted Open': predictions[:, 0],
        'Actual Open': actuals[:, 0],
        'Predicted Close': predictions[:, 1],
        'Actual Close': actuals[:, 1]
    })
    
    plt.figure(figsize=(12, 6))
    plt.plot(results['Actual Open'], label='Actual Open', color='blue')
    plt.plot(results['Predicted Open'], label='Predicted Open', color='red', linestyle='--')
    plt.title(f'Open - {title}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Open')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(results['Actual Close'], label='Actual Close', color='blue')
    plt.plot(results['Predicted Close'], label='Predicted Close', color='red', linestyle='--')
    plt.title(f'Close - {title}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Close')
    plt.legend()
    plt.show()
    
# Visualization function to show the trend of dx/dt for each layer
def plot_dx_dt_history(epoch_dx_dt_history):

    epochs = list(range(1, len(epoch_dx_dt_history) + 1))
    
    # Get the names of the layers (e.g., 'layer1', 'layer2', 'layer3')
    layer_names = epoch_dx_dt_history[0].keys()
    
    fig = go.Figure()

    # Add a line for each layer to show the trend of dx/dt
    for layer in layer_names:
        # Collect the average dx/dt values for the current layer in each epoch
        layer_dx_dt_values = [epoch_data[layer] for epoch_data in epoch_dx_dt_history]
        
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=layer_dx_dt_values, 
            mode='lines+markers',
            name=f'{layer} dx/dt',
            line=dict(width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title="Trend of dx/dt for Each Layer Across Epochs",
        xaxis_title="Epoch",
        yaxis_title="dx/dt Average Value",
        legend_title="Layer Name"
    )

    fig.show()
