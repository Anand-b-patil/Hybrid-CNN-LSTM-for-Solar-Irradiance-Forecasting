import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import torch
from datetime import datetime

class SolarVisualization:
    def __init__(self):
        # Set style for better visualizations
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

    def plot_time_series(self, df, actual_col='actual_power', pred_col='predicted_power'):
        """Plot actual vs predicted power over time"""
        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df[actual_col], label='Actual Power', linewidth=2)
        plt.plot(df.index, df[pred_col], label='Predicted Power', linestyle='--', linewidth=2)
        plt.title('Solar Power Generation: Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, df):
        """Plot correlation matrix of features"""
        correlation_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def plot_error_distribution(self, df, actual_col='actual_power', pred_col='predicted_power'):
        """Plot distribution of prediction errors"""
        df['error'] = df[actual_col] - df[pred_col]
        df['absolute_error'] = abs(df['error'])

        plt.figure(figsize=(12, 6))
        plt.hist(df['error'], bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error (kW)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self, feature_importance):
        """Plot feature importance scores"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(feature_importance.values()),
                    y=list(feature_importance.keys()),
                    palette='viridis')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

    def plot_performance_metrics(self, actual, predicted):
        """Plot model performance metrics"""
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)

        metrics = {
            'RMSE': rmse,
            'R2 Score': r2
        }

        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(metrics.values()),
                    y=list(metrics.keys()),
                    palette='muted')
        plt.title('Model Performance Metrics')
        plt.xlabel('Score')
        plt.tight_layout()
        plt.show()

    def plot_daily_pattern(self, df, power_col='actual_power'):
        """Plot average daily power generation pattern"""
        df['hour'] = df.index.hour
        daily_pattern = df.groupby('hour')[power_col].mean()

        plt.figure(figsize=(12, 6))
        plt.plot(daily_pattern.index, daily_pattern.values, marker='o')
        plt.title('Average Daily Solar Power Generation Pattern')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Power (kW)')
        plt.grid(True)
        plt.xticks(range(24))
        plt.tight_layout()
        plt.show()

    def plot_prediction_scatter(self, df, actual_col='actual_power', pred_col='predicted_power'):
        """Plot scatter plot of actual vs predicted values"""
        plt.figure(figsize=(10, 8))
        plt.scatter(df[actual_col], df[pred_col], alpha=0.5)
        plt.plot([df[actual_col].min(), df[actual_col].max()],
                [df[actual_col].min(), df[actual_col].max()],
                'r--', label='Perfect Prediction')
        plt.title('Actual vs Predicted Power')
        plt.xlabel('Actual Power (kW)')
        plt.ylabel('Predicted Power (kW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    actual_power = np.random.normal(100, 20, 100)
    predicted_power = actual_power + np.random.normal(0, 5, 100)
    temperature = np.random.normal(25, 5, 100)
    humidity = np.random.normal(60, 10, 100)

    df = pd.DataFrame({
        'datetime': dates,
        'actual_power': actual_power,
        'predicted_power': predicted_power,
        'temperature': temperature,
        'humidity': humidity
    })
    df.set_index('datetime', inplace=True)

    # Initialize visualization class
    viz = SolarVisualization()

    # Generate all visualizations
    viz.plot_time_series(df)
    viz.plot_correlation_matrix(df)
    viz.plot_error_distribution(df)
    
    # Example feature importance
    feature_importance = {
        'temperature': 0.4,
        'humidity': 0.3,
        'time_of_day': 0.2,
        'cloud_cover': 0.1
    }
    viz.plot_feature_importance(feature_importance)
    
    viz.plot_performance_metrics(df['actual_power'], df['predicted_power'])
    viz.plot_daily_pattern(df)
    viz.plot_prediction_scatter(df) 