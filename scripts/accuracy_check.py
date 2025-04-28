import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
from datetime import datetime, timedelta

class ModelAccuracyChecker:
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

    def calculate_metrics(self, y_true, y_pred):
        """Calculate various accuracy metrics"""
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'R2': r2_score(y_true, y_pred),
            'Max Error': np.max(np.abs(y_true - y_pred)),
            'Mean Error': np.mean(y_true - y_pred),
            'Std Error': np.std(y_true - y_pred)
        }
        return metrics

    def plot_prediction_comparison(self, y_true, y_pred, title='Actual vs Predicted Values'):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', label='Perfect Prediction')
        plt.title(title)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_error_distribution(self, y_true, y_pred):
        """Plot distribution of prediction errors"""
        errors = y_true - y_pred
        plt.figure(figsize=(12, 6))
        sns.histplot(data=errors, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def plot_residuals(self, y_true, y_pred):
        """Plot residuals vs predicted values"""
        residuals = y_true - y_pred
        plt.figure(figsize=(12, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs Predicted Values')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_time_series_comparison(self, y_true, y_pred, dates=None):
        """Plot time series comparison of actual and predicted values"""
        plt.figure(figsize=(15, 6))
        if dates is not None:
            plt.plot(dates, y_true, label='Actual', linewidth=2)
            plt.plot(dates, y_pred, label='Predicted', linestyle='--', linewidth=2)
        else:
            plt.plot(y_true, label='Actual', linewidth=2)
            plt.plot(y_pred, label='Predicted', linestyle='--', linewidth=2)
        plt.title('Time Series Comparison: Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_error_by_magnitude(self, y_true, y_pred):
        """Plot absolute error vs actual value magnitude"""
        absolute_errors = np.abs(y_true - y_pred)
        plt.figure(figsize=(12, 6))
        plt.scatter(y_true, absolute_errors, alpha=0.5)
        plt.title('Absolute Error vs Actual Value Magnitude')
        plt.xlabel('Actual Value')
        plt.ylabel('Absolute Error')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_cumulative_error(self, y_true, y_pred):
        """Plot cumulative error over time"""
        cumulative_error = np.cumsum(y_true - y_pred)
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_error)
        plt.title('Cumulative Error Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative Error')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def analyze_error_patterns(self, y_true, y_pred, dates=None, weather_data=None):
        """Analyze patterns in prediction errors"""
        errors = y_true - y_pred
        absolute_errors = np.abs(errors)
        
        # Create error analysis DataFrame
        error_df = pd.DataFrame({
            'error': errors,
            'absolute_error': absolute_errors,
            'relative_error': absolute_errors / y_true * 100
        })
        
        if dates is not None:
            error_df['datetime'] = dates
            error_df.set_index('datetime', inplace=True)
            
            # Analyze errors by time of day
            error_df['hour'] = error_df.index.hour
            hourly_errors = error_df.groupby('hour')['absolute_error'].mean()
            
            plt.figure(figsize=(12, 6))
            plt.plot(hourly_errors.index, hourly_errors.values, marker='o')
            plt.title('Average Absolute Error by Hour of Day')
            plt.xlabel('Hour of Day')
            plt.ylabel('Average Absolute Error')
            plt.grid(True)
            plt.show()
            
            # Analyze errors by day of week
            error_df['day_of_week'] = error_df.index.dayofweek
            daily_errors = error_df.groupby('day_of_week')['absolute_error'].mean()
            
            plt.figure(figsize=(12, 6))
            plt.plot(daily_errors.index, daily_errors.values, marker='o')
            plt.title('Average Absolute Error by Day of Week')
            plt.xlabel('Day of Week')
            plt.ylabel('Average Absolute Error')
            plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            plt.grid(True)
            plt.show()
        
        # Analyze error distribution by magnitude
        error_df['magnitude_bin'] = pd.cut(y_true, bins=10)
        magnitude_errors = error_df.groupby('magnitude_bin')['absolute_error'].mean()
        
        plt.figure(figsize=(12, 6))
        magnitude_errors.plot(kind='bar')
        plt.title('Average Absolute Error by Power Magnitude')
        plt.xlabel('Power Magnitude Range')
        plt.ylabel('Average Absolute Error')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
        
        # Identify worst predictions
        worst_predictions = error_df.nlargest(10, 'absolute_error')
        print("\nTop 10 Worst Predictions:")
        print(worst_predictions[['error', 'absolute_error', 'relative_error']])
        
        return error_df

    def analyze_performance_over_time(self, y_true, y_pred, dates, window_size=7):
        """Analyze model performance over time using rolling windows"""
        if dates is None:
            raise ValueError("Dates are required for performance over time analysis")
            
        # Create performance DataFrame
        performance_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'error': y_true - y_pred,
            'absolute_error': np.abs(y_true - y_pred)
        }, index=dates)
        
        # Calculate rolling metrics
        performance_df['rolling_mae'] = performance_df['absolute_error'].rolling(window=window_size).mean()
        performance_df['rolling_rmse'] = performance_df['error'].rolling(window=window_size).apply(
            lambda x: np.sqrt(np.mean(x**2))
        )
        performance_df['rolling_r2'] = performance_df.rolling(window=window_size).apply(
            lambda x: r2_score(x['y_true'], x['y_pred'])
        )
        
        # Plot rolling metrics
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        performance_df['rolling_mae'].plot(ax=axes[0])
        axes[0].set_title('Rolling MAE')
        axes[0].set_ylabel('MAE')
        axes[0].grid(True)
        
        performance_df['rolling_rmse'].plot(ax=axes[1])
        axes[1].set_title('Rolling RMSE')
        axes[1].set_ylabel('RMSE')
        axes[1].grid(True)
        
        performance_df['rolling_r2'].plot(ax=axes[2])
        axes[2].set_title('Rolling R² Score')
        axes[2].set_ylabel('R²')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Identify periods of poor performance
        poor_performance = performance_df[performance_df['rolling_mae'] > performance_df['rolling_mae'].mean() + performance_df['rolling_mae'].std()]
        print("\nPeriods of Poor Performance:")
        print(poor_performance[['rolling_mae', 'rolling_rmse', 'rolling_r2']])
        
        return performance_df

    def analyze_weather_impact(self, y_true, y_pred, weather_data):
        """Analyze impact of weather conditions on prediction accuracy"""
        if weather_data is None:
            raise ValueError("Weather data is required for weather impact analysis")
            
        # Calculate errors
        errors = np.abs(y_true - y_pred)
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'error': errors,
            'temperature': weather_data.get('temperature', None),
            'humidity': weather_data.get('humidity', None),
            'cloud_cover': weather_data.get('cloud_cover', None),
            'wind_speed': weather_data.get('wind_speed', None)
        })
        
        # Plot error vs weather variables
        weather_vars = ['temperature', 'humidity', 'cloud_cover', 'wind_speed']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(weather_vars):
            if var in analysis_df.columns:
                sns.scatterplot(data=analysis_df, x=var, y='error', ax=axes[i])
                axes[i].set_title(f'Error vs {var.title()}')
                axes[i].set_xlabel(var.title())
                axes[i].set_ylabel('Absolute Error')
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate correlations
        correlations = analysis_df.corr()['error'].drop('error')
        print("\nCorrelation between Errors and Weather Variables:")
        print(correlations)
        
        return analysis_df

    def analyze_accuracy(self, y_true, y_pred, dates=None, weather_data=None):
        """Perform comprehensive accuracy analysis"""
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        print("\nModel Accuracy Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Generate all plots
        self.plot_prediction_comparison(y_true, y_pred)
        self.plot_error_distribution(y_true, y_pred)
        self.plot_residuals(y_true, y_pred)
        self.plot_time_series_comparison(y_true, y_pred, dates)
        self.plot_error_by_magnitude(y_true, y_pred)
        self.plot_cumulative_error(y_true, y_pred)
        
        # Perform detailed error analysis
        error_df = self.analyze_error_patterns(y_true, y_pred, dates, weather_data)
        
        # Analyze performance over time if dates are provided
        if dates is not None:
            performance_df = self.analyze_performance_over_time(y_true, y_pred, dates)
        
        # Analyze weather impact if weather data is provided
        if weather_data is not None:
            weather_analysis = self.analyze_weather_impact(y_true, y_pred, weather_data)

        return metrics, error_df

# Example usage
if __name__ == "__main__":
    # Initialize the checker
    checker = ModelAccuracyChecker()
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    y_true = np.random.normal(100, 20, n_samples)
    y_pred = y_true + np.random.normal(0, 5, n_samples)  # Add some noise
    
    # Generate sample weather data
    weather_data = {
        'temperature': np.random.normal(25, 5, n_samples),
        'humidity': np.random.normal(60, 10, n_samples),
        'cloud_cover': np.random.uniform(0, 100, n_samples),
        'wind_speed': np.random.normal(10, 3, n_samples)
    }
    
    # Perform comprehensive accuracy analysis
    metrics, error_df = checker.analyze_accuracy(y_true, y_pred, dates, weather_data)
    
    # Print detailed metrics
    print("\nDetailed Analysis:")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
    print(f"R-squared Score: {metrics['R2']:.4f}")
    print(f"Maximum Error: {metrics['Max Error']:.2f}")
    print(f"Mean Error: {metrics['Mean Error']:.2f}")
    print(f"Standard Deviation of Error: {metrics['Std Error']:.2f}") 