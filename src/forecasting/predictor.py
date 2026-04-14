import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta

class ThroughputPredictor:
    def __init__(self):
        """
        Initializes the Throughput Predictor module.
        Uses lightweight Exponential Smoothing for edge-compatible time-series forecasting.
        """
        self.model = None
        self.last_train_data = None

    def train_or_update(self, historical_df):
        """
        Fits a lightweight forecasting model on the provided historical data.
        
        :param historical_df: DataFrame with 'timestamp', 'passenger_count', and 'upcoming_flight_load'.
        """
        if len(historical_df) < 5:
            # Not enough data points to train a meaningful model yet
            return False

        # Prepare data for time-series forecasting
        # Note: In a real-world edge scenario, we use Exponential Smoothing as it's 
        # computationally inexpensive and effective for short-term trends.
        data = historical_df['passenger_count'].astype(float).values
        
        try:
            # Simple Exponential Smoothing for rapid updates
            self.model = ExponentialSmoothing(
                data, 
                trend='add', 
                seasonal=None, 
                initialization_method="estimated"
            ).fit()
            self.last_train_data = historical_df
            return True
        except Exception as e:
            print(f"Model training error: {e}")
            return False

    def predict_next_30_mins(self, current_state_df):
        """
        Predicts queue size for the next 30 minutes in 5-minute intervals.
        
        :param current_state_df: The most recent DataFrame from QueueDataMerger.
        :return: Dictionary with forecasted values.
        """
        if self.model is None:
            # Fallback if model isn't trained: return current count as a flat line
            current_count = current_state_df['passenger_count'].iloc[-1]
            return {f"+{i*5}m": int(current_count) for i in range(1, 7)}

        # Forecast next 6 steps (assuming 5-minute intervals)
        forecast = self.model.forecast(6)
        
        # Post-processing: Incorporate 'upcoming_flight_load' as a multiplier/bias
        # Feature Engineering Logic: If flight load in the next hour is high, 
        # we adjust the statistical forecast upward.
        last_flight_load = current_state_df['upcoming_flight_load'].iloc[-1]
        load_multiplier = 1.0 + (last_flight_load / 1000.0) # Simple heuristic for edge
        
        predictions = {}
        for i, val in enumerate(forecast):
            interval = (i + 1) * 5
            # Ensure no negative values and apply flight load bias
            adjusted_val = max(0, int(val * load_multiplier))
            predictions[f"+{interval}m"] = adjusted_val
            
        return predictions

if __name__ == "__main__":
    # Local verification with mock data
    predictor = ThroughputPredictor()
    
    # Create mock historical data (12 points = 1 hour of 5-min intervals)
    mock_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2026-04-14', periods=12, freq='5min'),
        'passenger_count': [40, 42, 45, 48, 55, 60, 62, 65, 70, 75, 78, 80],
        'upcoming_flight_load': [500] * 12
    })
    
    if predictor.train_or_update(mock_data):
        forecast_results = predictor.predict_next_30_mins(mock_data)
        print("Queue Forecast for Next 30 Minutes:")
        print(forecast_results)
