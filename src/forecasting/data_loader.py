import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class QueueDataMerger:
    def __init__(self):
        """
        Initializes the Data Merger module for SmartFlow Airport.
        Responsible for combining vision-based crowd counts with flight schedules.
        """
        # Historical buffer to store real-time counts [timestamp, passenger_count]
        self.history_df = pd.DataFrame(columns=['timestamp', 'passenger_count'])
        self.history_df['timestamp'] = pd.to_datetime(self.history_df['timestamp'])
        
        # Mock Flight Schedule
        self.flight_schedule = self.generate_mock_flights()

    def generate_mock_flights(self):
        """
        Generates a synthetic flight schedule for testing.
        In production, this would load from an external API or database.
        
        :return: DataFrame [Flight_ID, Departure_Time, Gate, Expected_Passengers]
        """
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        data = []
        for i in range(24):  # Next 24 hours of flights
            departure = now + timedelta(hours=i, minutes=np.random.choice([0, 15, 30, 45]))
            data.append({
                "Flight_ID": f"SF{100 + i}",
                "Departure_Time": departure,
                "Gate": f"G{np.random.randint(1, 20)}",
                "Expected_Passengers": np.random.randint(100, 250)
            })
        
        df = pd.DataFrame(data)
        df['Departure_Time'] = pd.to_datetime(df['Departure_Time'])
        return df

    def merge_realtime_data(self, current_timestamp, current_queue_count):
        """
        Appends real-time camera counts to historical data and aligns with flights.
        
        :param current_timestamp: datetime object of the reading.
        :param current_queue_count: Integer count from the vision module.
        :return: Merged DataFrame containing history + engineered flight features.
        """
        # 1. Append new data point
        new_row = pd.DataFrame([{
            'timestamp': pd.to_datetime(current_timestamp), 
            'passenger_count': current_queue_count
        }])
        self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)
        
        # Keep only last 24 hours to prevent memory bloat on edge
        cutoff = pd.to_datetime(current_timestamp) - timedelta(hours=24)
        self.history_df = self.history_df[self.history_df['timestamp'] > cutoff]

        # 2. Feature Engineering: Impact of upcoming flights
        # We calculate 'Upcoming_Load' - the sum of expected passengers for flights 
        # departing in the next 60-120 minutes (typical check-in window).
        merged_data = self.history_df.copy()
        merged_data['upcoming_flight_load'] = merged_data['timestamp'].apply(
            lambda x: self._calculate_flight_impact(x)
        )
        
        return merged_data

    def _calculate_flight_impact(self, timestamp):
        """
        Helper to calculate the passenger load of flights departing soon.
        """
        # Look for flights departing between 30 and 120 minutes from the given timestamp
        window_start = timestamp + timedelta(minutes=30)
        window_end = timestamp + timedelta(minutes=120)
        
        upcoming = self.flight_schedule[
            (self.flight_schedule['Departure_Time'] >= window_start) & 
            (self.flight_schedule['Departure_Time'] <= window_end)
        ]
        
        return upcoming['Expected_Passengers'].sum()

if __name__ == "__main__":
    # Verification
    merger = QueueDataMerger()
    now = datetime.now()
    
    # Simulate 5 real-time readings
    for i in range(5):
        ts = now + timedelta(minutes=i*5)
        count = 50 + (i * 10) # Simulating increasing queue
        processed_df = merger.merge_realtime_data(ts, count)
    
    print("Processed Time-Series with Flight Impact Features:")
    print(processed_df.tail())
