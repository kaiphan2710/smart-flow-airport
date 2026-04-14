import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
from datetime import datetime

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.forecasting.predictor import ThroughputPredictor
from src.optimization.router import DynamicRouter

app = FastAPI(
    title="SmartFlow Airport API",
    description="Operations and Routing API for Smart Airport Flow Management",
    version="1.0.0"
)

# Initialize Core Modules
predictor = ThroughputPredictor()
router = DynamicRouter()

# Request/Response Models
class PassengerRouteRequest(BaseModel):
    passenger_features: Dict[str, Any]
    airline: str
    lane_states: Dict[str, int]

class SystemStatusResponse(BaseModel):
    current_queues: Dict[str, int]
    forecast_30m: Dict[str, int]
    capacity_recommendation: Dict[str, Any]

@app.get("/")
def read_root():
    return {"status": "Online", "module": "SmartFlow-Airport-Core"}

@app.post("/route_passenger")
def route_passenger(request: PassengerRouteRequest):
    """
    Endpoint for individual passenger routing using Neuro-symbolic logic.
    Accepts vision-derived features and returns the optimal lane.
    """
    recommendation = router.route_individual_passenger(
        passenger_features=request.passenger_features,
        airline=request.airline,
        lane_states=request.lane_states
    )
    return recommendation

@app.get("/system_status", response_model=SystemStatusResponse)
def get_system_status():
    """
    Summarizes current airport operational state:
    1. Current Queue Distribution
    2. 30-minute Forecast
    3. Counter Capacity Recommendations
    """
    # Mock current state for API demonstration
    current_queues = {"Economy": 28, "Priority": 3, "Automated": 8}
    total_current_count = sum(current_queues.values())

    # Mock historical data for predictor (needed for forecast generation)
    mock_history = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now(), periods=10, freq='5min'),
        'passenger_count': [30, 32, 35, 33, 36, 38, 40, 39, 41, total_current_count],
        'upcoming_flight_load': [450] * 10
    })
    
    # Update predictor and get forecast
    predictor.train_or_update(mock_history)
    forecast = predictor.predict_next_30_mins(mock_history)
    
    # Get operational recommendation based on forecast
    capacity_rec = router.evaluate_checkin_capacity(
        predicted_queues=forecast, 
        open_counters=2
    )
    
    return {
        "current_queues": current_queues,
        "forecast_30m": forecast,
        "capacity_recommendation": capacity_rec
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
