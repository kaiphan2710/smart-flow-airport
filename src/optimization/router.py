import numpy as np
from ortools.linear_solver import pywraplp

class DynamicRouter:
    def __init__(self):
        """
        Initializes the Dynamic Router for SmartFlow Airport.
        This module integrates Operations Research (OR) and Neuro-symbolic AI 
        to optimize airport throughput and individual passenger experience.
        """
        pass

    def evaluate_checkin_capacity(self, predicted_queues, open_counters, max_capacity_per_counter=20):
        """
        Operations Research Optimization:
        Determines the optimal number of counters to open based on predicted load.
        Uses a heuristic approach to balance labor cost vs. passenger wait time.
        
        :param predicted_queues: Dict of predicted counts (e.g., {'+15m': 45}) or average count.
        :param open_counters: Integer, currently active counters.
        :param max_capacity_per_counter: Max passengers one counter can handle efficiently.
        :return: Action recommendation (Dictionary).
        """
        # Calculate mean predicted load from Phase 2 forecasting output
        if isinstance(predicted_queues, dict):
            avg_load = sum(predicted_queues.values()) / len(predicted_queues)
        else:
            avg_load = predicted_queues
            
        # Target: Keep queue per counter <= max_capacity_per_counter
        ideal_counters = int(np.ceil(avg_load / max_capacity_per_counter))
        
        # Constraint: Ensure at least one counter is always open
        ideal_counters = max(1, ideal_counters)

        if ideal_counters > open_counters:
            return {
                "action": "OPEN_COUNTER",
                "quantity": ideal_counters - open_counters,
                "reason": f"Predicted average load of {avg_load:.1f} exceeds current capacity ({open_counters * max_capacity_per_counter})."
            }
        elif ideal_counters < open_counters:
            return {
                "action": "CLOSE_COUNTER",
                "quantity": open_counters - ideal_counters,
                "reason": f"Predicted load of {avg_load:.1f} allows for labor optimization by closing {open_counters - ideal_counters} counter(s)."
            }
        
        return {"action": "MAINTAIN", "quantity": 0, "reason": "Current capacity is optimal for predicted demand."}

    def route_individual_passenger(self, passenger_features, airline, lane_states):
        """
        Neuro-symbolic Routing Engine:
        Combines Neural Network outputs (Vision Pipeline) with Symbolic Logic (Business Rules).
        
        Logic Explanation:
        The 'Neural' part provides the attributes (e.g., is there a suitcase?), 
        and the 'Symbolic' part applies hard-coded airport policies to ensure 
        predictable and safe routing decisions.
        
        :param passenger_features: Dict (ML output: {'has_heavy_luggage': bool, 'is_backpack_only': bool})
        :param airline: String (e.g., 'Qantas')
        :param lane_states: Dict (Current lengths: {'Economy': 25, 'Priority': 2, 'Automated': 5})
        :return: Routing recommendation (Dictionary).
        """
        has_heavy = passenger_features.get('has_heavy_luggage', False)
        is_light = passenger_features.get('is_backpack_only', False) or passenger_features.get('no_luggage', False)
        
        recommendation = "Economy" # Default symbolic state
        reason = "Assigned to standard Economy lane."

        # Rule A: IF (Heavy Luggage) AND (Airline == "Qantas") AND (Economy Queue > 20) -> Route to "Priority".
        # This prevents bottlenecking at standard counters for high-friction passengers.
        if has_heavy and airline == "Qantas" and lane_states.get("Economy", 0) > 20:
            recommendation = "Priority"
            reason = "High-friction passenger (heavy luggage) redirected to Priority to maintain Economy throughput."
        
        # Rule B: IF (No Luggage / Backpack only) -> Route to "Automated Kiosk".
        # Fast-tracking low-friction passengers to high-efficiency automated systems.
        elif is_light and lane_states.get("Automated", 0) < 15:
            recommendation = "Automated"
            reason = "Low-friction passenger (light luggage) routed to high-speed Automated Kiosk."

        # Rule C: Load Balancing Override
        # If the recommended lane is critically full, find the shortest alternative.
        if lane_states.get(recommendation, 0) > 40:
            best_alt = min(lane_states, key=lane_states.get)
            reason = f"Override: Original recommendation ({recommendation}) is saturated. Routing to {best_alt} for load balancing."
            recommendation = best_alt

        return {
            "target_lane": recommendation,
            "reason": reason,
            "lane_status_at_routing": lane_states
        }

if __name__ == "__main__":
    # Local verification
    router = DynamicRouter()
    
    # Test 1: Operational Capacity
    print("--- Capacity Optimization Test ---")
    predictions = {"+5m": 45, "+15m": 55, "+30m": 60}
    capacity_action = router.evaluate_checkin_capacity(predictions, open_counters=2)
    print(capacity_action)

    # Test 2: Neuro-symbolic Individual Routing
    print("\n--- Individual Routing Test (Rule A) ---")
    p_features = {"has_heavy_luggage": True}
    lanes = {"Economy": 25, "Priority": 2, "Automated": 5}
    route = router.route_individual_passenger(p_features, "Qantas", lanes)
    print(route)

    print("\n--- Individual Routing Test (Rule B) ---")
    p_features = {"is_backpack_only": True}
    route = router.route_individual_passenger(p_features, "Jetstar", lanes)
    print(route)
