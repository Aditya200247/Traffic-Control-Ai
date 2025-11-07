import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import holidays
import datetime

# --- 1. CORE MACHINE LEARNING MODEL SECTION ---

def generate_realistic_data(records=25000):
    """
    Generates an enriched, realistic traffic dataset for Indore, India.
    Aware of current time (Oct 2025) to simulate relevant patterns.
    """
    print("Generating realistic, time-aware sample traffic data for Indore...")
    
    # Key locations and road segments in Indore
    road_segments = [
        'AB_Road_VijayNagar', 'Ring_Road_Radisson', 'Palasia_Square', 
        'Sarwate_Bus_Stand', 'Bhawarkua_Square', 'Airport_Road'
    ]
    weather_conditions = ['Clear', 'Hazy', 'Light Rain', 'Cloudy']
    
    # Generate a time series starting from Jan 1, 2025, leading up to the current date
    start_date = '2025-01-01'
    datetimes = pd.to_datetime(np.arange(records) * 300, unit='s', origin=start_date)
    
    df = pd.DataFrame({
        'timestamp': datetimes,
        'road_segment_id': np.random.choice(road_segments, records),
        'base_vehicle_count': np.random.randint(20, 300, records),
        'weather': np.random.choice(weather_conditions, records, p=[0.6, 0.2, 0.05, 0.15])
    })

    # --- Feature Engineering & Realistic Pattern Simulation ---
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Use 'holidays' library for Indian holidays (Madhya Pradesh)
    in_holidays = holidays.country_holidays('IN', subdiv='MP')
    df['is_holiday'] = df['timestamp'].dt.date.astype('datetime64').isin(in_holidays).astype(int)

    # --- Simulate Congestion Patterns ---
    # 1. Peak Hours (8-11 AM, 6-9 PM)
    peak_hour_multiplier = np.where(df['hour'].isin([8, 9, 10, 18, 19, 20]), np.random.uniform(1.8, 2.8, records), 1)
    
    # 2. Weekend Traffic (higher in commercial areas like Vijay Nagar)
    weekend_multiplier = np.where((df['is_weekend'] == 1) & (df['road_segment_id'] == 'AB_Road_VijayNagar'), 1.5, 1)
    
    # 3. Holiday/Festival Rush (e.g., around Diwali/Dussehra in Oct/Nov)
    # The model is aware of being in October 2025
    festival_multiplier = np.where((df['timestamp'].dt.month.isin([10, 11])) | (df['is_holiday'] == 1), 1.6, 1)
    
    # Apply multipliers to create the final vehicle count
    df['final_vehicle_count'] = df['base_vehicle_count'] * peak_hour_multiplier * weekend_multiplier * festival_multiplier
    
    # Target variable: Average speed is inversely related to vehicle count
    df['average_speed_kmh'] = 60 - (df['final_vehicle_count'] / 15)
    
    # Weather impact
    df.loc[df['weather'] == 'Light Rain', 'average_speed_kmh'] *= 0.7
    
    # Final cleanup
    df['average_speed_kmh'] = df['average_speed_kmh'].clip(5, 70).round(2)
    df['final_vehicle_count'] = df['final_vehicle_count'].astype(int).clip(10)
    
    print(f"Data generated with {records} records up to {df['timestamp'].max()}.")
    return df

def train_and_save_model():
    """Trains a LightGBM Regressor model and saves the entire pipeline."""
    data = generate_realistic_data()
    
    # Preprocessing
    data = pd.get_dummies(data, columns=['road_segment_id', 'weather'])
    
    target = 'average_speed_kmh'
    features = [col for col in data.columns if col not in ['timestamp', 'base_vehicle_count', target]]
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=40)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained. Mean Absolute Error on test set: {mae:.2f} km/h")
    
    # Save model and columns
    pipeline = {'model': model, 'columns': X.columns}
    joblib.dump(pipeline, 'traffic_model_pipeline.pkl')
    print("Model pipeline saved to 'traffic_model_pipeline.pkl'")

# --- 2. BACKEND API SERVER (Flask) ---

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Simulate Indore's Road Network ---
# This defines how key locations are connected by our road segments
ROUTE_NETWORK = {
    'Vijay Nagar Square': {
        'Rajwada Palace': ['AB_Road_VijayNagar', 'Palasia_Square', 'Sarwate_Bus_Stand'],
        'Airport': ['AB_Road_VijayNagar', 'Ring_Road_Radisson', 'Airport_Road']
    },
    'Rajwada Palace': {
        'Vijay Nagar Square': ['Sarwate_Bus_Stand', 'Palasia_Square', 'AB_Road_VijayNagar'],
        'Airport': ['Sarwate_Bus_Stand', 'Bhawarkua_Square', 'Airport_Road']
    },
     'Airport': {
        'Rajwada Palace': ['Airport_Road', 'Bhawarkua_Square', 'Sarwate_Bus_Stand'],
        'Vijay Nagar Square': ['Airport_Road', 'Ring_Road_Radisson', 'AB_Road_VijayNagar']
    }
}
# Base distance for each segment in km
SEGMENT_DISTANCES = {
    'AB_Road_VijayNagar': 3.5, 'Ring_Road_Radisson': 4.0, 'Palasia_Square': 2.0,
    'Sarwate_Bus_Stand': 1.5, 'Bhawarkua_Square': 2.5, 'Airport_Road': 3.0
}


def get_prediction_for_segment(segment_id, model, columns):
    """Runs a single prediction for a given road segment."""
    now = datetime.datetime.now()
    # In a real app, this data would come from live sensors/APIs
    mock_live_data = {
        'timestamp': now,
        'road_segment_id': segment_id,
        'weather': 'Clear' 
    }
    
    df = pd.DataFrame([mock_live_data])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    in_holidays = holidays.country_holidays('IN', subdiv='MP')
    df['is_holiday'] = df['timestamp'].dt.date.astype('datetime64').isin(in_holidays).astype(int)
    df['final_vehicle_count'] = np.random.randint(100, 400) # Simulate live count

    df = pd.get_dummies(df).drop(columns=['timestamp'])
    df_aligned = df.reindex(columns=columns, fill_value=0)
    
    return model.predict(df_aligned)[0]


@app.route('/api/v1/route-predict', methods=['POST'])
def predict_route():
    """API endpoint to predict travel time for different route options."""
    try:
        data = request.get_json()
        start = data['from']
        end = data['to']

        if start not in ROUTE_NETWORK or end not in ROUTE_NETWORK[start]:
            return jsonify({'error': 'Route not found in our network simulation.'}), 404

        # --- Simulate 3 different route choices ---
        
        # 1. Fastest Route (Standard, most direct path)
        fastest_segments = ROUTE_NETWORK[start][end]
        fastest_time = 0
        fastest_dist = 0
        predicted_speeds = []
        for segment in fastest_segments:
            speed = get_prediction_for_segment(segment, model_pipeline['model'], model_pipeline['columns'])
            predicted_speeds.append(speed)
            dist = SEGMENT_DISTANCES[segment]
            fastest_dist += dist
            fastest_time += (dist / speed) * 60  # time in minutes

        # 2. Balanced Route (Slightly longer, avoids a busy segment)
        balanced_segments = fastest_segments[1:] # Simulate avoiding the first segment
        balanced_segments.append('Bhawarkua_Square') # Add a different one
        balanced_time = 0
        balanced_dist = 0
        for segment in balanced_segments:
            speed = get_prediction_for_segment(segment, model_pipeline['model'], model_pipeline['columns']) * 1.2 # Assume less congestion
            dist = SEGMENT_DISTANCES.get(segment, 2.5)
            balanced_dist += dist
            balanced_time += (dist / speed) * 60

        # 3. Eco Route (Similar to balanced, focuses on steady speed)
        eco_time = balanced_time * 1.1 
        eco_dist = balanced_dist * 1.05

        # Create the response payload that matches the frontend
        response = {
            "routes": [
                {'id': 'fastest', 'name': 'Fastest Route', 'icon': 'Zap', 'color': '#007AFF', 
                 'distance': f"{fastest_dist:.1f} km", 'time': f"{int(fastest_time)} min", 
                 'congestion': np.mean(predicted_speeds) / 50},
                {'id': 'balanced', 'name': 'Balanced Load', 'icon': 'Car', 'color': '#34C759', 
                 'distance': f"{balanced_dist:.1f} km", 'time': f"{int(balanced_time)} min", 'congestion': 0.5},
                {'id': 'eco', 'name': 'Eco Route', 'icon': 'Leaf', 'color': '#5856D6', 
                 'distance': f"{eco_dist:.1f} km", 'time': f"{int(eco_time)} min", 'congestion': 0.3},
            ]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # On start, check if model exists. If not, train it.
    try:
        model_pipeline = joblib.load('traffic_model_pipeline.pkl')
        print("Existing model pipeline loaded.")
    except FileNotFoundError:
        print("Model not found. Training a new model...")
        train_and_save_model()
        model_pipeline = joblib.load('traffic_model_pipeline.pkl')
    
    print("\n--- SmartMove Backend Ready ---")
    print("Starting Flask server...")
    print("API is available at http://127.0.0.1:5000/api/v1/route-predict")
    app.run(debug=True, port=5000)