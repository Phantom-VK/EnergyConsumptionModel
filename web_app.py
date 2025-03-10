from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib

matplotlib.use('Agg')  # For running without a display
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from suggestion_generation import generate_suggestions  # Import the function

app = Flask(__name__)

# Load the model
model = None
try:
    with open('energy_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError as e:
    print("Model file not found. Please run the ML model script first.")


# Function to load datasets
def load_data():
    devices_df = pd.read_csv('devices.csv') if os.path.exists('devices.csv') else None
    energy_df = pd.read_csv('energy_consumption.csv') if os.path.exists('energy_consumption.csv') else None
    timetable_df = pd.read_csv('timetable.csv') if os.path.exists('timetable.csv') else None
    event_df = pd.read_csv('events.csv') if os.path.exists('events.csv') else None
    suggestions_df = pd.read_csv('energy_saving_suggestions.csv') if os.path.exists(
        'energy_saving_suggestions.csv') else None
    return devices_df, energy_df, timetable_df, event_df, suggestions_df


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Suggestion Generation Page
@app.route('/suggestions', methods=['GET', 'POST'])
def suggestions():
    if request.method == 'POST':
        # Get the building ID from the form
        building_id = int(request.form['building_id'])

        # Load datasets
        devices_df, energy_df, timetable_df, event_df, _ = load_data()

        # Generate suggestions
        if devices_df is not None and energy_df is not None:
            generated_suggestions = generate_suggestions(building_id, devices_df, energy_df)
            return render_template('suggestions.html', suggestions=generated_suggestions, building_id=building_id)
        else:
            return "Data files not found. Please run the data generation script first."

    # Render the form for GET requests
    return render_template('suggestions_form.html')



# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if model is None:
        return "Model not found. Please run the ML model script first.", 500
    if request.method == 'POST':
        # Get form data
        building_id = int(request.form['building_id'])
        temperature = float(request.form['temperature'])
        is_weekend = 1 if request.form.get('is_weekend') else 0
        num_devices = int(request.form['num_devices'])
        avg_power = float(request.form['avg_power'])
        num_events = int(request.form['num_events'])

        # Make prediction if model is loaded
        if model is not None:
            features = np.array([[temperature, is_weekend, num_devices, avg_power, num_events]])
            prediction = model.predict(features)[0]

            # Generate future dates for forecasting
            start_date = datetime.now()
            dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]

            # Generate predictions for a week with slight variations
            forecasts = []
            for i, date in enumerate(dates):
                # Vary temperature slightly for future days
                temp_var = temperature + np.random.uniform(-2, 2)
                # Weekend adjustment
                weekend_adj = 1 if (start_date + timedelta(days=i)).weekday() >= 5 else 0

                features = np.array([[temp_var, weekend_adj, num_devices, avg_power, num_events]])
                pred = model.predict(features)[0]
                forecasts.append({
                    'date': date,
                    'prediction': round(pred, 2)
                })

            # Get suggestions
            _, _, _, _, suggestions_df = load_data()
            if suggestions_df is not None:
                building_suggestions = suggestions_df[suggestions_df['Building_ID'] == building_id]
                top_suggestions = building_suggestions.sort_values('Potential_Savings_kWh', ascending=False).head(3)
                top_suggestions = top_suggestions.to_dict('records')
            else:
                top_suggestions = []

            return render_template('prediction.html',
                                   building_id=building_id,
                                   prediction=round(prediction, 2),
                                   forecasts=forecasts,
                                   suggestions=top_suggestions)

    return render_template('predict_form.html')


# Data visualization page
@app.route('/visualize')
def visualize():
    devices_df, energy_df, timetable_df, event_df, _ = load_data()
    if energy_df is not None:
        # Create energy consumption visualization
        plt.figure(figsize=(12, 6))
        for building_id in energy_df['Building_ID'].unique():
            building_data = energy_df[energy_df['Building_ID'] == building_id]
            plt.plot(building_data['Date'], building_data['Total_Energy_kWh'],
                     label=f'Building {building_id}')

        plt.xlabel('Date')
        plt.ylabel('Energy Consumption (kWh)')
        plt.title('Energy Consumption by Building')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/energy_consumption.png')

        # Create temperature vs. energy visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(energy_df['Temperature_C'], energy_df['Total_Energy_kWh'],
                    alpha=0.6, c=energy_df['Building_ID'], cmap='viridis')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Energy Consumption (kWh)')
        plt.title('Temperature vs. Energy Consumption')
        plt.colorbar(label='Building ID')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('static/temp_vs_energy.png')

        # Create weekday vs. weekend comparison
        weekday_avg = energy_df[energy_df['Is_Weekend'] == 0].groupby('Building_ID')['Total_Energy_kWh'].mean()
        weekend_avg = energy_df[energy_df['Is_Weekend'] == 1].groupby('Building_ID')['Total_Energy_kWh'].mean()

        buildings = weekday_avg.index

        plt.figure(figsize=(10, 6))
        width = 0.35
        plt.bar(buildings - width / 2, weekday_avg, width, label='Weekday')
        plt.bar(buildings + width / 2, weekend_avg, width, label='Weekend')
        plt.xlabel('Building ID')
        plt.ylabel('Average Energy Consumption (kWh)')
        plt.title('Weekday vs. Weekend Energy Consumption')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('static/weekday_weekend.png')

        return render_template('visualize.html')
    else:
        return "Data files not found. Please run the data generation script first."


if __name__ == "__main__":
    app.run(debug=True)