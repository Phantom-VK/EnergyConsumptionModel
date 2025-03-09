import pandas as pd
import numpy as np
from datetime import datetime, timedelta

devices_data = {
    'Device_ID': [1, 2, 3, 4, 5],
    'Device_Name': ['Light', 'Fan', 'AC', 'Projector', 'Computer'],
    'Power_Rating_kW': [0.04, 0.075, 1.5, 0.3, 0.2],
    'Location': ['BuildingA', 'BuildingA', 'BuildingB', 'BuildingC', 'BuildingB']
}
devices_df = pd.DataFrame(devices_data)

timetable_data = {
    'Class_ID': [101, 102, 103, 104, 105],
    'Room': ['RoomA', 'RoomB', 'RoomA', 'RoomC', 'RoomB'],
    'Start_Time': ['09:00', '11:00', '14:00', '10:00', '15:00'],
    'End_Time': ['10:00', '12:00', '15:00', '12:00', '17:00'],
    'Devices_Used': ['Light,Fan', 'Light,Projector', 'Light,AC,Fan', 'Light,Projector,Computer', 'Light,Computer']
}
timetable_df = pd.DataFrame(timetable_data)

# Energy Consumption Dataset
# 30 days of data for 3 buildings
start_date = datetime(2023, 8, 1)
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
building_ids = [1, 2, 3]
energy_data = []

for date in dates:
    for building_id in building_ids:
        # Adding some randomness to make data more realistic
        base_consumption = 500 if building_id == 1 else (700 if building_id == 2 else 600)
        # Weekend factor (lower consumption on weekends)
        dt = datetime.strptime(date, '%Y-%m-%d')
        weekend_factor = 0.7 if dt.weekday() >= 5 else 1.0
        # Temperature factor (higher consumption on hotter days)
        temp = np.random.randint(25, 35)
        temp_factor = 1.0 + (temp - 25) * 0.02  # 2% increase per degree above 25°C

        total_energy = base_consumption * weekend_factor * temp_factor * np.random.uniform(0.9, 1.1)

        energy_data.append({
            'Building_ID': building_id,
            'Date': date,
            'Total_Energy_kWh': round(total_energy, 2),
            'Temperature_C': temp,
            'Is_Weekend': 1 if dt.weekday() >= 5 else 0
        })

energy_df = pd.DataFrame(energy_data)

# Create Event Dataset
event_data = {
    'Event_ID': [201, 202, 203, 204],
    'Event_Name': ['Seminar', 'Workshop', 'Conference', 'Meeting'],
    'Location': ['BuildingA', 'BuildingB', 'BuildingC', 'BuildingA'],
    'Start_Time': ['14:00', '18:00', '10:00', '16:00'],
    'End_Time': ['16:00', '20:00', '16:00', '17:00'],
    'Devices_Used': ['Light,AC,Projector', 'Light,Projector', 'Light,AC,Projector,Computer', 'Light,Computer']
}
event_df = pd.DataFrame(event_data)

# Combined Dataset for ML Model
combined_data = []

for _, row in energy_df.iterrows():
    date = row['Date']
    building_id = row['Building_ID']
    temp = row['Temperature_C']
    is_weekend = row['Is_Weekend']

    # number of classes in this building for this day
    building_name = f'Building{["A", "B", "C"][building_id - 1]}'

    # devices used in this building
    devices_in_building = devices_df[devices_df['Location'] == building_name]
    num_devices = len(devices_in_building)

    # Calculate average power rating
    avg_power = devices_in_building['Power_Rating_kW'].mean() if num_devices > 0 else 0

    # Count events in this building for this day
    events_in_building = event_df[event_df['Location'] == building_name]
    num_events = len(events_in_building)

    combined_data.append({
        'Date': date,
        'Building_ID': building_id,
        'Temperature_C': temp,
        'Is_Weekend': is_weekend,
        'Num_Devices': num_devices,
        'Avg_Power_Rating': avg_power,
        'Num_Events': num_events,
        'Total_Energy_kWh': row['Total_Energy_kWh']
    })

combined_df = pd.DataFrame(combined_data)

# SavING datasets
devices_df.to_csv('devices.csv', index=False)
timetable_df.to_csv('timetable.csv', index=False)
energy_df.to_csv('energy_consumption.csv', index=False)
event_df.to_csv('events.csv', index=False)
combined_df.to_csv('combined_data.csv', index=False)

print("Sample datasets created and saved to CSV files.")
print("Head of combined dataset:")
print(combined_df.head())