from sklearn.cluster import KMeans

def generate_suggestions(building_id, devices_df, energy_df):
    """
    Generate energy-saving suggestions based on device usage and patterns.
    """
    building_name = f'Building{chr(64 + building_id)}'  # Convert 1->A, 2->B, etc.

    building_devices = devices_df[devices_df['Location'] == building_name]

    suggestions = []

    # 1. Identify high power consumption devices
    if not building_devices.empty:
        high_power_devices = building_devices[building_devices['Power_Rating_kW'] > 0.5]
        for _, device in high_power_devices.iterrows():
            saving_kwh = device['Power_Rating_kW'] * 1  # Assuming 1 hour reduction
            suggestions.append({
                'Building_ID': building_id,
                'Suggestion': f"Reduce {device['Device_Name']} usage by 1 hour per day",
                'Potential_Savings_kWh': round(saving_kwh, 2),
                'Priority': 'High' if saving_kwh > 1 else 'Medium'
            })

    # 2. Check weekend usage patterns
    weekend_energy = energy_df[(energy_df['Building_ID'] == building_id) & (energy_df['Is_Weekend'] == 1)]
    weekday_energy = energy_df[(energy_df['Building_ID'] == building_id) & (energy_df['Is_Weekend'] == 0)]

    if not weekend_energy.empty and not weekday_energy.empty:
        weekend_avg = weekend_energy['Total_Energy_kWh'].mean()
        weekday_avg = weekday_energy['Total_Energy_kWh'].mean()

        if weekend_avg > weekday_avg * 0.5:
            potential_savings = round((weekend_avg - weekday_avg * 0.5) * 2, 2)
            suggestions.append({
                'Building_ID': building_id,
                'Suggestion': "Reduce weekend energy usage by turning off non-essential devices",
                'Potential_Savings_kWh': potential_savings,
                'Priority': 'High' if potential_savings > 50 else 'Medium'
            })

    # 3. Temperature-based suggestions for AC usage
    high_temp_days = energy_df[(energy_df['Building_ID'] == building_id) & (energy_df['Temperature_C'] > 30)]
    if not high_temp_days.empty:
        ac_devices = building_devices[building_devices['Device_Name'] == 'AC']
        if not ac_devices.empty:
            avg_ac_power = ac_devices['Power_Rating_kW'].mean()
            potential_savings = round(avg_ac_power * 0.5 * len(high_temp_days), 2)  # 0.5 hour reduction per hot day
            suggestions.append({
                'Building_ID': building_id,
                'Suggestion': "Optimize AC temperature settings on hot days (set to 24°C instead of lower)",
                'Potential_Savings_kWh': potential_savings,
                'Priority': 'Medium'
            })

    # 4. Clustering-based suggestions
    #  Clustering to identify unusual energy patterns
    energy_features = energy_df[['Total_Energy_kWh', 'Temperature_C', 'Is_Weekend']]
    kmeans = KMeans(n_clusters=3)
    energy_df['Cluster'] = kmeans.fit_predict(energy_features)

    # Generate suggestions based on clusters
    for cluster in energy_df['Cluster'].unique():
        cluster_data = energy_df[energy_df['Cluster'] == cluster]
        avg_energy = cluster_data['Total_Energy_kWh'].mean()
        if avg_energy > energy_df['Total_Energy_kWh'].mean():
            suggestions.append({
                'Building_ID': building_id,
                'Suggestion': f"High energy consumption detected in cluster {cluster}. Investigate usage patterns.",
                'Potential_Savings_kWh': round(avg_energy * 0.1, 2),  # Assume 10% savings
                'Priority': 'High'
            })

    # 5. General suggestions based on building type
    suggestions.append({
        'Building_ID': building_id,
        'Suggestion': "Install motion sensors for lighting in common areas",
        'Potential_Savings_kWh': 30,
        'Priority': 'Medium'
    })

    suggestions.append({
        'Building_ID': building_id,
        'Suggestion': "Implement a power-down policy for all devices after working hours",
        'Potential_Savings_kWh': 50,
        'Priority': 'High'
    })

    return suggestions