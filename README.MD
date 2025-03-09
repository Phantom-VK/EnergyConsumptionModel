# Energy Consumption Prediction and Optimization System

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-red)

A machine learning-based system for predicting and optimizing energy consumption in educational buildings. This project uses historical energy data, device information, and timetables to predict future energy consumption and provide actionable suggestions to minimize energy usage.

---

## Features

- **Energy Consumption Prediction**:
  - Predicts daily, weekly, and monthly energy consumption using a Linear Regression model.
  - Considers factors like temperature, number of devices, and weekend usage.

- **Energy-Saving Suggestions**:
  - Provides rule-based and data-driven suggestions to reduce energy consumption.
  - Uses KMeans clustering to identify unusual energy usage patterns.

- **Web Application**:
  - A Flask-based web app for easy interaction.
  - Users can input building details, view predictions, and explore visualizations.

- **Visualizations**:
  - Displays energy consumption trends, temperature vs. energy plots, and weekday vs. weekend comparisons.

---

## Project Structure
energy-consumption-prediction/
├── templates/ # HTML templates for the Flask app
│ ├── index.html # Home page
│ ├── predict_form.html # Form for energy prediction
│ ├── prediction.html # Prediction results page
│ └── visualize.html # Visualization page
├── static/ # Static files (images, CSS, etc.)
│ ├── energy_consumption.png # Energy consumption visualization
│ ├── temp_vs_energy.png # Temperature vs. energy plot
│ └── weekday_weekend.png # Weekday vs. weekend comparison
├── datasetCreation.py # Script to generate sample datasets
├── model.py # Script to train the ML model
├── suggestion_generation.py # Script to generate energy-saving suggestions
├── web_app.py # Flask web application
├── README.md # Project documentation
├── requirements.txt # Python dependencies
└── energy_model.pkl # Trained ML model



---

## Installation

### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)

### Steps
Install Dependencies:
```pip install -r requirements.txt```
Generate Sample Data(You can create ur own):
```python datasetCreation.py```
Train the Machine Learning Model:
```python model.py```
Generate Energy-Saving Suggestions:
```python suggestion_generation.py```
Run the Flask Web App:
```python web_app.py```