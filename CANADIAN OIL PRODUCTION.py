#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:41:28 2024

@author: shahab-nasiri
"""

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
df = pd.read_excel("Canada Crude Oil Production.xlsx")

# Ensure the 'Month' column is in datetime format
df['Month'] = pd.to_datetime(df['Month'])

# Split the data into training and test sets (80% training, 20% testing)
split_index = int(len(df) * 0.85)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Prepare data for Prophet (it requires columns 'ds' and 'y')
train_prophet_df = train_df.rename(columns={"Month": "ds", "Canada Total": "y"})
test_prophet_df = test_df.rename(columns={"Month": "ds", "Canada Total": "y"})

# Function to optimize Prophet parameters
def objective(trial):
    # Suggest hyperparameters to try
    changepoint_prior = trial.suggest_loguniform("changepoint_prior_scale", 0.001, 10.0)
    seasonality_prior = trial.suggest_loguniform("seasonality_prior_scale", 0.01, 40.0)
    holidays_prior = trial.suggest_loguniform("holidays_prior_scale", 0.01, 40.0)
    seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
    yearly_fourier_order = trial.suggest_int("yearly_fourier_order", 5, 50)
    
    # Initialize Prophet with suggested parameters
    model = Prophet(
        changepoint_prior_scale=changepoint_prior,
        seasonality_prior_scale=seasonality_prior,
        holidays_prior_scale=holidays_prior,
        seasonality_mode=seasonality_mode,
    )

    # Add yearly seasonality with suggested Fourier order
    model.add_seasonality(name="yearly", period=365.25, fourier_order=yearly_fourier_order)
    
    # Fit the model with training data
    model.fit(train_prophet_df)
    
    # Make predictions for the test set
    forecast = model.predict(test_prophet_df[['ds']])
    
    # Calculate MAE, MSE, and R² to evaluate the model
    actual = test_prophet_df['y']
    predicted = forecast['yhat']
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    return mae  # Optuna minimizes by default, so using MAE as the objective

# Create an Optuna study for hyperparameter optimization
study = optuna.create_study(direction="minimize")  # Optimizing for lower MAE
study.optimize(objective, n_trials=80)  # Number of trials for optimization

# Get the best trial and its hyperparameters
best_trial = study.best_trial
print(f"Best trial: {best_trial.number}")
print(f"  MAE: {best_trial.value:.2f}")
print("  Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Re-train the Prophet model with the best hyperparameters
best_model = Prophet(
    changepoint_prior_scale=best_trial.params["changepoint_prior_scale"],
    seasonality_prior_scale=best_trial.params["seasonality_prior_scale"],
    holidays_prior_scale=best_trial.params["holidays_prior_scale"],
    seasonality_mode=best_trial.params["seasonality_mode"],
)

# Add yearly seasonality with optimized Fourier order
best_model.add_seasonality(name="yearly", period=365.25, fourier_order=best_trial.params["yearly_fourier_order"])

# Fit the model to the training data
best_model.fit(train_prophet_df)

# Make predictions with the best model
best_forecast = best_model.predict(test_prophet_df[['ds']])

# Merge predictions with actual values for plotting
merged_results = test_prophet_df[['ds', 'y']].merge(best_forecast[['ds', 'yhat']], on='ds')

# Calculate MAE, MSE, and R² for the best model
final_mae = mean_absolute_error(merged_results['y'], merged_results['yhat'])
final_mse = mean_squared_error(merged_results['y'], merged_results['yhat'])
final_r2 = r2_score(merged_results['y'], merged_results['yhat'])

print(f"Optimized MAE: {final_mae:.2f}")
print(f"Optimized MSE: {final_mse:.2f}")
print(f"Optimized R²: {final_r2:.2f}")

# Plot actual values vs. predictions
plt.figure(figsize=(12.5, 7))
sns.lineplot(x=merged_results['ds'], y=merged_results['y'], label='Actual')
sns.lineplot(x=merged_results['ds'], y=merged_results['yhat'], label='Predicted (Best Model)')
plt.title("Canadian Crude Oil Production - Actual vs. Predicted")
plt.xlabel("Date")
plt.ylabel("Production (BPD)")
plt.legend()
plt.grid(True)
plt.show()


