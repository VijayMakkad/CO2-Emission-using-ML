import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
import pickle

# Load the dataset
df = pd.read_csv("FuelConsumption.csv")

# Select relevant features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Training Data and Predictor Variable
# Use train-test-split to create a validation set
x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and fit the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Save model to disk
with open('model.pkl', 'wb') as model_file:
    pickle.dump(regressor, model_file)

# Plotting
predictions = regressor.predict(x_test)

plt.scatter(x_test['FUELCONSUMPTION_COMB'], y_test, color='blue', label='Actual Data')
plt.scatter(x_test['FUELCONSUMPTION_COMB'], predictions, color='red', label='Predictions')
plt.xlabel('Fuel Consumption (Combined)')
plt.ylabel('CO2 Emission')
plt.legend()
plt.title('Actual vs Predicted CO2 Emission')
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# R-squared
r2 = regressor.score(x_test, y_test)
print(f'R-squared: {r2}')

