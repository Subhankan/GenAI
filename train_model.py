import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump

# Load the training data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# Initialize and train the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())  # Make sure y_train is correctly shaped

# Save the trained model
dump(model, 'shipping_cost_predictor.pkl')