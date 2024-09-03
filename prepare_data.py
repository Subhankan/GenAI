from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv('shipping_data.csv')
data.fillna({'Fuel Efficiency miles per gallon': data['Fuel Efficiency miles per gallon'].median()}, inplace=True)

data['Date of Shipment'] = pd.to_datetime(data['Date of Shipment'])
data['Month of Shipment'] = data['Date of Shipment'].dt.month
data['Weekday of Shipment'] = data['Date of Shipment'].dt.weekday
data.drop('Date of Shipment', axis=1, inplace=True)

categorical_features = ['Carrier', 'Origin', 'Destination', 'Shipment Type']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

X = data.drop('Total Shipment Cost', axis=1)
y = data['Total Shipment Cost']
X_preprocessed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Saving transformed data and preprocessor
pd.DataFrame(X_train, columns=feature_names).to_csv('X_train.csv', index=False)
pd.DataFrame(X_test, columns=feature_names).to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
dump(preprocessor, 'preprocessor.pkl')
