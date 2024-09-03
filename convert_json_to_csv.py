import pandas as pd

# Load JSON data
data = pd.read_json('shipping_data.json')

# Convert to CSV
data.to_csv('shipping_data.csv', index=False)
