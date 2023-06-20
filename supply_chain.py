import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the supply chain data
supply_chain_df = pd.read_csv("supply_chain.csv")

# Create the features and target variables
features = supply_chain_df[["demand", "lead_time"]]
target = supply_chain_df["inventory_cost"]

# Create a linear regression model
model = LinearRegression()
model.fit(features, target)

# Predict the inventory cost for a new demand and lead time
new_demand = 1000
new_lead_time = 10
predicted_inventory_cost = model.predict([[new_demand, new_lead_time]])

print("The predicted inventory cost is: " + str(predicted_inventory_cost))

