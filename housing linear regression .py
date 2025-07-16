# house_price_regression.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Create the dataset
data = pd.DataFrame({
    'square_feet': [1500, 1800, 2400, 3000, 3500, 2100, 1600, 2800, 2200, 3200],
    'bedrooms': [3, 4, 3, 5, 4, 3, 2, 4, 3, 5],
    'bathrooms': [2, 2, 3, 4, 3, 2, 1, 3, 2, 4],
    'price': [300000, 360000, 450000, 600000, 650000, 400000, 290000, 500000, 420000, 630000]
})

print("\n📊 Dataset Preview:")
print(data.head())

# Step 2: Prepare features and target
X = data[['square_feet', 'bedrooms', 'bathrooms']]
y = data['price']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Evaluation:")
print("Mean Squared Error:", round(mse, 2))
print("R-squared Score:", round(r2, 4))

# Step 7: Predict new house price
new_house = pd.DataFrame({
    'square_feet': [2000],
    'bedrooms': [3],
    'bathrooms': [2]
})
predicted_price = model.predict(new_house)
print("\n🏠 Predicted price for 2000 sqft, 3BR, 2BA:", round(predicted_price[0], 2))

# Step 8: Visualization
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.tight_layout()
plt.show()
3
