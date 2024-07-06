import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Load the dataset
file = r'C:\Users\ASUS\Downloads\house_prices.csv'
df = pd.read_csv(file)

# Display the first few rows and basic information
print("First few rows of the dataset:")
print(df.head())

print("\nBasic information about the dataset:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Step 2: Data Preprocessing
# Drop unnecessary columns if any
df.drop(columns=['Unnamed: 6', 'Unnamed: 7'], inplace=True)

# Encode categorical variables (if any)
df = pd.get_dummies(df, columns=['location'])

# Step 3: Feature Engineering (if any)
# Example of adding a new feature (total_rooms)
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Step 4: Model Building and Evaluation
# Separate features and target variable
X = df.drop(columns=['id', 'price'])
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Check the shape of X and y after splitting
print(f"\nShape of X_train: {X_train.shape}, Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}, Shape of y_test: {y_test.shape}")

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"\nRoot Mean Squared Error: {rmse}")

# Save the trained model
joblib.dump(model, 'house_price_model.pkl')
print("\nTrained model saved as 'house_price_model.pkl'")
