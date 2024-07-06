# Python_Project
This github repository contains practice assignments on Python ! 
Project Title: Predicting House Prices
Project Overview:
The goal of this project is to predict house prices based on various features such as square footage, number of bedrooms and bathrooms, location, and other relevant factors. This is a classic regression problem in data science.

**Project Workflow:**
Exploratory Data Analysis (EDA): Understanding the dataset, identifying patterns, and visualizing relationships between variables.

Data Preprocessing: Cleaning the data, handling missing values, scaling numerical features, and encoding categorical variables.

Feature Engineering: Creating new features, transforming variables to improve model performance.

Model Building: Training and evaluating regression models (e.g., Linear Regression, Random Forest, Gradient Boosting) to predict house prices.

Model Evaluation: Assessing model performance using metrics like RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R-squared.

Deployment and Reporting: Saving the best-performing model, preparing reports and presentations summarizing findings, insights, and next steps.

Additional Considerations:
Version Control: Using Git for version control to track changes, collaborate with team members, and maintain project history.

Documentation: Writing clear and concise documentation (in notebooks, README, and reports) to facilitate understanding and reproducibility.

This structure provides a comprehensive framework for organizing and presenting your Master's in Data Science project focused on predicting house prices. Adjust the specifics according to your dataset and project requirements.




id,area,bedrooms,bathrooms,location,price
1,2100,3,2,"City A",410000
2,1600,2,2,"City B",330000
3,2600,4,3,"City A",550000
4,1800,3,2,"City C",410000
5,2200,3,3,"City B",480000
6,3000,4,3,"City C",700000
7,2500,4,2,"City A",620000
8,2000,3,2,"City B",410000
9,1900,3,2,"City C",390000
10,2800,4,3,"City A",600000
11,2400,3,2,"City B",420000
12,1800,2,2,"City C",330000
13,2300,3,3,"City A",500000
14,3100,4,3,"City C",720000
15,2600,4,2,"City B",590000
16,2000,3,2,"City A",410000
17,1700,2,2,"City C",340000
18,2500,3,3,"City B",480000
19,2900,4,3,"City A",650000
20,3000,4,3,"City C",710000
21,2200,3,2,"City B",430000
22,1900,3,2,"City C",380000
23,2400,3,3,"City A",510000
24,2600,4,3,"City B",600000
25,2800,4,3,"City A",620000
26,2000,3,2,"City C",400000
27,1700,2,2,"City B",350000
28,2500,3,3,"City A",470000
29,2900,4,3,"City C",680000
30,3000,4,3,"City A",690000
31,2100,3,2,"City B",440000
32,1600,2,2,"City C",320000
33,2600,4,3,"City A",540000
34,1800,3,2,"City B",410000
35,2200,3,3,"City C",490000
36,3000,4,3,"City A",710000
37,2500,4,2,"City B",630000
38,2000,3,2,"City A",420000
39,1900,3,2,"City C",370000
40,2800,4,3,"City B",610000
41,2400,3,2,"City A",430000
42,1800,2,2,"City C",340000
43,2300,3,3,"City B",500000
44,3100,4,3,"City A",720000
45,2600,4,2,"City C",590000
46,2000,3,2,"City B",400000
47,1700,2,2,"City A",360000
48,2500,3,3,"City C",470000
49,2900,4,3,"City B",670000
50,3000,4,3,"City A",680000
51,2100,3,2,"City C",450000
52,1600,2,2,"City A",310000
53,2600,4,3,"City B",550000
54,1800,3,2,"City C",390000
55,2200,3,3,"City A",480000
56,3000,4,3,"City B",700000
57,2500,4,2,"City C",620000
58,2000,3,2,"City A",410000
59,1900,3,2,"City B",380000
60,2800,4,3,"City C",720000
61,2400,3,2,"City A",440000
62,1800,2,2,"City B",350000
63,2300,3,3,"City C",510000
64,3100,4,3,"City A",730000
65,2600,4,2,"City B",600000
66,2000,3,2,"City A",420000
67,1700,2,2,"City C",360000
68,2500,3,3,"City B",480000
69,2900,4,3,"City A",670000
70,3000,4,3,"City C",710000
71,2100,3,2,"City B",450000
72,1600,2,2,"City C",320000
73,2600,4,3,"City A",540000
74,1800,3,2,"City B",400000
75,2200,3,3,"City C",490000
76,3000,4,3,"City A",690000
77,2500,4,2,"City B",630000
78,2000,3,2,"City A",430000
79,1900,3,2,"City C",370000
80,2800,4,3,"City B",610000
81,2400,3,2,"City A",440000
82,1800,2,2,"City C",340000
83,2300,3,3,"City B",500000
84,3100,4,3,"City A",720000
85,2600,4,2,"City C",590000
86,2000,3,2,"City B",410000
87,1700,2,2,"City A",360000
88,2500,3,3,"City C",470000
89,2900,4,3,"City B",670000
90,3000,4,3,"City A",680000
91,2100,3,2,"City C",450000
92,1600,2,2,"City A",310000
93,2600,4,3,"City B",550000
94,1800,3,2,"City C",390000
95,2200,3,3,"City A",480000
96,3000,4,3,"City B",700000
97,2500,4,2,"City C",620000
98,2000,3,2,"City A",410000
99,1900,3,2,"City B",380000
100,2800,4,3,"City C",720000







import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Load the dataset
df = pd.read_csv('house_prices.csv')

# Display the first few rows and basic information
print("First few rows of the dataset:")
print(df.head())

print("\nBasic information about the dataset:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Step 2: Data Preprocessing
# Handle missing values (if any)
df.dropna(inplace=True)

# Encode categorical variables (e.g., location)
df = pd.get_dummies(df, columns=['location'])

# Step 3: Feature Engineering
# Example of adding a new feature (e.g., total rooms)
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Step 4: Model Building and Evaluation
# Separate features and target variable
X = df.drop(columns=['id', 'price'])
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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





Notes:
Updated Dataset: The house_prices.csv now contains 100 sample records, which is smaller and easier to manage for testing purposes.
Combined Workflow: This script covers all steps from data loading to model evaluation in one file (house_price_prediction.py).
Model Saving: The trained model (house_price_model.pkl) is saved in the current working directory.
To run this script:

Save the Python script as house_price_prediction.py.
Save the house_prices.csv file with 100 sample entries in the same directory as the script.
Execute the script using Python (python house_price_prediction.py).
This script will load the data, perform preprocessing and feature engineering, train a Linear Regression model, evaluate it using RMSE, and save the trained model for future use. Adjust and expand the code based on your specific project requirements and dataset characteristics.

