import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset from the CSV file
data = pd.read_csv('C:/Users/SILVIA NGUGI/PYTH/model/WorldsBestRestaurants.csv')

# Check for missing values
print(data.isnull().sum())

# One-hot encoding for categorical variables
data_encoded = pd.get_dummies(data, columns=['restaurant', 'location', 'country'], drop_first=True)

# Display the first few rows of the encoded dataset
print(data_encoded.head())

# Define features (X) and target variable (y)
X = data_encoded.drop(columns=['rank'])
y = data_encoded['rank']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model instance
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Output predictions for verification (optional)
print(predictions)