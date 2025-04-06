from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X, y):
    """Train a Logistic Regression model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'logistic_regression_model.pkl')
    
    return model, X_test, y_test