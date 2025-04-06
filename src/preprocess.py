import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the dataset."""
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    
    return X_scaled, y