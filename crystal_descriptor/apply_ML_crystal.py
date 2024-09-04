import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
import joblib
import logging

# Configure logging to display information during code execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
model_path = 'xgboost_best_model_crystal.pkl'
best_rf_model = joblib.load(model_path)
logging.info(f"Model loaded from {model_path}")

# Define the continuous features and categorical features used as predictors, as well as the label
continuous_features = ['a', 'b', 'c', 'Cell Volume', 'Calc. Density', 'Alpha', 'Beta', 'Gamma',
                       'R-factor', 'Number of chiral center', 'S', 'R', 'M',
                       'Number of Carbon Chiral Atom', 'Number of Chiral Center having H',
                       'Number of chiral resd', 'Number of chiral families',
                       'Unique Chemical Units', 'Z Prime', 'Z Value']
categorical_feature = ['Space Gp. Number']
all_features = continuous_features + categorical_feature

# Load the new input data from a CSV file
input_file = 'processed_2021all_chipi.csv' 
new_data = pd.read_csv(input_file)

# Ensure the input data contains the 'Identifier' column along with the required features
if 'Identifier' not in new_data.columns:
    logging.error("The input data must contain an 'Identifier' column.")
    raise ValueError("The input data must contain an 'Identifier' column.")

# Ensure all required features are present
missing_features = [feature for feature in all_features if feature not in new_data.columns]
if missing_features:
    logging.error(f"The following required features are missing from the input data: {missing_features}")
    raise ValueError(f"The following required features are missing: {missing_features}")

# Select only the necessary features for prediction
X_new = new_data[all_features].copy()

# Convert continuous features to numeric data types, handling any errors by converting them to NaN
for column in continuous_features:
    X_new.loc[:, column] = pd.to_numeric(X_new[column], errors='coerce')

# Handle missing values and scale the data
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
    ('scaler', StandardScaler())
])

# Apply preprocessing to the new data
X_new_preprocessed = preprocessor.fit_transform(X_new)

# Predict the probabilities using the loaded model
predicted_proba = best_rf_model.predict_proba(X_new_preprocessed)[:, 1]

# Load the actual conglomerates and non-conglomerates data for validation
congloms_actual_df = pd.read_csv('processed_2021congloms_chipi.csv')
non_congloms_actual_df = pd.read_csv('processed_2021noncongloms_chipi.csv')

# Label the actual data
congloms_actual_df['Actual'] = 'Conglomerate'
non_congloms_actual_df['Actual'] = 'Non-Conglomerate'

# Combine the actual conglomerates and non-conglomerates data
actual_data = pd.concat([congloms_actual_df, non_congloms_actual_df], ignore_index=True)

# List of thresholds to evaluate
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Evaluate and print results for each threshold
for threshold in thresholds:
    print(f"\nThreshold: {threshold}")
    
    # Apply the threshold to make predictions
    predictions = (predicted_proba >= threshold).astype(int)
    
    # Create a dictionary to map predictions to their actual labels (1 for conglomerate, 0 for non-conglomerate)
    new_data['Prediction'] = predictions
    new_data['Prediction'] = new_data['Prediction'].map({0: 'Non-Conglomerate', 1: 'Conglomerate'})
    
    # Merge the predictions with the actual labels
    comparison_df = pd.merge(new_data[['Identifier', 'Prediction']], 
                             actual_data[['Identifier', 'Actual']], 
                             on='Identifier', how='inner')
    
    # Ensure correct mapping to numerical labels for evaluation
    y_true = comparison_df['Actual'].map({'Non-Conglomerate': 0, 'Conglomerate': 1})
    y_pred = comparison_df['Prediction'].map({'Non-Conglomerate': 0, 'Conglomerate': 1})
    
    # Evaluate the predictions
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


