import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import joblib
import logging

# Configure logging to display information during code execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the ChiPi datasets for conglomerates and non-conglomerates
congloms_df = pd.read_csv('processed_congloms_chipi.csv', low_memory=False)
non_congloms_df = pd.read_csv('processed_non-congloms_chipi.csv', low_memory=False)

# Add a label column to differentiate between conglomerates (1) and non-conglomerates (0)
congloms_df['label'] = 1
non_congloms_df['label'] = 0

# Combine both datasets into a single DataFrame for processing
combined_df = pd.concat([congloms_df, non_congloms_df], ignore_index=True)

# Define the continuous features and categorical features used as predictors, as well as the label
continuous_features = ['a', 'b', 'c', 'Cell Volume', 'Calc. Density', 'Alpha', 'Beta', 'Gamma',
                       'R-factor', 'Number of chiral center', 'S', 'R', 'M',
                       'Number of Carbon Chiral Atom', 'Number of Chiral Center having H',
                       'Number of chiral resd', 'Number of chiral families',
                       'Unique Chemical Units', 'Z Prime', 'Z Value']
categorical_feature = ['Space Gp. Number']
X = combined_df[continuous_features + categorical_feature].copy()
y = combined_df['label']

# Convert continuous features to numeric data types, handling any errors by converting them to NaN
for column in continuous_features:
    X[column] = pd.to_numeric(X[column], errors='coerce')

# Handle missing values and scale the data
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Apply the preprocessing steps to the predictor variables
X_preprocessed = preprocessor.fit_transform(X)

# Apply random undersampling to balance the dataset between conglomerates and non-conglomerates
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_preprocessed, y)

# Set the number of trees and initialize variables for tracking the best parameters
num_trees = 245
best_depth = 0
best_learning_rate = 0.0
best_accuracy = 0.0
accuracy_results = []

# Step 1: Optimize interaction depth (max_depth)
for depth in range(1, 21):
    xgb_model = XGBClassifier(n_estimators=num_trees, max_depth=depth, random_state=42, eval_metric='logloss')

    # Perform cross-validation to evaluate the model's accuracy for each depth
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, X_res, y_res, cv=cv, scoring='accuracy')
    mean_cv_score = cv_scores.mean()

    logging.info(f"Interaction Depth: {depth}, Mean cross-validation score: {mean_cv_score}")
    accuracy_results.append((depth, mean_cv_score))

    # Update the best depth if the current depth results in a higher accuracy
    if mean_cv_score > best_accuracy:
        best_accuracy = mean_cv_score
        best_depth = depth

# Plot interaction depth optimization results
depth_df = pd.DataFrame(accuracy_results, columns=['interaction_depth', 'mean_accuracy'])
plt.figure(figsize=(10, 6))
plt.plot(depth_df['interaction_depth'], depth_df['mean_accuracy'], marker='o')
plt.title('Accuracy vs. Interaction Depth in XGBoost')
plt.xlabel('Interaction Depth')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.grid(True)
plt.axvline(x=best_depth, color='r', linestyle='--', label=f'Optimal Depth: {best_depth}')
plt.legend()
plt.show()

logging.info(f"Optimal interaction depth: {best_depth}")

# Step 2: Optimize learning rate
# Reset best_accuracy and test different values for learning_rate to find the optimal rate
best_accuracy = 0.0
learning_rate_results = []
learning_rates = np.linspace(0.0025, 0.25, 50)

for learning_rate in learning_rates:
    xgb_model = XGBClassifier(n_estimators=num_trees, max_depth=best_depth, learning_rate=learning_rate, random_state=42, eval_metric='logloss')

    # Perform cross-validation to evaluate the model's accuracy for each learning rate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, X_res, y_res, cv=cv, scoring='accuracy')
    mean_cv_score = cv_scores.mean()

    logging.info(f"Learning Rate: {learning_rate}, Mean cross-validation score: {mean_cv_score}")
    learning_rate_results.append((learning_rate, mean_cv_score))

    # Update the best learning rate if the current rate results in a higher accuracy
    if mean_cv_score > best_accuracy:
        best_accuracy = mean_cv_score
        best_learning_rate = learning_rate

# Plot learning rate optimization results
learning_rate_df = pd.DataFrame(learning_rate_results, columns=['learning_rate', 'mean_accuracy'])
plt.figure(figsize=(10, 6))
plt.plot(learning_rate_df['learning_rate'], learning_rate_df['mean_accuracy'], marker='o')
plt.title('Accuracy vs. Learning Rate in XGBoost')
plt.xlabel('Learning Rate')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.grid(True)
plt.axvline(x=best_learning_rate, color='r', linestyle='--', label=f'Optimal Learning Rate: {best_learning_rate}')
plt.legend()
plt.show()

logging.info(f"Optimal learning rate: {best_learning_rate}")

# Step 3: Train the final XGBoost model with the optimal parameters
final_xgb_model = XGBClassifier(n_estimators=num_trees, max_depth=best_depth, learning_rate=best_learning_rate, random_state=42, eval_metric='logloss')
final_xgb_model.fit(X_res, y_res)

# Split the data into a training set and a testing set for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
final_xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = final_xgb_model.predict(X_test)

# Evaluate the model using classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the trained model
model_output_path = 'xgboost_best_model_crystal.pkl'
joblib.dump(final_xgb_model, model_output_path)
logging.info(f"Model saved to {model_output_path}")

# Calculate and display the feature importances for the model
feature_importances = final_xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': continuous_features + categorical_feature,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature importances:")
print(importance_df.to_string(index=False))  # Display all feature importances

# Save feature importances to a CSV file
importance_df.to_csv('feature_importances_xgboost_crystal.csv', index=False)
