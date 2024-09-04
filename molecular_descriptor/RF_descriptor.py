import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib
import logging

# Configure logging to display information during code execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the datasets containing descriptors for conglomerates and non-conglomerates
congloms_descriptors_df = pd.read_csv('congloms_descriptors_cut4.csv')
non_congloms_descriptors_df = pd.read_csv('non_congloms_descriptors_cut4.csv')

# Add a label column to differentiate between conglomerates (1) and non-conglomerates (0)
congloms_descriptors_df['label'] = 1
non_congloms_descriptors_df['label'] = 0

# Combine both datasets into a single DataFrame for processing
combined_df = pd.concat([congloms_descriptors_df, non_congloms_descriptors_df], ignore_index=True)
# Separate the features (X) from the label (y)
X = combined_df.drop('label', axis=1)
y = combined_df['label']

# Handle any infinite values and NaN values by replacing them with the mean of the column
X.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Standardize the data to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Apply random undersampling to balance the dataset
undersampler = RandomUnderSampler(random_state=42)
X_res, y_res = undersampler.fit_resample(X, y)

# Optimize the number of variables available at each split
max_features_options = range(1, 51)  # 1 to p (number of features)
accuracy_results = []

# Test different values for the max_features parameter to find the optimal setting
for max_features in max_features_options:
    # Initialize the RandomForestClassifier with 250 trees and current max_features
    rf_model = RandomForestClassifier(n_estimators=250, max_features=max_features, random_state=42, class_weight='balanced')

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_res, y_res, cv=cv, scoring='accuracy')
    mean_cv_score = cv_scores.mean() # Compute the mean accuracy across all folds

    # Log the mean cross-validation score for the current max_features value
    logging.info(f"Max features: {max_features}, Mean cross-validation score: {mean_cv_score}")
    accuracy_results.append((max_features, mean_cv_score))

# Convert accuracy results to a DataFrame for easy plotting
accuracy_df = pd.DataFrame(accuracy_results, columns=['max_features', 'mean_accuracy'])

# Identify the optimal max_features value at each split that resulted in the highest mean cross-validation accuracy
optimal_max_features = accuracy_df.loc[accuracy_df['mean_accuracy'].idxmax(), 'max_features']
logging.info(f"Optimal max_features: {optimal_max_features}")

# Plot the accuracy results
plt.figure(figsize=(10, 6))
plt.plot(accuracy_df['max_features'], accuracy_df['mean_accuracy'], marker='o')
plt.title('Accuracy vs. Number of Features Available at Each Split')
plt.xlabel('Number of Features at Each Split (max_features)')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.grid(True)
plt.axvline(x=optimal_max_features, color='r', linestyle='--', label=f'Optimal max_features: {optimal_max_features}')
plt.legend()
plt.show()

# Train the final model with the optimal max_features
rf_model = RandomForestClassifier(n_estimators=250, max_features=optimal_max_features, random_state=42, class_weight='balanced')

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model using classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the trained model
model_output_path = 'all_ML_rf_undersampled_optimized_2.pkl'
joblib.dump(rf_model, model_output_path)
logging.info(f"Model saved to {model_output_path}")

# Calculate and display the feature importances for the model
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature importances:")
print(importance_df.to_string(index=False))  # Display all feature importances

# Save feature importances to a CSV file
importance_df.to_csv('feature_importances_optimized.csv', index=False)
