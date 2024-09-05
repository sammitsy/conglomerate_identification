import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
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

# Set the number of trees and initialize variables for tracking the best parameters
num_trees = 250
best_depth = 0
best_shrinkage = 0.0
best_accuracy = 0.0
accuracy_results = []

# Step 1: Optimize interaction depth (max_depth)
for depth in range(1, 21):
    boosting_model = GradientBoostingClassifier(n_estimators=num_trees, max_depth=depth, random_state=42)

    # Perform cross-validation to evaluate the model's accuracy for each depth
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(boosting_model, X_res, y_res, cv=cv, scoring='accuracy')
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
plt.title('Accuracy vs. Interaction Depth in Boosting')
plt.xlabel('Interaction Depth')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.grid(True)
plt.axvline(x=best_depth, color='r', linestyle='--', label=f'Optimal Depth: {best_depth}')
plt.legend()
plt.show()

logging.info(f"Optimal interaction depth: {best_depth}")

# Step 2: Optimize shrinkage parameter
# Reset best_accuracy and test different values for learning_rate (shrinkage) to find the optimal rate
best_accuracy = 0.0
shrinkage_results = []
shrinkage_values = np.linspace(0.0025, 0.25, 50)

for shrinkage in shrinkage_values:
    boosting_model = GradientBoostingClassifier(n_estimators=num_trees, max_depth=best_depth, learning_rate=shrinkage, random_state=42)

    # Perform cross-validation to evaluate the model's accuracy for each shrinkage value
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(boosting_model, X_res, y_res, cv=cv, scoring='accuracy')
    mean_cv_score = cv_scores.mean()

    logging.info(f"Shrinkage: {shrinkage}, Mean cross-validation score: {mean_cv_score}")
    shrinkage_results.append((shrinkage, mean_cv_score))

    # Update the best shrinkage if the current rate results in a higher accuracy
    if mean_cv_score > best_accuracy:
        best_accuracy = mean_cv_score
        best_shrinkage = shrinkage

# Plot shrinkage optimization results
shrinkage_df = pd.DataFrame(shrinkage_results, columns=['shrinkage', 'mean_accuracy'])
plt.figure(figsize=(10, 6))
plt.plot(shrinkage_df['shrinkage'], shrinkage_df['mean_accuracy'], marker='o')
plt.title('Accuracy vs. Shrinkage Parameter in Boosting')
plt.xlabel('Shrinkage Parameter')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.grid(True)
plt.axvline(x=best_shrinkage, color='r', linestyle='--', label=f'Optimal Shrinkage: {best_shrinkage}')
plt.legend()
plt.show()

logging.info(f"Optimal shrinkage parameter: {best_shrinkage}")

# Step 3: Train the final boosted model with the optimal parameters
final_boosting_model = GradientBoostingClassifier(n_estimators=num_trees, max_depth=best_depth, learning_rate=best_shrinkage, random_state=42)
final_boosting_model.fit(X_res, y_res)

# Split the data into a training set and a testing set for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
final_boosting_model.fit(X_train, y_train)

# Predict on the test set
y_pred = final_boosting_model.predict(X_test)

# Evaluate the model using classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the trained model
model_output_path = 'all_ML_boosting_best_model.pkl'
joblib.dump(final_boosting_model, model_output_path)
logging.info(f"Model saved to {model_output_path}")

# Calculate and display the feature importances for the model
feature_importances = final_boosting_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature importances:")
print(importance_df.to_string(index=False))  # Display all feature importances

# Save feature importances to a CSV file
importance_df.to_csv('feature_importances_boosting_best_model.csv', index=False)
