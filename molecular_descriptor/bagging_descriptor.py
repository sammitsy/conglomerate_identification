import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib
import logging
import matplotlib.pyplot as plt

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

# Initialize variables to track the best number of trees and the corresponding accuracy
best_num_trees = 0
best_accuracy = 0.0
num_trees_list = []
accuracy_list = []

# Loop over different numbers of trees to find the optimal number of trees
for num_trees in range(5, 300, 5):
    bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=num_trees, random_state=42)

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(bagging_model, X_res, y_res, cv=cv, scoring='accuracy')
    mean_cv_score = cv_scores.mean()
    
    # Log the number of trees and the corresponding cross-validation accuracy
    logging.info(f"Number of trees: {num_trees}, Mean cross-validation score: {mean_cv_score}")
    
    # Store results
    num_trees_list.append(num_trees)
    accuracy_list.append(mean_cv_score)
    
    # Update the best model if the current model has the highest accuracy so far
    if mean_cv_score > best_accuracy:
        best_accuracy = mean_cv_score
        best_num_trees = num_trees

# Plot the accuracy for each model against the number of trees used in the BaggingClassifier
plt.figure(figsize=(10, 6))
plt.plot(num_trees_list, accuracy_list, marker='o', linestyle='-', color='blue')
plt.axvline(x=best_num_trees, color='green', linestyle='--', label=f'Best Model: {best_num_trees} Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Bagging Classifier Accuracy vs. Number of Trees')
plt.legend()
plt.grid(True)
plt.show()

# Log the optimal number of trees and the corresponding accuracy
logging.info(f"Best number of trees: {best_num_trees} with accuracy: {best_accuracy}")

# Train the final BaggingClassifier using the optimal number of trees
final_bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=best_num_trees, random_state=42)
final_bagging_model.fit(X_res, y_res)

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
final_bagging_model.fit(X_train, y_train)

# Predict on the test set
y_pred = final_bagging_model.predict(X_test)

# Evaluate the model using classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the trained model
model_output_path = 'all_ML_bagging_best_model_2.pkl'
joblib.dump(final_bagging_model, model_output_path)
logging.info(f"Model saved to {model_output_path}")

# Feature importances are not directly available in BaggingClassifier; access them through individual base estimators
importances = np.zeros(X.shape[1])
for estimator in final_bagging_model.estimators_:
    if hasattr(estimator, 'feature_importances_'):
        importances += estimator.feature_importances_

# Average the importances across all base estimators
importances /= len(final_bagging_model.estimators_)

# Create a DataFrame to display the feature importances
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature importances:")
print(importance_df.to_string(index=False))  # Display all feature importances

# Save feature importances to a CSV file
importance_df.to_csv('feature_importances_bagging_best_model_2.csv', index=False)
