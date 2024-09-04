import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# Initialize a DecisionTreeClassifier to find the cost-complexity pruning path
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_res, y_res)
# Extract the effective alpha values (ccp_alphas) and corresponding tree impurities from the pruning path
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Plot the total impurity of leaves against the effective alpha values to visualize the pruning process
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
plt.xlabel("Effective alpha")
plt.ylabel("Total impurity of leaves")
plt.title("Total Impurity vs effective alpha for training set")
plt.show()

# Cross-validate for each alpha value and determine the best alpha value
cv_scores = []
for ccp_alpha in ccp_alphas:
    model = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='accuracy')
    cv_scores.append(scores.mean())

# Find the best alpha that has the highest cross-validated accuracy
best_alpha_idx = np.argmax(cv_scores)
best_ccp_alpha = ccp_alphas[best_alpha_idx]

# Log the best alpha value and the corresponding cross-validation score
logging.info(f"Best ccp_alpha: {best_ccp_alpha} with cross-validation score: {cv_scores[best_alpha_idx]}")

# Re-train the model using the best ccp_alpha
pruning_model = DecisionTreeClassifier(random_state=42, ccp_alpha=best_ccp_alpha)

# Split the data into a training set and a testing set for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
pruning_model.fit(X_train, y_train)

# Predict on the test set
y_pred = pruning_model.predict(X_test)

# Evaluate the model using classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
model_output_path = 'all_ML_pruned_tree_cut3_optimized.pkl'
joblib.dump(pruning_model, model_output_path)
logging.info(f"Model saved to {model_output_path}")

# Plot the structure of the pruned decision tree
plt.figure(figsize=(20, 10))
plot_tree(pruning_model, filled=True, feature_names=X.columns, class_names=['Non-Conglomerate', 'Conglomerate'])
plt.show()

# Calculate and display the feature importances for the model
feature_importances = pruning_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature importances:")
print(importance_df.to_string(index=False))  # Display all feature importances

# Save feature importances to a CSV file
importance_df.to_csv('feature_importances_pruned_tree_optimized.csv', index=False)
