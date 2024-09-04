import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
congloms_descriptors_df = pd.read_csv('congloms_descriptors_all.csv')
non_congloms_descriptors_df = pd.read_csv('non_congloms_descriptors_all.csv')

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

# Apply random undersampling to balance the dataset between conglomerates and non-conglomerates
undersampler = RandomUnderSampler(random_state=42)
X_res, y_res = undersampler.fit_resample(X, y)

# Initialize an unpruned DecisionTreeClassifier
unpruned_tree_model = DecisionTreeClassifier(random_state=42)

# Perform cross-validation to evaluate the unpruned tree
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(unpruned_tree_model, X_res, y_res, cv=cv, scoring='accuracy')
logging.info(f"Unpruned Tree Cross-validation scores: {cv_scores}")
logging.info(f"Mean cross-validation score for unpruned tree: {cv_scores.mean()}")

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train the unpruned tree model on the training data
unpruned_tree_model.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = unpruned_tree_model.predict(X_test)

# Evaluate the unpruned tree model using classification report and confusion matrix
print("Classification Report for Unpruned Tree:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix for Unpruned Tree:")
print(confusion_matrix(y_test, y_pred))

# Save the trained unpruned tree model
unpruned_model_output_path = 'unpruned_initial_tree_model_6.pkl'
joblib.dump(unpruned_tree_model, unpruned_model_output_path)
logging.info(f"Unpruned tree model saved to {unpruned_model_output_path}")

# Plot the decision tree structure to visualize how the unpruned tree makes decisions
plt.figure(figsize=(20, 10))
plot_tree(unpruned_tree_model, filled=True, feature_names=X.columns, class_names=['Non-Conglomerate', 'Conglomerate'])
plt.title("Unpruned Decision Tree")
plt.show()

# Calculate and display the feature importances for the unpruned tree
feature_importances = unpruned_tree_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature importances for Unpruned Tree:")
print(importance_df.to_string(index=False))  # Display all feature importances

# Save feature importances to a CSV file
importance_df.to_csv('feature_importances_unpruned_tree.csv', index=False)
