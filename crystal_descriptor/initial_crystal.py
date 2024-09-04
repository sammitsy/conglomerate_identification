import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
continuous_features = ['a', 'b', 'c', 'Cell Volume', 'Calc. Density', 'Beta',
                       'R-factor', 'Number of chiral center', 
                       'Number of Chiral Center having H'
                       ]
categorical_feature = ['Space Gp. Number']
X = combined_df[continuous_features + categorical_feature].copy()
y = combined_df['label']

# Convert continuous features to numeric data types, handling any errors by converting them to NaN
for column in continuous_features:
    X[column] = pd.to_numeric(X[column], errors='coerce')

# Preprocessing steps:
# 1. Impute missing values by replacing them with the mean of the column
# 2. Scale the features to have a mean of 0 and standard deviation of 1 (standardization)
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Apply the preprocessing steps to the predictor variables
X_preprocessed = preprocessor.fit_transform(X)

# Apply random undersampling to balance the dataset between conglomerates and non-conglomerates
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_preprocessed, y)

# Train an initial unpruned classification tree
unpruned_tree_model = DecisionTreeClassifier(random_state=42)

# Perform cross-validation to evaluate the unpruned tree
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(unpruned_tree_model, X_res, y_res, cv=cv, scoring='accuracy')
logging.info(f"Cross-validation scores: {cv_scores}")
logging.info(f"Mean cross-validation score: {cv_scores.mean()}")

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
# Train the unpruned tree model on the training data
unpruned_tree_model.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = unpruned_tree_model.predict(X_test)

# Evaluate the unpruned tree model using classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the trained unpruned tree model
model_output_path = 'unpruned_initial_classification_tree_3.pkl'
joblib.dump(unpruned_tree_model, model_output_path)
logging.info(f"Model saved to {model_output_path}")

# Plot the decision tree structure to visualize how the unpruned tree makes decisions
plt.figure(figsize=(20, 10))
plot_tree(unpruned_tree_model, filled=True, feature_names=X.columns, class_names=['Non-Conglomerate', 'Conglomerate'])
plt.title('Unpruned Initial Classification Tree')
plt.show()

# Calculate and display the feature importances for the unpruned tree
feature_importances = unpruned_tree_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature importances:")
print(importance_df.to_string(index=False))  # Display all feature importances

# Save feature importances to a CSV file
importance_df.to_csv('unpruned_tree_feature_importances.csv', index=False)
