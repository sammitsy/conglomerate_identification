# Machine Learning Models for Identification of Conglomerate Crystals
This project aims to classify crystal structures as either conglomerates or non-conglomerates using tree-based machine learning models. Conglomerates are a unique class of crystal structures that exhibit specific properties, especially in chiral systems. This project leverages molecular and crystal descriptors to develop models capable of predicting these classifications, helping researchers understand and predict structural behaviours more effectively.

## Data Sources:
CSD (Cambridge Structural Database):
- Contains a vast repository of crystallographic data that provides the structural basis for this project. Training data is gained from Version 5.41 (November 2019), and unseen validation data is from Version 5.43 (November 2021).

Molecular Descriptors from RDKit:
- A way to represent the data gained from the CSD numerically based on molecular-related features.

Crystal Descriptors from ChiPi (Clever & Coquerel, 2020):
- Another way to represent the data gained from the CSD numerically based on crystal-related features.

## Related Research Publications:
Walsh, M. P., Barclay, J. A., Begg, C. S., Xuan, J., Johnson, N. T., Cole, J. C., & Kitching, M. O. (2022). Identifying a hidden conglomerate chiral pool in the CSD. JACS Au, 2(10), 2235-2250.

Walsh, M. P., Barclay, J. A., Begg, C. S., Xuan, J., & Kitching, M. O. (2023). Conglomerate crystallization in the cambridge structural database (2020–2021). Crystal Growth & Design, 23(4), 2837-2844.

## Working Environment:
Python 3.11.4

## Overview of the Codes:
### data_filtering Folder:
compound_name.py: 
- Filters out structures based on specific patterns (indicators '(+)', '(-)', '(S)', '(R)') in the compound names.

filter_pvalue.py: 
- Calculates p-values for hypothesis testing regarding the three filtering methods.

### molecular_descriptor Folder:
descriptor.py: 
- Calculates molecular descriptors from SMILES strings and prepares them for machine learning.

initial_descriptor.py: 
- Trains an initial unpruned decision tree model using molecular descriptors.

bagging_descriptor.py: 
- Trains a bagging model using molecular descriptors.

RF_descriptor.py: 
- Trains a Random Forest model using molecular descriptors.

pruning_descriptor.py: 
- Trains a pruned decision tree model using molecular descriptors.

boosting_descriptor.py: 
- Trains a Gradient Boosting model using molecular descriptors.

xgboost_descriptor.py: 
- Trains an XGBoost model using molecular descriptors.

apply_ML_descriptor.py: 
- Applies a trained model with molecular descriptors to the unseen validation dataset to evaluate model performance.

descriptor_plot.py: 
- Plots the performance of molecular descriptor models.

### crystal_descriptor Folder:
initial_crystal.py: 
- Trains an initial unpruned decision tree model using crystal descriptors.

bagging_crystal.py: 
- Trains a bagging model using crystal descriptors.

RF_crystal.py: 
- Trains a Random Forest model using crystal descriptors.

pruning_crystal_ccp.py: 
- Trains a pruned decision tree model using crystal descriptors.

boosting_crystal.py: 
- Trains a Gradient Boosting model using crystal descriptors.

xgboost_crystal.py: 
- Trains an XGBoost model using crystal descriptors.

apply_ML_crystal.py: 
- Applies a trained model with crystal descriptors to the unseen validation dataset to evaluate model performance.

crystal_plot.py: 
- Plots the performance of crystal descriptor models.

molecular_crystal_plot.py: 
- Provides comparative plots between molecular and crystal descriptor models, showcasing data loss and performance across different thresholds.


