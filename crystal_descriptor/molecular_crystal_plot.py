import matplotlib.pyplot as plt

# Data for Bagging (Molecular descriptor model)
mol_bagging_conglomerates_loss = [0, 0.58, 1.46, 2.92, 4.08, 6.41, 8.75, 13.12, 17.2, 25.36]
mol_bagging_total_data_loss = [6.27, 10.72, 14.83, 19.3, 23.9, 29.22, 34.9, 41.28, 49.27, 57.88]

# Data for Random Forest (Molecular descriptor model)
#mol_rf_conglomerates_loss = [0, 0.29, 0.87, 2.04, 3.5, 5.54, 9.04, 14.58, 19.53, 28.86]
#mol_rf_total_data_loss = [4.05, 8.38, 13.31, 18.15, 23.30, 29.51, 36.36, 43.02, 50.29, 58.45]

# Data for Pruning (Molecular descriptor model)
#mol_pruning_conglomerates_loss = [5.54, 6.41, 11.66, 11.66, 15.74, 25.66]
#mol_pruning_total_data_loss = [23.79, 28.04, 37.28, 37.35, 40.51, 49.29]

# Data for Boosting (Molecular descriptor model)
mol_boosting_conglomerates_loss = [0, 0.58, 1.46, 4.37, 8.75, 11.66, 15.16, 19.53, 23.91, 26.24, 29.45, 30.03]
mol_boosting_total_data_loss = [10.63, 17.38, 21.41, 27.46, 35.32, 40.54, 42.91, 48.57, 53.34, 56.78, 59.13, 60.15]


# Data for Bagging (Crystal descriptor model)
crys_bagging_conglomerates_loss = [0, 0.29, 0.58, 0.87, 3.5, 11.66, 19.83, 25.66, 29.45, 37.9]
crys_bagging_total_data_loss = [6.26, 6.97, 11.07, 15, 22.6, 34.47, 45.45, 53.78, 61.32, 68.49]

# Data for Random Forest (Crystal descriptor model)
#crys_rf_conglomerates_loss = [0, 0.29, 1.17, 2.33, 5.25, 10.79, 16.62, 27.41, 37.03]
#crys_rf_total_data_loss = [5.8, 9.75, 14.89, 19.51, 24.17, 30.58, 40.04, 53.96, 65.87]

# Data for Pruning (Crystal descriptor model)
#crys_pruning_conglomerates_loss = [1.75, 3.5, 6.41, 11.37, 13.99, 21.87, 25.66, 28.28]
#crys_pruning_total_data_loss = [1.02, 6.95, 20.22, 27.8, 29.04, 38.33, 44.78, 46.51]

# Data for Boosting (Crystal descriptor model)
crys_boosting_conglomerates_loss = [0, 0.58, 2.04, 3.21, 4.08, 13.12, 23.91, 34.99]
crys_boosting_total_data_loss = [4.23, 10.06, 16.85, 23.17, 29.37, 39.51, 50.76, 64.67]

# Data for XGBoost (Crystal descriptor model)
crys_xgboost_conglomerates_loss = [0, 0.87, 2.04, 8.75, 13.12, 14.58, 18.37, 22.16, 28.57, 32.94]
crys_xgboost_total_data_loss = [7.63, 11.36, 16.08, 28.92, 41.28, 46.75, 51.05, 56.69, 62.38, 67.81]

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(mol_bagging_total_data_loss, mol_bagging_conglomerates_loss, marker='o', linestyle='-', label='Bagging (Molecular descriptors)')
#plt.plot(mol_rf_total_data_loss, mol_rf_conglomerates_loss, marker='o', linestyle='-', label='Random Forest (Molecular descriptors)')
#plt.plot(mol_pruning_total_data_loss, mol_pruning_conglomerates_loss, marker='o', linestyle='-', label='Pruning (Molecular descriptors)')
plt.plot(mol_boosting_total_data_loss, mol_boosting_conglomerates_loss, marker='o', linestyle='-', label='Boosting (Molecular descriptors)')

plt.plot(crys_bagging_total_data_loss, crys_bagging_conglomerates_loss, marker='o', linestyle='-', label='Bagging (Crystal descriptors)')
#plt.plot(crys_rf_total_data_loss, crys_rf_conglomerates_loss, marker='o', linestyle='-', label='Random Forest (Crystal descriptors)')
#plt.plot(crys_pruning_total_data_loss, crys_pruning_conglomerates_loss, marker='o', linestyle='-', label='Pruning (Crystal descriptors)')
plt.plot(crys_boosting_total_data_loss, crys_boosting_conglomerates_loss, marker='o', linestyle='-', label='Boosting (Crystal descriptors)')
plt.plot(crys_xgboost_total_data_loss, crys_xgboost_conglomerates_loss, marker='o', linestyle='-', label='XGBoost (Crystal descriptors)')

plt.title('Conglomerates Data Loss vs Total Data Removed across Thresholds between Best Performing Molecular and Crystal Descriptor Models', fontsize=12)
plt.xlabel('% of Total Data Removed', fontsize=14)
plt.ylabel('% of Conglomerates Data Lost', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
