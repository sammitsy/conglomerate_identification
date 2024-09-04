import matplotlib.pyplot as plt

# Data for Bagging
bagging_conglomerates_loss = [0, 0.58, 1.46, 2.92, 4.08, 6.41, 8.75, 13.12, 17.2, 25.36]
bagging_total_data_loss = [6.27, 10.72, 14.83, 19.3, 23.9, 29.22, 34.9, 41.28, 49.27, 57.88]

# Data for Random Forest
rf_conglomerates_loss = [0, 0.29, 0.87, 2.04, 3.5, 5.54, 9.04, 14.58, 19.53, 28.86]
rf_total_data_loss = [4.05, 8.38, 13.31, 18.15, 23.30, 29.51, 36.36, 43.02, 50.29, 58.45]

# Data for Pruning
pruning_conglomerates_loss = [5.54, 6.41, 11.66, 11.66, 15.74, 25.66]
pruning_total_data_loss = [23.79, 28.04, 37.28, 37.35, 40.51, 49.29]

# Data for Boosting
boosting_conglomerates_loss = [0, 0.58, 1.46, 4.37, 8.75, 11.66, 15.16, 19.53, 23.91, 26.24, 29.45, 30.03]
boosting_total_data_loss = [10.63, 17.38, 21.41, 27.46, 35.32, 40.54, 42.91, 48.57, 53.34, 56.78, 59.13, 60.15]

# Data for XGBoost
xgboost_conglomerates_loss = [0, 0.29, 0.87, 1.46, 6.41, 10.79, 13.99, 16.03, 18.37, 24.2, 25.95, 30.03, 31.49]
xgboost_total_data_loss = [7.02, 7.48, 15.25, 20.03, 31.6, 37.39, 41.51, 45.31, 48.59, 53.41, 55.92, 58.69, 60.89]

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(bagging_total_data_loss, bagging_conglomerates_loss, marker='o', linestyle='-', label='Bagging')
plt.plot(rf_total_data_loss, rf_conglomerates_loss, marker='o', linestyle='-', label='Random Forest')
plt.plot(pruning_total_data_loss, pruning_conglomerates_loss, marker='o', linestyle='-', label='Pruning')
plt.plot(boosting_total_data_loss, boosting_conglomerates_loss, marker='o', linestyle='-', label='Boosting')
plt.plot(xgboost_total_data_loss, xgboost_conglomerates_loss, marker='o', linestyle='-', label='XGBoost')

plt.title('Conglomerates Data Loss vs Total Data Removed across Thresholds for Molecular Descriptor Models', fontsize=12)
plt.xlabel('% of Total Data Removed', fontsize=12)
plt.ylabel('% of Conglomerates Data Lost', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
