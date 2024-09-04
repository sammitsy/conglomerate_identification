import matplotlib.pyplot as plt

# Data for Bagging
bagging_conglomerates_loss = [0, 0.29, 0.58, 0.87, 3.5, 11.66, 19.83, 25.66, 29.45, 37.9]
bagging_total_data_loss = [6.26, 6.97, 11.07, 15, 22.6, 34.47, 45.45, 53.78, 61.32, 68.49]

# Data for Random Forest
rf_conglomerates_loss = [0, 0.29, 1.17, 2.33, 5.25, 10.79, 16.62, 27.41, 37.03]
rf_total_data_loss = [5.8, 9.75, 14.89, 19.51, 24.17, 30.58, 40.04, 53.96, 65.87]

# Data for Pruning
pruning_conglomerates_loss = [1.75, 3.5, 6.41, 11.37, 13.99, 21.87, 25.66, 28.28]
pruning_total_data_loss = [1.02, 6.95, 20.22, 27.8, 29.04, 38.33, 44.78, 46.51]

# Data for Boosting
boosting_conglomerates_loss = [0, 0.58, 2.04, 3.21, 4.08, 13.12, 23.91, 34.99]
boosting_total_data_loss = [4.23, 10.06, 16.85, 23.17, 29.37, 39.51, 50.76, 64.67]

# Data for XGBoost
xgboost_conglomerates_loss = [0, 0.87, 2.04, 8.75, 13.12, 14.58, 18.37, 22.16, 28.57, 32.94]
xgboost_total_data_loss = [7.63, 11.36, 16.08, 28.92, 41.28, 46.75, 51.05, 56.69, 62.38, 67.81]


# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(bagging_total_data_loss, bagging_conglomerates_loss, marker='o', linestyle='-', label='Bagging')
plt.plot(rf_total_data_loss, rf_conglomerates_loss, marker='o', linestyle='-', label='Random Forest')
plt.plot(pruning_total_data_loss, pruning_conglomerates_loss, marker='o', linestyle='-', label='Pruning')
plt.plot(boosting_total_data_loss, boosting_conglomerates_loss, marker='o', linestyle='-', label='Boosting')
plt.plot(xgboost_total_data_loss, xgboost_conglomerates_loss, marker='o', linestyle='-', label='XGBoost')

plt.title('Conglomerates Data Loss vs Total Data Removed across Thresholds for Crystal Descriptor Models', fontsize=12)
plt.xlabel('% of Total Data Removed', fontsize=12)
plt.ylabel('% of Conglomerates Data Lost', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
