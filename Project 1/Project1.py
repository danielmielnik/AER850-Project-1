import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Step 1: Data Processing
df = pd.read_csv('Project_1_Data.csv')
print(df)

# Step 2: Data Visualization
# Individual Steps
for i in range(1, 14):
    step = df.loc[df['Step'] == i]
    print(step)
    
    X = step.get("X")
    Y = step.get("Y")
    Z = step.get("Z")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X, Y, Z)
    # Add same axis for each
    
    plt.show()

# Combined Steps
X = df['X']
Y = df['Y']
Z = df['Z'] 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X, Y, Z)

# Step 3: Correlation Analysis
correlation = df.corr()
print(correlation)

plt.figure()
sb.heatmap(correlation)

# Step 4 and 5: Classification Model Development and Performance Analysis
coordinates = df[['X', 'Y', 'Z']]
step = df['Step']
print(coordinates, step)

coord_train, coord_test, step_train, step_test = train_test_split(coordinates, step, test_size=0.2, random_state = 42)

# Model 1: Random Forrest
# Training
RandomForrest = RandomForestClassifier()
RandomForrest.fit(coord_train, step_train)
RandomForrest_Pred = RandomForrest.predict(coord_test)
RF_mae_train = mean_absolute_error(RandomForrest_Pred, step_test)
print("Model 2 training MAE is: ", round(RF_mae_train,2))


# Cross Validation
RandomForrest_cv_score = cross_val_score(RandomForrest, coord_train, step_train, cv=5, scoring='neg_mean_absolute_error')
RF_cv_mae = -RandomForrest_cv_score.mean()
print("Model 2 Mean Absolute Error (CV):", round(RF_cv_mae, 2))

param_grid = {
     'n_estimators': [10, 30, 50],
     'max_depth': [None, 10, 20, 30],
     'min_samples_split': [2, 5, 10],
     'min_samples_leaf': [1, 2, 4],
     'max_features': ['sqrt', 'log2']
 }
my_model3 = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(my_model3, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
grid_search.fit(coord_train, step_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_model3 = grid_search.best_estimator_

# Performance Anaylysis
RF_accuracy = accuracy_score(step_test, RandomForrest_Pred)

