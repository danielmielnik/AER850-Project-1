import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
print("Model 1 training MAE is: ", round(RF_mae_train,2))


# Cross Val,idation
RandomForrest_cv_score = cross_val_score(RandomForrest, coord_train, step_train, cv=5, scoring='neg_mean_absolute_error')
RF_cv_mae = -RandomForrest_cv_score.mean()
print("Model 1 Mean Absolute Error (CV):", round(RF_cv_mae, 2))

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
RF_accuracy_score = accuracy_score(step_test, RandomForrest_Pred)
RF_confusion_matrix = confusion_matrix(step_test, RandomForrest_Pred)
RF_classification_report = classification_report(step_test, RandomForrest_Pred)

print("Model 1 Performance Analysis: Random Forrest\n")
print("Accuracy Score:", RF_accuracy_score)
print("\nConfusion Matrix:\n", RF_confusion_matrix)
print("\nClassification Report:\n", RF_classification_report)

# Model 2: Support Vector Machine (SVM)
SVM = SVC()
SVM.fit(coord_train, step_train)
SVM_pred = SVM.predict(coord_test)
SVM_mae_train = mean_absolute_error(SVM_pred, step_test)
print("Model 2 training MAE is: ", round(SVM_mae_train,2))

# Cross Val,idation
SVM_cv_score = cross_val_score(SVM, coord_train, step_train, cv=5, scoring='neg_mean_absolute_error')
SVM_cv_mae = -SVM_cv_score.mean()
print("Model 2 Mean Absolute Error (CV):", round(SVM_cv_mae, 2))

# Performance Anaylysis
SVM_accuracy_score = accuracy_score(step_test, SVM_pred)
SVM_confusion_matrix = confusion_matrix(step_test, SVM_pred)
SVM_classification_report = classification_report(step_test, SVM_pred)

print("Model 2 Performance Analysis: Support Vector Machine\n")
print("Accuracy Score:", SVM_accuracy_score)
print("\nConfusion Matrix:\n", SVM_confusion_matrix)
print("\nClassification Report:\n", SVM_classification_report)

# Model 3: Linnear Regression
LinnearReg = LogisticRegression()
LinnearReg.fit(coord_train, step_train)
LinnearReg_pred = LinnearReg.predict(coord_test)
LinnearReg_mae_train = mean_absolute_error(LinnearReg_pred, step_test)
print("Model 3 training MAE is: ", round(LinnearReg_mae_train,2))

# Cross Val,idation
LinnearReg_cv_score = cross_val_score(LinnearReg, coord_train, step_train, cv=5, scoring='neg_mean_absolute_error')
LinnearReg_cv_mae = -LinnearReg_cv_score.mean()
print("Model 3 Mean Absolute Error (CV):", round(LinnearReg_cv_mae, 2))

# Performance Anaylysis
LinnearReg_accuracy_score = accuracy_score(step_test, LinnearReg_pred)
LinnearReg_confusion_matrix = confusion_matrix(step_test, LinnearReg_pred)
LinnearReg_classification_report = classification_report(step_test, LinnearReg_pred)

print("Model 3 Performance Analysis: Linnear Regression\n")
print("Accuracy Score:", LinnearReg_accuracy_score)
print("\nConfusion Matrix:\n", LinnearReg_confusion_matrix)
print("\nClassification Report:\n", LinnearReg_classification_report)


