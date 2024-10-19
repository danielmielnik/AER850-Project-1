import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error #remove
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler

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

# Scaling
my_scaler = StandardScaler()
my_scaler.fit(coord_train)
scaled_data_train = my_scaler.transform(coord_train)
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns = coord_train.columns)
coord_train = scaled_data_train_df

scaled_data_test = my_scaler.transform(coord_test)
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns = coord_test.columns)
coord_test = scaled_data_test_df

# Model 1: Random Forrest
# Training
RandomForrest = RandomForestClassifier(random_state=4)
RandomForrest.fit(coord_train, step_train)
RandomForrest_Pred = RandomForrest.predict(coord_test)
#RF_mae_train = mean_absolute_error(RandomForrest_Pred, step_test)
#print("Model 1 training MAE is: ", round(RF_mae_train,2))


# Cross Val,idation
param_grid_RF = {
     'n_estimators': [10, 30, 50],
     'max_depth': [None, 10, 20, 30],
     'min_samples_split': [2, 5, 10],
     'min_samples_leaf': [1, 2, 4],
     'max_features': ['sqrt', 'log2']
 }

grid_search_RF = GridSearchCV(RandomForrest, param_grid_RF, cv=5, scoring='f1_weighted', n_jobs=1)
grid_search_RF.fit(coord_train, step_train)
best_params_RF = grid_search_RF.best_params_
print("Best Hyperparameters:", best_params_RF)
best_model_RF = grid_search_RF.best_estimator_
RandomForrest_Pred = best_model_RF.predict(coord_test)

# Performance Anaylysis
RF_accuracy_score = accuracy_score(step_test, RandomForrest_Pred)
RF_confusion_matrix = confusion_matrix(step_test, RandomForrest_Pred)
RF_classification_report = classification_report(step_test, RandomForrest_Pred)

print("Model 1 Performance Analysis: Random Forrest\n")
print("Accuracy Score:", RF_accuracy_score)
sb.heatmap(RF_confusion_matrix)
plt.show()

#print("\nConfusion Matrix:\n", RF_confusion_matrix)
print("\nClassification Report:\n", RF_classification_report)

# Model 2: Support Vector Machine (SVM)
SVM = SVC()
SVM.fit(coord_train, step_train)
SVM_pred = SVM.predict(coord_test)

# Cross Val,idation
param_grid_SVM = {
     'C': [0.1, 1, 5],
     'kernel': ['linear', 'poly', 'rbf'],
     'class_weight': ['Balanced', None]
 }

grid_search_SVM = GridSearchCV(SVM, param_grid_SVM, cv=5, scoring='f1_weighted', n_jobs=1)
grid_search_SVM.fit(coord_train, step_train)
best_params_SVM = grid_search_SVM.best_params_
print("Best Hyperparameters:", best_params_SVM)
best_model_SVM = grid_search_SVM.best_estimator_
SVM_Pred = best_model_SVM.predict(coord_test)

# Performance Anaylysis
SVM_accuracy_score = accuracy_score(step_test, SVM_pred)
SVM_confusion_matrix = confusion_matrix(step_test, SVM_pred)
SVM_classification_report = classification_report(step_test, SVM_pred)

print("Model 2 Performance Analysis: Support Vector Machine\n")
print("Accuracy Score:", SVM_accuracy_score)
print("\nConfusion Matrix:\n", SVM_confusion_matrix)
print("\nClassification Report:\n", SVM_classification_report)

# Model 3: Logistic Regression
LogisticReg = LogisticRegression()
LogisticReg.fit(coord_train, step_train)
LogisticReg_pred = LogisticReg.predict(coord_test)

# Cross Val,idation
param_grid_LR = {
     'C': [0.01, 0.1, 1, 10, 100],
     'max_iter': [100, 250, 500, 1000],
     'class_weight': ['Balanced', None]
 }

grid_search_LR = GridSearchCV(LogisticReg, param_grid_LR, cv=5, scoring='f1_weighted', n_jobs=1)
grid_search_LR.fit(coord_train, step_train)
best_params_LR = grid_search_LR.best_params_
print("Best Hyperparameters:", best_params_LR)
best_model_LR = grid_search_LR.best_estimator_
RandomForrest_Pred = best_model_LR.predict(coord_test)

# Performance Anaylysis
LogisticReg_accuracy_score = accuracy_score(step_test, LogisticReg_pred)
LogisticReg_confusion_matrix = confusion_matrix(step_test, LogisticReg_pred)
LogisticReg_classification_report = classification_report(step_test, LogisticReg_pred)

print("Model 3 Performance Analysis: Logistic Regression\n")
print("Accuracy Score:", LogisticReg_accuracy_score)
print("\nConfusion Matrix:\n", LogisticReg_confusion_matrix)
print("\nClassification Report:\n", LogisticReg_classification_report)

# Model 4: RandomizedCV



# Part 6: Stacked Model Performance Analysis
combined_model = [('SVM', best_model_SVM), ('RandomForrest', best_model_RF)]

final_model = LogisticRegression()

StackedModel = StackingClassifier(combined_model, final_model, cv = 5)
StackedModel.fit(coord_train, step_train)
SM_pred = StackedModel.predict(coord_test)

StackedModel_accuracy_score = accuracy_score(step_test, SM_pred)
StackedModel_confusion_matrix = confusion_matrix(step_test, SM_pred)
StackedModel_classification_report = classification_report(step_test, SM_pred)

print("Stacked Model Performance Analysis\n")
print("Accuracy Score:", StackedModel_accuracy_score)
print("\nConfusion Matrix:\n", StackedModel_confusion_matrix)
print("\nClassification Report:\n", StackedModel_classification_report)

    
# Part 7: Model Evaluation
joblib.dump(RandomForrest, 'rf_model.joblib')
loaded_rf_model = joblib.load('rf_model.joblib')

# Coordinates for prediction
coordinates_to_predict = np.array([[9.375, 3.0625, 1.51],
                                   [6.995, 5.125, 0.3875],
                                   [0, 3.0625, 1.93],
                                   [9.4, 3, 1.8],
                                   [9.4, 3, 1.3]])

# Make predictions using the loaded model for each set of coordinates
for idx, coord_set in enumerate(coordinates_to_predict):
    step_prediction = loaded_rf_model.predict(coord_set.reshape(1, -1))
    print(f"Predicted Step for coordinates {idx + 1}: {step_prediction}")


