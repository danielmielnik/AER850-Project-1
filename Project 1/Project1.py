import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Data Processing
df = pd.read_csv('Project_1_Data.csv')
print(df)

# Step 2: Data Visualization
count_step = df["Step"].value_counts()
count_step.plot(kind = "bar")
plt.title("Step Class Distribution")
plt.xlabel("Step")
plt.ylabel("No. of Instances")
plt.show()

X = df['X']
Y = df['Y']
Z = df['Z'] 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z)
plt.title("Step Class 3D Distribution")

coordinates = df[['X', 'Y', 'Z']]
step = df['Step']

#coord_train, coord_test, step_train, step_test = train_test_split(coordinates, step, test_size=0.2, random_state = 42)

my_splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 42)
for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
    
coord_train = strat_df_train.drop("Step", axis = 1)
step_train = strat_df_train["Step"]
coord_test = strat_df_test.drop("Step", axis = 1)
step_test = strat_df_test["Step"]

# Step 3: Correlation Analysis
correlation = coord_train.corr()
plt.figure()
sb.heatmap(correlation, cmap=sb.cubehelix_palette(as_cmap=True), annot=True)
plt.title("Correlation Matrix")

# Step 4 and 5: Classification Model Development and Performance Analysis
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
RandomForrest = RandomForestClassifier(class_weight='balanced', random_state=4)
RandomForrest.fit(coord_train, step_train)
RandomForrest_pred = RandomForrest.predict(coord_test)

# Cross Val,idation
param_grid_RF = {
     'n_estimators': [10, 30, 50],
     'max_depth': [None, 10, 20, 30],
     'min_samples_split': [2, 5, 10],
     'min_samples_leaf': [1, 2, 4, 8],
     'max_features': ['sqrt', 'log2']
 }

grid_search_RF = GridSearchCV(RandomForrest, param_grid_RF, cv=5, scoring='f1_weighted', n_jobs=1)
grid_search_RF.fit(coord_train, step_train)
best_params_RF = grid_search_RF.best_params_
best_model_RF = grid_search_RF.best_estimator_
RandomForrest_pred = best_model_RF.predict(coord_test)

# Performance Anaylysis
RF_accuracy_score = accuracy_score(step_test, RandomForrest_pred)
RF_f1_score = f1_score(step_test, RandomForrest_pred, average='weighted')
RF_precision_score = precision_score(step_test, RandomForrest_pred, average='weighted')
RF_classification_report = classification_report(step_test, RandomForrest_pred)

print("\nModel 1 Performance Analysis: Random Forrest")
print("\nBest Hyperparameters:", best_params_RF)
print("\nAccuracy Score:", RF_accuracy_score)
print("\nF1 Score:", RF_precision_score)
print("\nPrecision Score:", RF_f1_score)
print("\nClassification Report:\n", RF_classification_report)

# Model 2: Support Vector Machine (SVM)
# Training
SVM = SVC()
SVM.fit(coord_train, step_train)
SVM_pred = SVM.predict(coord_test)

# Cross Val,idation
param_grid_SVM = {
     'C': [0.1, 1, 5],
     'kernel': ['linear', 'poly', 'rbf'],
     'class_weight': [None]
 }

grid_search_SVM = GridSearchCV(SVM, param_grid_SVM, cv=5, scoring='f1_weighted', n_jobs=1)
grid_search_SVM.fit(coord_train, step_train)
best_params_SVM = grid_search_SVM.best_params_
best_model_SVM = grid_search_SVM.best_estimator_
SVM_pred = best_model_SVM.predict(coord_test)

# Performance Anaylysis
SVM_accuracy_score = accuracy_score(step_test, SVM_pred)
SVM_f1_score = f1_score(step_test, SVM_pred, average='weighted')
SVM_precision_score = precision_score(step_test, SVM_pred, average='weighted')
SVM_classification_report = classification_report(step_test, SVM_pred)

print("\nModel 2 Performance Analysis: Support Vector Machine")
print("\nBest Hyperparameters:", best_params_SVM)
print("\nAccuracy Score:", SVM_accuracy_score) # Seperate f1 score
print("\nF1 Score:", SVM_f1_score)
print("\nPrecision Score:", SVM_precision_score)
print("\nClassification Report:\n", SVM_classification_report)

# Model 3: Logistic Regression
# Training
LogisticReg = LogisticRegression(random_state=22)
LogisticReg.fit(coord_train, step_train)
LogisticReg_pred = LogisticReg.predict(coord_test)

# Cross Val,idation
param_grid_LR = {
     'C': [0.01, 0.1, 1, 10, 100],
     'max_iter': [1000],
     'class_weight': ['balanced', None]
 }

grid_search_LR = GridSearchCV(LogisticReg, param_grid_LR, cv=5, scoring='f1_weighted', n_jobs=1)
grid_search_LR.fit(coord_train, step_train)
best_params_LR = grid_search_LR.best_params_
best_model_LR = grid_search_LR.best_estimator_
LogisticReg_pred = best_model_LR.predict(coord_test)

# Performance Anaylysis
LogisticReg_accuracy_score = accuracy_score(step_test, LogisticReg_pred)
LogisticReg_f1_score = f1_score(step_test, LogisticReg_pred, average='weighted')
LogisticReg_precision_score = precision_score(step_test, LogisticReg_pred, average='weighted')
LogisticReg_classification_report = classification_report(step_test, LogisticReg_pred)

print("\nModel 3 Performance Analysis: Logistic Regression")
print("\nBest Hyperparameters:", best_params_LR)
print("\nAccuracy Score:", LogisticReg_accuracy_score)
print("\nF1 Score:", LogisticReg_f1_score)
print("\nPrecision Score:", LogisticReg_precision_score)
print("\nClassification Report:\n", LogisticReg_classification_report)

# Model 4: DT
# Training
DecTree = DecisionTreeClassifier(random_state=42)
DecTree.fit(coord_train, step_train)
DecTree_pred = DecTree.predict(coord_test)

# Randomized Cross Validation
param_grid_DT = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

RandomCV = RandomizedSearchCV(DecTree ,param_grid_DT, cv = 5, scoring = 'f1_weighted', n_jobs = 1)
RandomCV.fit(coord_train, step_train)
best_params_DT = RandomCV.best_params_
best_model_DT = RandomCV.best_estimator_
DecTree_pred = best_model_DT.predict(coord_test)

# Performance Anaylysis
DecTree_accuracy_score = accuracy_score(step_test, DecTree_pred)
DecTree_f1_score = f1_score(step_test, DecTree_pred, average='weighted')
DecTree_precision_score = precision_score(step_test, DecTree_pred, average='weighted')
DecTree_classification_report = classification_report(step_test, DecTree_pred)

print("\nModel 4 Performance Analysis: Decision Tree - RandomCV")
print("\nBest Hyperparameters:", best_params_DT)
print("\nAccuracy Score:", DecTree_accuracy_score)
print("\nF1 Score:", DecTree_f1_score)
print("\nPrecision Score:", DecTree_precision_score)
print("\nClassification Report:\n", DecTree_classification_report)


# Part 5: Confusion Matrix
f1_scores = {
    'RandomForrest_pred': RF_f1_score,
    'SVM_pred': SVM_f1_score,
    'LogisticReg_pred': LogisticReg_f1_score,
    'DecTree_pred': DecTree_f1_score}

model_preds = {
    'RandomForrest_pred': RandomForrest_pred,
    'SVM_pred': SVM_pred,
    'LogisticReg_pred': LogisticReg_pred,
    'DecTree_pred': DecTree_pred}

max_f1 = max(f1_scores, key=f1_scores.get)
print("\n\nMax F1 Score:", max_f1, ",", f1_scores[max_f1])

best_pred = model_preds[max_f1]
best_confusion_matrix = confusion_matrix(step_test, best_pred)

CM = ConfusionMatrixDisplay(best_confusion_matrix)
CM.plot(cmap=plt.cm.Blues)
plt.title(max_f1.replace('_pred', '') + " Confusion Matrix")
plt.show()


# Part 6: Stacked Model Performance Analysis
combined_model = [('SVM', best_model_SVM), ('RandomForrest', best_model_RF)]

final_model = LogisticRegression(max_iter = 200, class_weight='balanced')

StackedModel = StackingClassifier(combined_model, final_model, cv = 5)
StackedModel.fit(coord_train, step_train)
SM_pred = StackedModel.predict(coord_test)

StackedModel_accuracy_score = accuracy_score(step_test, SM_pred)
StackedModel_f1_score = f1_score(step_test, SM_pred, average='weighted')
StackedModel_precision_score = precision_score(step_test, SM_pred, average='weighted')
StackedModel_confusion_matrix = confusion_matrix(step_test, SM_pred)
StackedModel_classification_report = classification_report(step_test, SM_pred)

print("\n\nStacked Model Performance Analysis")
print("\nAccuracy Score:", StackedModel_accuracy_score)
print("\nF1 Score:", StackedModel_f1_score)
print("\nPrecision Score:", StackedModel_precision_score)
print("\nClassification Report:\n", StackedModel_classification_report)

CM_SM = ConfusionMatrixDisplay(StackedModel_confusion_matrix)
CM_SM.plot(cmap=plt.cm.Greens)
plt.title("StackingClassifier Confusion Matrix")
plt.show()

    
# Part 7: Model Evaluation
joblib.dump(best_model_SVM, 'SVM_model.joblib')
loaded_SVM_model = joblib.load('SVM_model.joblib')

coord_predict = np.array([[9.375, 3.0625, 1.51],
                          [6.995, 5.125, 0.3875],
                          [0, 3.0625, 1.93],
                          [9.4, 3, 1.8],
                          [9.4, 3, 1.3]])

prediction_df = pd.DataFrame(coord_predict, columns = coord_train.columns)
scaled_data = my_scaler.transform(prediction_df)
scaled_df = pd.DataFrame(scaled_data, columns = coord_train.columns)
class_pred = loaded_SVM_model.predict(scaled_df)
print("Predicted Maintenance Steps:", class_pred)


