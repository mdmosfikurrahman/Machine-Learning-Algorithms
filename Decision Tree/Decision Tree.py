# Import Packages and Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

# Read Dataset
df = pd.read_csv("../Sample Datasets/ChurnData.csv", na_values="?")

# Deal with Null Values
df.replace("?", np.NaN)
df.fillna(round(df.mean(),2), inplace=True)

# Data Type Casting
df['churn'] = df['churn'].astype('int')
df['confer'] = df['confer'].astype('float')

# Define Independent and Dependent Variables for Implement Models
X = np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon', 'longten', 'tollten', 'cardten', 'voice', 'pager', 'internet', 'callwait', 'confer', 'ebill', 'loglong', 'logtoll', 'lninc', 'custcat']])
y = np.asarray(df['churn'])

# Normalize Dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Spliting Train & Test Dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=4)

# Apply Machine Learning Algorithm
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred_dt = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test,y_test) * 100, 2)
print("Decision Tree Accuracy: ", acc_decision_tree)

# Cross Validated Score
cvs_dt = round((cross_val_score(decision_tree, X,y,cv=10,scoring='accuracy')).mean()*100,2)
print('Cross Validated Score:', cvs_dt)

# Classification Report
class_report = classification_report(y_test, Y_pred_dt)
print("Classificiation Report: \n", class_report)

# Confusion Matrix
cnf_matrix = confusion_matrix(y_test, Y_pred_dt, labels=[1,0])
sns.heatmap(cnf_matrix,annot=True,fmt='3.0f',cmap="Greens")
plt.title('Confusion Matrix (Decision Tree)', y=1.05, size=15)

# Jaccard Score
jac_score_dt = round(jaccard_score(y_test, Y_pred_dt, pos_label=0) * 100, 2)
print("Jaccard Score: ", jac_score_dt)

# Mean Absolute Error & Mean Squared Error
dt_mae = round((mean_absolute_error(y_test, Y_pred_dt)*100), 2)
dt_mse = round((mean_squared_error(y_test, Y_pred_dt)*100), 2)
print("Mean Absolute Error: ", dt_mae)
print("Mean Squared Error: ", dt_mse)

# ROC AUC Score
auc_dt = roc_auc_score(y_test, Y_pred_dt)
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, Y_pred_dt)
plt.figure(figsize=(12, 7))
plt.plot(fpr_dt, tpr_dt, label=f'AUC (Decision Tree) = {auc_dt:.2f}')
plt.title('ROC AUC Score')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.legend()