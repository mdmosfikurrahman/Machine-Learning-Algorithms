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
from sklearn.naive_bayes import GaussianNB
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
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred_gnb = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_test,y_test) * 100, 2)
print("Gaussian Naive Bayes Accuracy: ", acc_gaussian)

# Cross Validated Score
cvs_gnb = round((cross_val_score(gaussian, X,y,cv=10,scoring='accuracy')).mean()*100,2)
print('Cross Validated Score:', cvs_gnb)

# Classification Report
class_report = classification_report(y_test, Y_pred_gnb)
print("Classificiation Report: \n", class_report)

# Confusion Matrix
cnf_matrix = confusion_matrix(y_test, Y_pred_gnb, labels=[1,0])
sns.heatmap(cnf_matrix,annot=True,fmt='3.0f',cmap="Greens")
plt.title('Confusion Matrix (Gaussian Naive Bayes)', y=1.05, size=15)

# Jaccard Score
jac_score_gaussian = round(jaccard_score(y_test, Y_pred_gnb, pos_label=0) * 100, 2)
print("Jaccard Score: ", jac_score_gaussian)

# Mean Absolute Error & Mean Squard Error
gnb_mae = round((mean_absolute_error(y_test, Y_pred_gnb)*100), 2)
gnb_mse = round((mean_squared_error(y_test, Y_pred_gnb)*100), 2)
print("Mean Absolute Error: ", gnb_mae)
print("Mean Squared Error: ", gnb_mse)

# ROC AUC Score
auc_gnb = roc_auc_score(y_test, Y_pred_gnb)
fpr_gnb, tpr_gnb, thresholds_gnb = roc_curve(y_test, Y_pred_gnb)
plt.figure(figsize=(12, 7))
plt.plot(fpr_gnb, tpr_gnb, label=f'AUC (Gaussian Naive Bayes) = {auc_gnb:.2f}')
plt.title('ROC AUC Score')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.legend()