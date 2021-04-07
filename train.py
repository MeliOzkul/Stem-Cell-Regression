import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def Specificity(yTest,yPred):
    tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
    specificity = tn / (tn+fp)
    return specificity

def Sensitivity(yTest,yPred):
    tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
    sensitivity = tp / (tp+fn)
    return sensitivity

data=pd.read_csv('dataset.csv')
print(data.shape)

X=data.iloc[0:,2:]
y=data.iloc[0:,1]

X=np.array(X).astype('float64')
y=np.array(y).astype('uint8')

print(X.shape)
print(y.shape)

target_names=['no stem cell','stem cell']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))

print("Logistic Regression score: {0}".format(logisticRegr.score(X_test,y_test)))
print("Specificity:{0}".format(Specificity(y_test,y_pred)))
print("Sensitivity:{0}".format(Sensitivity(y_test,y_pred)))

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
plt.figure(figsize=(9, 4))
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g', xticklabels=target_names, yticklabels=target_names)
plt.title("Result")
plt.show()
