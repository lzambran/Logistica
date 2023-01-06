

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
# %matplotlib inline

dataframe = pd.read_csv(r"logistica.csv")
dataframe.head()

dataframe.describe()

print(dataframe.groupby('anime').size())

dataframe.drop(['anime'], axis = 1).hist()
plt.show()

x = np.array(dataframe.drop(['anime'], axis = 1))
y = np.array(dataframe['anime'])
x.shape

model = LogisticRegression(solver = 'liblinear', max_iter = 100)
model.fit(x, y)

LogisticRegression(
    C = 1.0, 
    class_weight = None, 
    dual = False, 
    fit_intercept = True, 
    intercept_scaling = 1, 
    max_iter = 100, 
    multi_class = 'ovr', 
    n_jobs = 1, 
    penalty = 'l2', 
    random_state = None, 
    solver = 'liblinear', 
    tol = 0.0001, 
    verbose = 0, 
    warm_start = False
)

predictions = model.predict(x)
print(predictions[0:5])

model.score(x, y)

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)

name = 'Regresion Logistica'
kfold = model_selection.KFold(n_splits = 10, random_state = seed, shuffle = True)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = 'accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))

X_new = pd.DataFrame({'duracion': [5], 'numero': [0], 'acciones': [10], 'valor': [10]})
model.predict(X_new)