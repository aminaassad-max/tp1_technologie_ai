import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("diabetes.csv")

print(data.head())

print(data.info())

print(data.isnull().sum())
print(data.describe())


columns_zero = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]



data[columns_zero] = data[columns_zero].replace(
    0,
    np.nan
)




for col in columns_zero:
    data[col] = data[col].fillna(
        data[col].median()
    )



print(data.isnull().sum())

X = data.drop(
    'Outcome',
    axis=1
)



y = data['Outcome']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("X_train :", X_train.shape)

print("X_test :", X_test.shape)
log_model = LogisticRegression(
    max_iter=1000
)


log_model.fit(
    X_train,
    y_train
)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


rf_model.fit(
    X_train,
    y_train
)


print("Modèles entraînés avec succès")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


y_pred_log = log_model.predict(
    X_test
)


y_pred_rf = rf_model.predict(
    X_test
)

accuracy_log = accuracy_score(
    y_test,
    y_pred_log
)


accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)


print(
    "Accuracy Logistic Regression :",
    accuracy_log
)


print(
    "Accuracy Random Forest :",
    accuracy_rf
)

print("\n===== Logistic Regression =====")

print(
    classification_report(
        y_test,
        y_pred_log
    )
)



print("\n===== Random Forest =====")

print(
    classification_report(
        y_test,
        y_pred_rf
    )
)

cm = confusion_matrix(
    y_test,
    y_pred_rf
)


plt.figure(figsize=(5,4))


sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)


plt.title(
    "Matrice de confusion - Diabetes Random Forest"
)


plt.xlabel(
    "Prédit"
)


plt.ylabel(
    "Réel"
)


plt.show()
