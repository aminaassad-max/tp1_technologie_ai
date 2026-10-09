import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



data = pd.read_csv('train.csv') 
print(data.head()) 
print(data.info()) 

print("Valeurs manquantes avant nettoyage :")
print(data.isnull().sum())

data = data[['Survived',
             'Pclass',
             'Sex',
             'Age',
             'SibSp',
             'Parch',
             'Fare',
             'Embarked']]


data['Age'] = data['Age'].fillna(data['Age'].median())


data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

data['Sex'] = data['Sex'].map({
    'male': 0,
    'female': 1
})

data['Embarked'] = data['Embarked'].map({
    'S': 0,
    'C': 1,
    'Q': 2
})



print("\nValeurs manquantes après nettoyage :")
print(data.isnull().sum())
print("\nAperçu des données nettoyées :")
print(data.head())
print("\nInformations sur les données :")
print(data.info())
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])
print(data.head())
print(data.info())
X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,              
    y,              
    test_size=0.2,  
    random_state=42 
)

print("Taille de X_train :", X_train.shape)
print("Taille de X_test :", X_test.shape)

print("Taille de y_train :", y_train.shape)
print("Taille de y_test :", y_test.shape)

log_model = LogisticRegression(
    max_iter=1000
)



rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
log_model.fit(
    X_train,
    y_train
)

rf_model.fit(
    X_train,
    y_train
)


print("Les deux modèles sont entraînés avec succès ")

y_pred_log = log_model.predict(X_test)


y_pred_rf = rf_model.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)

accuracy_rf = accuracy_score(y_test, y_pred_rf)


print("Accuracy Logistic Regression :", accuracy_log)

print("Accuracy Random Forest :", accuracy_rf)


print("\n===== Rapport Logistic Regression =====")
print(classification_report(y_test, y_pred_log))


print("\n===== Rapport Random Forest =====")
print(classification_report(y_test, y_pred_rf))

print("\nMatrice de confusion Random Forest :")
print(confusion_matrix(y_test, y_pred_rf))
cm = confusion_matrix(
    y_test,
    y_pred_rf
)


plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,      
    fmt='d',         
    cmap='Blues'    
)



plt.title('Matrice de confusion - Titanic (Random Forest)')

plt.xlabel('Prédit')

plt.ylabel('Réel')



plt.show()
