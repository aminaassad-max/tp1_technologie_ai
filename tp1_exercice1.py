import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor



data = pd.read_csv('uber.csv') 
print(data.head()) 
print(data.info()) 
print(data.isnull().sum())
data = data.dropna() 
data = data[data['fare_amount'] > 0] 
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime']) 
data['hour'] = data['pickup_datetime'].dt.hour 
data['day_of_week'] = data['pickup_datetime'].dt.dayofweek

def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))

    return R * c

data["distance"] = haversine(
    data["pickup_latitude"],
    data["pickup_longitude"],
    data["dropoff_latitude"],
    data["dropoff_longitude"]
)
X = data[['distance', 'hour', 'day_of_week']]
y = data['fare_amount']
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
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Quelques prédictions :")
print(y_pred[:10])
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("========== Évaluation du modèle ==========")
print("MAE  =", mae)
print("MSE  =", mse)
print("RMSE =", rmse)
print("R²   =", r2)
rf = RandomForestRegressor(
    n_estimators=100,      
    random_state=42        
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
plt.scatter(y_test, y_pred_rf, alpha=0.5) 
plt.xlabel('Valeurs réelles') 
plt.ylabel('Valeurs prédites') 
plt.title('Prédiction du tarif Uber') 
plt.show()
