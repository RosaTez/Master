# -*- coding: utf-8 -*-

# Librerías necesarias.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#%%
# Ruta dónde están guardados los datos
# ruta = "C:\\Users\Rosa_\OneDrive\Escritorio\HouseRentPrediction\House_Rent_Dataset.csv"
ruta = "https://github.com/RosaTez/Master/blob/main/House_Rent_Dataset.csv"
# Se cargan los datos
data_c = pd.read_csv(ruta, sep=',')

# Se crea un DataFrame unicamente con los atributos que se utilizarán
data = data_c.loc[:, ['BHK', 'Rent', 'Size', 'Area Type', 'Furnishing Status', 
                      'Tenant Preferred', 'Bathroom', 'Point of Contact']]

# Ver si hay algún dato faltante
data.isnull().sum()

# Se ven los 5 primeros datos
data.head(5)

# Se visualizan las variables categóricas
sns.countplot(x=data["BHK"])
plt.title('Gráfico de Barras de la Variable BHK')
plt.xlabel('Categoría')
plt.ylabel('Frecuencia')
plt.show()

sns.countplot(x=data["Area Type"])
plt.title('Gráfico de Barras de la Variable Area Type')
plt.xlabel('Categoría')
plt.ylabel('Frecuencia')
plt.show()

sns.countplot(x=data["Furnishing Status"])
plt.title('Gráfico de Barras de la Variable Furnishing Status')
plt.xlabel('Categoría')
plt.ylabel('Frecuencia')
plt.show()

sns.countplot(x=data["Tenant Preferred"])
plt.title('Gráfico de Barras de la Variable Tenant Preferred')
plt.xlabel('Categoría')
plt.ylabel('Frecuencia')
plt.show()

sns.countplot(x=data["Bathroom"])
plt.title('Gráfico de Barras de la Variable Bathroom')
plt.xlabel('Categoría')
plt.ylabel('Frecuencia')
plt.show()

sns.countplot(x=data["Point of Contact"])
plt.title('Gráfico de Barras de la Variable Point of Contact')
plt.xlabel('Categoría')
plt.ylabel('Frecuencia')
plt.show()

# Se describen las variables numéricas
data["Size"].describe()

data["Rent"].describe()


# Se convierten las variables categóricas en variables numéricas
encoder = LabelEncoder()
for column in data.select_dtypes(include = object).columns.tolist():
    data[column] = encoder.fit_transform(data[column])
    
# La variable Rent que queremos predecir tiene un amplio rango de valores
data['Rent'].max()
data['Rent'].min()
# Se observa que el máximo es 3500000 y el mínimo es 1200 por lo tanto se normalizará
data["Rent"] = np.log10(data["Rent"])

# Se separan las variables. La variable dependiente es la que queremos predecir
# y las variables independientes son los atributos.
var_dep = data["Rent"]
var_indep = data.drop(columns = ["Rent"])


# Se divide el conjunto de datos en dos. Uno de entrenamiento con el 80% de los
# datos y otro de test con el 20% restante. Además se establece una semilla para
# que los resultados sean siempre los mismos.
x_train, x_test, y_train, y_test = train_test_split(var_indep, var_dep, 
                                                    test_size = 0.2,
                                                    random_state = 42)

# Se muestra el tamaño de cada conjunto de datos
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# Se estandariza las características de los conjuntos de entrenamiento y test.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#%%
# Modelo de Random-Forest
modeloRF = RandomForestRegressor()

# Entrenar el modelo con los datos de entrenamiento
modeloRF.fit(x_train, y_train)

# Realizar predicciones en el conjunto de prueba
predict = modeloRF.predict(x_test)

plt.scatter(y_test, predict)
plt.title("Gráfico de Dispersión (Random-Forest)")
plt.xlabel("Valores reales (y_test)")
plt.ylabel("Predicciones del modelo")
plt.show()

r2 = r2_score(y_test, predict)
mse = mean_absolute_error(y_test, predict)

print(f'R-squared (R²): {r2}')
print(f'Mean Squared Error (MSE): {mse}')

#%%
# Modelo de Regresion Lineal Multiple
modelo_regresion = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo_regresion.fit(x_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo_regresion.predict(x_test)

plt.scatter(y_test, y_pred)
plt.title("Gráfico de Dispersión (Regresion Lineal Multiple)")
plt.xlabel("Valores reales (y_test)")
plt.ylabel("Predicciones del modelo")
plt.show()

# Evaluar el rendimiento del modelo
r2 = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)

print(f'R-squared (R²): {r2}')
print(f'Mean Squared Error (MSE): {mse}')


#%%
# Red Neuronal
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(x_test, y_test)
print(f'Error cuadrático medio en los datos de prueba: {loss}')

# Hacer predicciones
predictions = model.predict(x_test)
plt.scatter(y_test, predictions)
plt.title("Gráfico de Dispersión (Red Neuronal)")
plt.xlabel("Valores reales (y_test)")
plt.ylabel("Predicciones del modelo")
plt.show()


























