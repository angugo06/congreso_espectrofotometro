import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

new_line = '\n'

# importar datos del espectrofotómetro casero
dataset = pd.read_excel('tetvol.xlsx')
x = dataset.iloc[:, :-1].values.reshape(-1, 1)
y = dataset.iloc[:, -1].values.reshape(-1, 1)

# importar datos del espectrofotómetro comercial
dataset1 = pd.read_excel('tetabs.xlsx')
x1 = dataset1.iloc[:, :-1].values.reshape(-1, 1)
y1 = dataset1.iloc[:, -1].values.reshape(-1, 1)

# crear modelo de regresión linear
# para el espectrofotómetro casero
regression = LinearRegression()
regression.fit(x, y)
coef = round(regression.coef_[0][0], 3)
intercept = round(regression.intercept_[0], 3)
regresssion_equation = f"y={coef}x+{intercept}"

# Crear métrica de error cuadrado medio y r cuadrada
# para el espectrofotómetro casero
rmse = round(mean_squared_error(y, regression.predict(x), squared=False), 3)
r2 = round(r2_score(y, regression.predict(x)), 3)
plot_text = f"RMSE = {rmse}{new_line}r\u00B2 = {r2}{new_line}{regresssion_equation}"

# crear modelo de regresión linear
# para el espectrofotómetro comercial
regression1 = LinearRegression()
regression1.fit(x1, y1)
coef1 = round(regression1.coef_[0][0], 3)
intercept1 = round(regression1.intercept_[0], 3)
regresssion_equation1 = f"y={coef1}x+{intercept1}"

# Crear métrica de error cuadrado medio y r cuadrada
# para el espectrofotómetro comercial
rmse1 = round(mean_squared_error(y1, regression1.predict(x1), squared=False), 3)
r21 = round(r2_score(y1, regression1.predict(x1)), 3)
plot_text1 = f"RMSE = {rmse1}{new_line}r\u00B2 = {r21}{new_line}{regresssion_equation1}"

# crear modelo de regresión linear
# para la comparación de espectrofotómetros
regression2 = LinearRegression()
regression2.fit(y1, y)
coef2 = round(regression2.coef_[0][0], 3)
intercept2 = round(regression2.intercept_[0], 3)
regresssion_equation2 = f"y={coef2}x+{intercept2}"

# Crear métrica de error cuadrado medio y r cuadrada
# para la comparación de espectrofotómetros

rmse2 = round(mean_squared_error(y, regression2.predict(y1), squared=False), 3)
r22 = round(r2_score(y, regression2.predict(y1)), 3)
plot_text2 = f"RMSE = {rmse2}{new_line}r\u00B2 = {r22}{new_line}{regresssion_equation2}"

# visualizar regresión del espectrofotómetro casero
plt.scatter(x, y, color='red')
plt.plot(x, regression.predict(x), color='blue')
plt.title("Respuesta de espectrofotómetro casero", fontsize=18)
plt.xlabel('Concentración de [Cu(NH\u2083)\u2084]SO\u2084·H\u2082O', fontsize=16)
plt.ylabel('Ln((R\u1D62·V\u209B)/(V\u209B-V\u2091))', fontsize=16)
plt.text(0, 6.8, plot_text, fontsize=12)
plt.xticks(np.arange(min(x), max(x), 0.01))
plt.yticks(np.arange(min(y)-0.02, max(y), 0.3))
plt.savefig('Casero')
plt.show()

# visualizar regresión del espectrofotómetro comercial
plt.scatter(x1, y1, color='red')
plt.plot(x1, regression1.predict(x1), color='blue')
plt.title("Respuesta de espectrofotómetro comercial", fontsize=18)
plt.xlabel('Concentración de [Cu(NH\u2083)\u2084]SO\u2084·H\u2082O', fontsize=16)
plt.ylabel('Absorbancia', fontsize=16)
plt.text(0.0, 1.3, plot_text1, fontsize=12)
plt.xticks(np.arange(min(x), max(x), 0.01))
plt.yticks(np.arange(min(y1), max(y1), 0.3))
plt.savefig('Comercial')
plt.show()

# visualizar regresión de la comparación de espectromotómetros
plt.scatter(y1, y, color='red')
plt.plot(y1, regression2.predict(y1), color='blue')
plt.title("Comparación de espectrofotómetros", fontsize=18)
plt.xlabel('Absorbancia', fontsize=16)
plt.ylabel('Ln((R\u1D62·V\u209B)/(V\u209B-V\u2091))', fontsize=16)
plt.text(0.0, 6.8, plot_text2, fontsize=12)
plt.yticks(np.arange(min(y)-0.02, max(y), 0.3))
plt.savefig('Comparacion')
plt.show()
