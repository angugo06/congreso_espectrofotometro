import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

new_line = '\n'

# importing dataset
dataset = pd.read_excel('../datasets/tetvol.xlsx')
x = dataset.iloc[:, :-1].values.reshape(-1, 1)
y = dataset.iloc[:, -1].values.reshape(-1, 1)
dataset1 = pd.read_excel('../datasets/tetabs.xlsx')
x1 = dataset1.iloc[:, :-1].values.reshape(-1, 1)
y1 = dataset1.iloc[:, -1].values.reshape(-1, 1)


# linear regression
regression = LinearRegression()
regression.fit(x, y)
rmse = mean_squared_error(x, y, squared=False)
r2 = r2_score(x,y)
plot_text = f"RMSE = {rmse}{new_line}r\u00B2 = {r2}"

regression1 = LinearRegression()
regression1.fit(x1, y1)
rmse1 = mean_squared_error(x1, y1, squared=False)
r21 = r2_score(x1, y1)
plot_text1 = f"RMSE = {rmse1}{new_line}r\u00B2 = {r21}"

regression2 = LinearRegression()
regression2.fit(y1, y)
rmse2 = mean_squared_error(y1, y)
r22 = r2_score(y1, y)
plot_text2 = f"RMSE = {rmse2}{new_line}r\u00B2 = {r22}"

# visualize regression
plt.scatter(x, y, color='red')
plt.plot(x, regression.predict(x), color='blue')
plt.title("Respuestra de espectrofotómetro casero")
plt.xlabel('Concentración de [Cu(NH\u2083)\u2084]SO\u2084·H\u2082O')
plt.ylabel('Ln(R\u1D62·V\u209B/V\u209B-V\u2091)')
plt.text(0, 0.7, plot_text)
plt.show()

plt.scatter(x1, y1, color='red')
plt.plot(x1, regression1.predict(x1), color='blue')
plt.title("Respuestra de espectrofotómetro comercial")
plt.xlabel('Concentración de [Cu(NH\u2083)\u2084]SO\u2084·H\u2082O')
plt.ylabel('Absorbancia')
plt.text(0.0, 1.3, plot_text1)
plt.show()

plt.scatter(y1, y, color='red')
plt.plot(y1, regression2.predict(y1), color='blue')
plt.title("Comparación de espectrofotómetros")
plt.xlabel('Absorbancia')
plt.ylabel('Ln(R\u1D62·V\u209B/V\u209B-V\u2091)')
plt.text(0.0, 0.7, plot_text2)
plt.show()
