import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

invest_value = 75 # Valor estimado para investir. (em Mil)

# Importando a base de dados
base_data = pd.read_excel("src/Investimento_x_Venda.xlsx")
print(base_data.head())

# Usando Machine Learning para descobrir a venda estimada de acordo com investimento em marketing.
reg = linear_model.LinearRegression()
reg.fit(base_data["Investimento em marketing"].values.reshape(-1, 1), base_data["Venda Qtd"])

print(reg.coef_)
print(reg.intercept_)

plt.scatter(base_data["Investimento em marketing"],base_data["Venda Qtd"])
plt.scatter(75,reg.predict([[75]])[0],color="k")
x = np.array(base_data["Investimento em marketing"])
y = reg.intercept_ + x * reg.coef_
plt.plot(x, y, "r")
plt.show()

result_estimated_sale = reg.predict([[invest_value]])

print("Investindo ",invest_value," Mil em marketing, a estimativa é de: ",round(result_estimated_sale[0])," vendas")


