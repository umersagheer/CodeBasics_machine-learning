import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


df = pd.read_csv("canada_per_capita_income.csv")

year = df.drop('per_capita_income_(US$)', axis='columns')
income = df['per_capita_income_(US$)']


reg = linear_model.LinearRegression()
reg.fit(year, income)

ansPrediction = reg.predict([[2022]])
print(ansPrediction)

prediction = reg.predict(year)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Income', fontsize=20)
plt.scatter(year,income,color='red',marker='*')
plt.plot(year, prediction, color='blue',)
plt.show()