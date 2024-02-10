import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("homeprices.csv")

# plotting the graph on the given data frame
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='*')
plt.show()

# storing the area and price
area= df.drop('price',axis='columns')
price = df['price']

# Creating a linear regression model and training it on data
reg = linear_model.LinearRegression()
reg.fit(area,price)

# plotting the graph of the predicted data
plt.xlabel('area', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.scatter(area,price,color='red',marker='*')
plt.plot(area, reg.predict(area), color='blue',)
plt.show()

# printing the prediction on a given area
predicted_price = reg.predict([[3300]])
print("predicted_price: ",predicted_price,"\n")


# How it actually predicts?
# Procedure:
# coef = reg.coef_
# print("coef: ",coef)

# intercept = reg.intercept_
# print("intercept: ",intercept)

# y = m*x + c
# y= coef*3300+intercept
# print("y: ",y)

#1 Making a dataframe 
#2 predicting it
#3  then adding the predicted values to the data frame
#4  and then concatinating it into a file corresponding to the area
# 1
area_df = pd.read_csv("areas.csv")

# 2
p = reg.predict(area_df)
# print("p: ",p)

# 3
area_df["prices"] = p

# 4
area_df.to_csv("prediction.csv", index=False)
