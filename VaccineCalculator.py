'''looked at some sample code from https://towardsdatascience.com/linear-regression-in-python-sklearn-vs-excel-6790187dc9ca'''

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('VacineData.xls',index_col=0)
#'df = pd.DataFrame({"sentences": splitted, "count": orr})'
##df.to_csv("raven.txt", index=False)



Training_data=pd.read_excel("VacineData.xls", sheet_name="Sheet2")
Test_data=pd.read_excel("VacineData.xls", sheet_name="Sheet3")

print(Training_data)
print(Test_data)
print(Training_data.columns)
##SourceData_train_dependent=Training_data.columns[[1, 3]].copy() #  New dataframe with only independent variable value for training dataset
SourceData_train_dependent=Training_data[['total']].copy()
SourceData_train_independent= Training_data.drop(Training_data.columns[[0, 1, 3]],axis = 1) # Drop depedent variable from training dataset
print(Training_data.columns)
'''SourceData_train_dependent=Training_data.columns[[1, 3]].copy() #  New dataframe with only independent variable value for training dataset'''
#SourceData_test_dependent=Test_data.columns[[1, 3]].copy()
SourceData_test_dependent=Test_data[['total']].copy()
SourceData_test_independent=Test_data.drop(Test_data.columns[[0, 1, 3]],axis = 1)

##['total','added today']
#print("Line 31")
print(SourceData_train_independent)

sc_X = StandardScaler()## this probably isnt necesary
X_train=sc_X.fit_transform(SourceData_train_independent.values) #scale the independent variables
y_train=SourceData_train_dependent # scaling is not required for dependent variable
X_test=sc_X.transform(SourceData_test_independent)
y_test=SourceData_test_dependent
'''
X_train = SourceData_train_independent.values
y_train=SourceData_train_dependent # scaling is not required for dependent variable

X_test=SourceData_test_independent
y_test=SourceData_test_dependent'''
print("x")
print(X_train)

print("y")
print(y_train)


reg = LogisticRegression().fit(X_train, y_train)
print("The Logistic regression score on training data is ", round(reg.score(X_train, y_train),2))


predict=reg.predict(X_test)

print(predict)

reg = LinearRegression().fit(X_train, y_train)
print("The linear regression score on training data is ", round(reg.score(X_train, y_train),2))


predict=reg.predict(X_test)

print(y_test)
print(predict)

print("The linear regression score on testing data is ", r2_score(y_test, predict))




##print(predict)

futureprections =SourceData_test_independent.copy()
print(futureprections)
print(futureprections['days since firstvaccine'].iloc[-1]+1)
print(predict[-1])
print((predict[-1] - futureprections["Yesterday's total"].iloc[-7])/7)#futureprections["Yesterday's total"].iloc[-8] replaces the number
print(predict[-7])
currPredictFrame = pd.DataFrame([[futureprections['days since firstvaccine'].iloc[-1]+1, predict[-1], (predict[-1] - futureprections["Yesterday's total"].iloc[-7])/7, futureprections["Yesterday's total"].iloc[-6] -futureprections["Yesterday's total"].iloc[-7]]], columns=futureprections.columns)

x_predict = sc_X.transform(currPredictFrame)
print(currPredictFrame)

print(x_predict)

newPredict = reg.predict(x_predict)
print(newPredict)
predict = np.append(predict,newPredict)
futureprections = futureprections.append(currPredictFrame)
print(futureprections)
print(predict)

while newPredict < 789000:
    currPredictFrame = pd.DataFrame([[futureprections['days since firstvaccine'].iloc[-1]+1, predict[-1], (predict[-1] - futureprections["Yesterday's total"].iloc[-7])/7, futureprections["Yesterday's total"].iloc[-6] -futureprections["Yesterday's total"].iloc[-7]]], columns=futureprections.columns)

    x_predict = sc_X.transform(currPredictFrame)


    newPredict = reg.predict(x_predict)
    print(newPredict)
    predict = np.append(predict,newPredict)
    futureprections = futureprections.append(currPredictFrame)

print(futureprections['days since firstvaccine'].iloc[-1]+1)

#print(currPredictFrame)


##def futurePredict(model, data):

'''

loop


'''

## time series predictions
