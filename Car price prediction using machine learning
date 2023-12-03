import warnings

warnings.filterwarnings('ignore')

import pandas as pd

data=pd.read_csv('/car details.csv ')

data.head()

data.tail()

print("Number of rows",data.shape[0])

print("Number of columns",data.shape[1])

data.info()

data.isnull().sum()

data.describe()

data.head(1)

import datetime

date_time=datetime.datetime.now()

Age=date_time

data.drop('Year',axis=1,inplace=True)

import seaborn as sns

sns.barplot(data[' '])

sorted(data['Selling_Price'],reverse=True)

data[{data['SElling_Price']>=33.0) & (data['Selling_Price<=35.0)]

data.shape

data.head(!)

data['Fuel_Type'].unique()

data['Fuel_Type']=data['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})
data['Fuel_Type'].unique()
data['Seller_Type'].unique()
data['Seller_Type']=data['Seller_Type'].map({'Dealer':0,'Induvidual':1})
data['Transmission].unique()
data['Transmission']=data['Transmission'].map({'Manual':0,'Automatic:1})
data['Transmission].unique()
data.head()
x=data.drop(['Car_name','Selling_Price'],axis=1)
y=data['Selling_Price']
x
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
data.head()
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomforestRegressor
from sklearn.ensemble import GradientBoostRegressor
from xgboost import XGBRegressor
lr=LinearRegression()
lr.fit(x_train,y_train)
rf=RandomforestRegressor()
rf.fit(x_train,y_train)
xgb=GradientBoostingRegressor()
xgb.fit(X_train,y_train)
xg=XGBRegressor()
xg.fit(X_train,y_train) 
y_pred1=lr.predict(X_test)
y_pred2=rf.predict(X_test)
y_pred3=gbr.predict(X_test)
y_pred4=xg.predict(X_test)
from sklearn import metrics
score1=metrics.r2_score(y_test,y_pred1)
score2=metrics.r2_score(y_test,y_pred2)
score3=metrics.r2_score(y_test,y_pred3)
score4=metrics.r2_score(y_test,y_pred4)
print(score1,score2,score3,score4)
final_data=pd.DataFrame({'Models':['LR','RF','GBR','XG'],"R2_SCORE":[score1,score2,score3,s
core4]})
final_data
sns.barplot(final_data['Models'],final_data['R2_SCORE'])
xg=XGBRegressor()
xg_final=xg.fit(x,y)
import joblib
joblib.dump(xg_final,'car_price_predictor')
model=joblib.load('car_price_predictor')
//Prediction of new data

import pandas as pd

data_new=pd.DataFrame({ 'Present_Price':5.59,'kms_driven':27000,

 'Fuel_Type':0,

 'Seller_Type':0;

 'Transmission':0,

 'Owner':0,

 'Age':8

},index=[0])

model.predict(data_new)
