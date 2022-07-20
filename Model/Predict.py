import pandas as pd
data_spring = pd.read_csv('spring.csv',encoding='CP949')
#data_spring.head()
data_summer = pd.read_csv('summer.csv',encoding='CP949')
#data_summer.head()
data_fall = pd.read_csv('fall.csv',encoding='CP949')
#data_fall.head()
data_winter = pd.read_csv('winter.csv',encoding='CP949')
#data_winter.head()
data_spring_test = pd.read_csv('spring_test.csv',encoding='CP949')
#data_spring_test.head()

df_spring = pd.DataFrame(data_spring)
df_summer = pd.DataFrame(data_summer)
df_fall = pd.DataFrame(data_fall)
df_winter = pd.DataFrame(data_winter)
df_spring_test = pd.DataFrame(data_spring_test)

df_spring = df_spring.fillna(0)
df_summer = df_summer.fillna(0)
df_fall = df_fall.fillna(0)
df_winter = df_winter.fillna(0)
df_spring_test = df_spring_test.fillna(0)

df_sp=df_spring.drop(['지점','일시'],axis=1)
df_su=df_summer.drop(['지점','일시'],axis=1)
df_fa=df_fall.drop(['지점','일시'],axis=1)
df_wi=df_winter.drop(['지점','일시'],axis=1)
df_spring_test=df_spring_test.drop(['지점','일시'],axis=1)
train_spring = df_sp.drop('지점명', axis=1)
train_summer = df_su.drop('지점명', axis=1)
train_fall = df_fa.drop('지점명', axis=1)
train_winter = df_wi.drop('지점명', axis=1)
test_spring = df_spring_test.drop('지점명',axis=1)

y_label_spring=[]
for i in range(len(train_spring)):
  if train_spring['일강수량(mm)'][i]>0:
    y_label_spring.append(0.0)
  else:
    y_label_spring.append(10.0)

for i in range(len(train_spring)):
  if y_label_spring[i]==10.0:
    if train_spring['평균기온(°C)'][i]<13:
      y_label_spring[i]-=3
    elif train_spring['평균기온(°C)'][i]<17:
      y_label_spring[i]-=2
    elif train_spring['평균기온(°C)'][i]<20:
      y_label_spring[i]-=1
    if train_spring['평균 상대습도(%)'][i]>70:
      y_label_spring[i]-=1
    if train_spring['평균 풍속(m/s)'][i]>1.2:
      y_label_spring[i]-=1 
    if train_spring['평균 풍속(m/s)'][i]>9:
      y_label_spring[i]-=3 

y_label_summer=[]
for i in range(len(train_summer)):
  if train_summer['일강수량(mm)'][i]>0:
    y_label_summer.append(0.0)
  else:
    y_label_summer.append(10.0)

for i in range(len(train_summer)):
  if train_summer['평균기온(°C)'][i]>9:
    print(train_summer['평균 풍속(m/s)'][i])

for i in range(len(train_summer)):
  if y_label_summer[i]==10.0:
    if train_summer['평균기온(°C)'][i]>25:
      y_label_summer[i]-=1
    if train_summer['평균기온(°C)'][i]>28:
      y_label_summer[i]-=1
    if train_summer['평균기온(°C)'][i]>30:
      y_label_summer[i]-=2
    if train_summer['평균 상대습도(%)'][i]>65:
      y_label_summer[i]-=1
    if train_summer['평균 상대습도(%)'][i]>75:
      y_label_summer[i]-=1
    if train_summer['평균 풍속(m/s)'][i]>2:
      y_label_summer[i]-=1 
    if train_summer['평균 풍속(m/s)'][i]>9:
      y_label_summer[i]-=2

y_label_fall=[]
for i in range(len(train_fall)):
  if train_fall['일강수량(mm)'][i]>0:
    y_label_fall.append(0.0)
  else:
    y_label_fall.append(10.0)

for i in range(len(train_fall)):
  if y_label_fall[i]==10.0:
    if train_fall['평균기온(°C)'][i]<15:
      y_label_fall[i]-=1
    if train_fall['평균기온(°C)'][i]<10:
      y_label_fall[i]-=1
    if train_fall['평균기온(°C)'][i]>25:
      y_label_fall[i]-=1
    if train_fall['평균 상대습도(%)'][i]>59:
      y_label_fall[i]-=1
    if train_fall['평균 풍속(m/s)'][i]>1.2:
      y_label_fall[i]-=1 
    if train_fall['평균 풍속(m/s)'][i]>9:
      y_label_fall[i]-=2

y_label_winter=[]
for i in range(len(train_winter)):
  if train_winter['일강수량(mm)'][i]>0 or train_winter['일 최심적설(cm)'][i]>10:
    y_label_winter.append(0.0)
  else:
    y_label_winter.append(10.0)

for i in range(len(train_winter)):
  if y_label_winter[i]==10.0:
    if train_winter['평균기온(°C)'][i]<1.5:
      y_label_winter[i]-=2
    if train_winter['평균 상대습도(%)'][i]<59:
      y_label_winter[i]-=1
    if train_winter['평균 풍속(m/s)'][i]>2.3:
      y_label_winter[i]-=1 
    if train_winter['평균 풍속(m/s)'][i]>2.3:
      y_label_winter[i]-=1 
    if train_winter['일 최심적설(cm)'][i]>0.5:
      y_label_winter[i]-=3
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_spring, y_label_spring, test_size=0.3, random_state=42)

from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=10)
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
#0.6986494914777108

from sklearn import svm
regr = svm.SVR(C=500)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
#0.6318079802765615

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(train_spring, y_label_spring)

y_pred = regr.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
#0.9557758902558627
