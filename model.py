# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from nsepy import get_history
from datetime import date
data = get_history(symbol="SBIN", start=date(2018,1,1), end=date(2022,11,10))

df =pd.DataFrame(data,columns=['Open','Close'])

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=50)


# Fitting Support vector Regression to the dataset
from sklearn.svm import SVR
sv_regressor = SVR(kernel = 'linear')
sv_regressor.fit(X_train, y_train)
svr_pred=sv_regressor.predict(X_test)

# Saving model to disk
pickle.dump(sv_regressor, open('modell.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('modell.pkl','rb'))
print(model.predict([[613]]))
