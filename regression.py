import pandas as pd
from iexfinance.stocks import get_historical_data
import numpy as np
import math, datetime
from datetime import datetime
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
key = 'pk_9c1ed2a08a2c4af2be14db7a6c97d602'
start = datetime(2015, 1, 1)
end = datetime.now()
df = get_historical_data("AAPL", start, end, token=key, output_format="pandas")

df['HL_PTC'] = (df['open'] - df['close']) / df['close'] * 100.0
df['PTC_change'] = (df['close'] - df['open']) / df['open'] * 100.0
df = df[['close', 'HL_PTC', 'PTC_change', 'volume']]

forecast_col = 'close'
df.fillna(-9999, inplace=True)
forecast_out = int(math.ceil(0.1 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


df.dropna(inplace=True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# clf = LinearRegression(n_jobs=1)
# clf.fit(X_train, y_train)
# with open("linearreagression.pickle", 'wb') as f:
#     pickle.dump(clf, f)

pickle_in = open("linearreagression.pickle", 'rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test, y_test)

#print(accuracy)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]
df['close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()