import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def daily_price_historical(symbol, comparison_symbol, all_data=True, limit=1, aggregate=1, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    if all_data:
        url += '&allData=true'
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    return df

#loading dataset
bitcoin=daily_price_historical('BTC','USD')

#BTC price over time
plt.figure(figsize = (12, 7))
plt.plot(bitcoin["time"], bitcoin["close"], color='goldenrod', lw=2)
plt.title("Bitcoin Price over time", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("$ Price", size=20)

#BTC volume over time
plt.figure(figsize = (12, 7))
plt.plot(bitcoin["timestamp"], bitcoin["volumeto"], color='royalblue', lw=2)
plt.title("Bitcoin Volume over time", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("Volume", size=20);

#columns which we will use to fit our model
required_features = ['high', 'low', 'open', 'volumefrom', 'volumeto']
#target variable
output_label = 'close'

#divideing the bitcoin dataset into train and test parts
x_train, x_test, y_train, y_test = train_test_split(
bitcoin[required_features],
bitcoin[output_label],
test_size = 0.3
)

#creating model
model = LinearRegression()
model.fit(x_train, y_train)


print("Model score: "+ str(model.score(x_test, y_test)))
print("Linear model predict: "+ str(model.predict(x_test)))


#predicting price
future_set = bitcoin.shift(periods=30).tail(30)
prediction = model.predict(future_set[required_features])



#prediction graph
plt.figure(figsize = (14, 7))
plt.plot(bitcoin["time"][-400:-60], bitcoin["close"][-400:-60], color='goldenrod', lw=2)
plt.plot(future_set["time"], prediction, color='deeppink', lw=2)
plt.title("Prediction", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("$ Price", size=20)
plt.show()


