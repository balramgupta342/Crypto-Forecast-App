#Importing Libraries
import streamlit as st
import yfinance as yf
from datetime import date, timedelta
import tensorflow as tf
import numpy as np
from pandas_datareader import data as pdr
from sklearn.preprocessing import StandardScaler
import plotly.express as px

#Title
st.title('Crypto Forecast App')

#For Coin and Date Selection
coins = ('BTC', 'ETH', 'USDT', 'BNB','XRP','ADA','DOGE','SOL','TRX','DOT','DAI','MATIC','LTC','AVAX','LINK')
selected_coin = st.selectbox('Select coin for prediction', coins)
selected_date = st.date_input("Enter date ",max_value=date.today())

#Date variables
START = "2010-01-01"
TODAY = selected_date

#Load Available data 
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_coin+"-USD")
data_load_state.text('Loading data... done!')


st.subheader('Raw data')
st.write(data.tail())

x_axis = st.selectbox('Select X-axis value', options=data.columns)
y_axis = st.selectbox('Select Y-axis value', options=data.columns, index=4)
plot = px.line(data, x = x_axis, y= y_axis)
st.plotly_chart(plot)

#Model import
model = tf.keras.models.load_model("model")
#Fetching and Preparing data
yf.pdr_override()
quote = pdr.get_data_yahoo(selected_coin+"-USD", start=selected_date-timedelta(days = 61), end = selected_date-timedelta(days = 1))
# create a new dataframe
new_df = quote.filter(['Close'])
# get last 60 day closing value and convert to array
last_60_days = new_df[-60:].values
# scale data to values between 0 and 1
scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_df.values)
last_60_days_scaled = scaler.transform(last_60_days)
# last_60_days_scaled
X_test = []
# append past 60 days to X_test
X_test.append(last_60_days_scaled)
# convert X_test dataset to numpy array
X_test = np.array(X_test)
# reshape data
X_test = np.reshape(X_test, (X_test.shape[0] ,X_test.shape[1],1))
# get predicted price
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price = pred_price[0][0]

#Show prdicted price
st.subheader(f"The estimated closing price of {selected_coin} for {selected_date} is ${pred_price:.2f}")