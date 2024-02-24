from datetime import datetime, timedelta
import pytz
import time
import smtplib
import MetaTrader5 as mt5

import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas_ta as ta
import talib
from config import *
    
    
class MT5DataDownloader:
    def __init__(self):
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.data_dir = 'data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def start_mt5(self):
        uname = int(self.login)
        pword = str(self.password)
        trading_server = str(self.server)
        filepath = str(self.path)

        if mt5.initialize(login=uname, password=pword, server=trading_server, path=filepath):
            if mt5.login(login=uname, password=pword, server=trading_server):
                return True
            else:
                print("Login Fail")
                quit()
                return PermissionError

    def get_timeframe(self, interval):
        timeframe_mapping = {
            '1 min': mt5.TIMEFRAME_M1,
            '2 min': mt5.TIMEFRAME_M2,
            '3 min': mt5.TIMEFRAME_M3,
            '4 min': mt5.TIMEFRAME_M4,
            '5 min': mt5.TIMEFRAME_M5,
            '6 min': mt5.TIMEFRAME_M6,
            '10 min': mt5.TIMEFRAME_M10,
            '12 min': mt5.TIMEFRAME_M12,
            '15 min': mt5.TIMEFRAME_M15,
            '20 min': mt5.TIMEFRAME_M20,
            '30 min': mt5.TIMEFRAME_M30,
            '1 h': mt5.TIMEFRAME_H1,
            '2 h': mt5.TIMEFRAME_H2,
            '3h': mt5.TIMEFRAME_H3,
            '4h': mt5.TIMEFRAME_H4,
            '6h': mt5.TIMEFRAME_H6,
            '8h': mt5.TIMEFRAME_H8,
            '12h': mt5.TIMEFRAME_H12,
            '1 day': mt5.TIMEFRAME_D1,
            '1 week': mt5.TIMEFRAME_W1,
            '1 month': mt5.TIMEFRAME_MN1
        }
        return timeframe_mapping.get(interval, None)

    def get_data(self, symbol, interval, days_back):
        file_name = f"{symbol}_{interval}.csv"
        file_path = os.path.join(self.data_dir, file_name)

        if os.path.isfile(file_path):
            print(f"Data file {file_name} already exists. Skipping download.")
            data= pd.read_csv(file_path)
            data['time'] = pd.to_datetime(data['time'])  # Correct conversion
            data = data.set_index('time')
            return data

        print('The required data is not found in the database folder')
        print('getting data from broker.....')

        self.start_mt5()

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Failed to select {symbol}")
            mt5.shutdown()
            return None

        target_timezone = pytz.timezone("America/New_York")
        current_time_my = datetime.now()
        end_date = current_time_my.replace(tzinfo=target_timezone)
        start_date = end_date - timedelta(days=days_back)

        data = pd.DataFrame(mt5.copy_rates_range(symbol, self.get_timeframe(interval), start_date, end_date))
        data = data.iloc[:, :6]
        data.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        data = data.set_index('time')
        data.index = pd.to_datetime(data.index, unit='s')
        data = data.astype(float)

        # Check and count NaN values
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            print(f"Number of NaN rows in the data: {nan_count}")
        else:
            print('No NaN rows')

        # Check and count rows where open equals close
        open_equals_close_count = len(data[data['open'] == data['close']])
        if open_equals_close_count > 0:
            print(f"Number of rows where high equals low: {open_equals_close_count}")

        # Remove NaN values
        data.dropna(inplace=True)

        # Remove rows where open equals close
        data = data[data['high'] != data['low']]

        # Print the number of cleaned rows
        cleaned_row_count = len(data)
        print(f"Number of rows after cleaning: {cleaned_row_count}")

        mt5.shutdown()

        data.to_csv(file_path)
        print(f"Data saved to {file_path}")
        return data

