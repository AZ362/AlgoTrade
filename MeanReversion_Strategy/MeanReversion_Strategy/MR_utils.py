from DATA.Utils import *
from backtesting import Strategy
from backtesting import Backtest


def ema_trend(df):
    return df.ema
def Bullish_Engulfing(df):
    return df.BullishEngulfing
def Bearish_Engulfing(df):
    return df.BearishEngulfing
def Hammer(df):
    return df.Hammer
def InvertedHammer(df):
    return df.InvertedHammer

def HangingMan(df):
    return df.HangingMan




def ShootingStar(df):
    return df.ShootingStar

def DragonFlyDoji(df):
    return df.DragonFlyDoji
def GravestoneDoji(df):
    return df.GravestoneDoji


def optim_Equity_ExposureTime(series):
    if series['# Trades'] < 10:
        return -1
    else:
        return series['Equity Final [$]']/series['Exposure Time [%]']


def optim_WinRate(series):
    if series['# Trades'] < 10:
        return -1
    else:
        return series['Win Rate [%]']

class MeanReversion(Strategy):
    
    SIZE= 0.99
    DEVIATION=0.5 ## 1>> 0.01  from price
    #intsize= 0.99
    
   
    CANDLESTICK_INDICATORS ={
    "BullishEngulfing": False,
    "Hammer": False,
    "InvertedHammer": False,
    "DragonFlyDoji": False,
    
    "BearishEngulfing": False,
    "ShootingStar": False,
    "Hangingman":False,
    "GravestoneDoji": False
    }
    
    BACK_CANDLES= 2  # Default value
    REWARD= 1   # Default value
    
    
    USE_RSI= False
    RSI= 14
    OVERBOUTH_RSI= 70
    OVERSELL_RSI= 30
    
    RISKPCT=0.01 ## default

    def init(self):
        
         # Wrapper function for ema_trend
        def ema_trend_wrapper():
            return ema_trend(self.data)
        
        # Wrapper functions for other indicators
        def Bullish_Engulfing_wrapper():
            return Bullish_Engulfing(self.data)
        
        def Bearish_Engulfing_wrapper():
            return Bearish_Engulfing(self.data)
        
        def Hammer_wrapper():
            return Hammer(self.data)
        
        def InvertedHammer_wrapper():
            return InvertedHammer(self.data)
        
        def HangingMan_wrapper():
            return HangingMan(self.data)
        
        def ShootingStar_wrapper():
            return ShootingStar(self.data)
        
        def DragonFlyDoji_wrapper():
            return DragonFlyDoji(self.data)
        
        def GravestoneDoji_wrapper():
            return GravestoneDoji(self.data)
        # Initialization of the EMA slow as it's not conditional
        self.ema = self.I(ema_trend_wrapper)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.RSI)
        # Dynamically create indicators based on candlestick indicators
        for pattern, is_active in self.CANDLESTICK_INDICATORS.items():
            if is_active:
                if pattern == "BullishEngulfing":
                    self.BullishEngulfing = self.I(Bullish_Engulfing_wrapper)
                    #print(self.Bullish_Engulfing)
                elif pattern == "BearishEngulfing":
                    self.BearishEngulfing = self.I(Bearish_Engulfing_wrapper)
                elif pattern == "Hammer":
                    self.Hammer = self.I(Hammer)
                elif pattern == "InvertedHammer":
                    self.InvertedHammer = self.I(InvertedHammer_wrapper)
                elif pattern == "HangingMan":
                    self.HangingMan = self.I(HangingMan_wrapper)
                elif pattern == "DragonFlyDoji":
                    self.DragonFlyDoji = self.I(DragonFlyDoji_wrapper)
                elif pattern == "ShootingStar":
                    self.ShootingStar = self.I(ShootingStar_wrapper)
                    #print(self.ShootingStar)
                elif pattern == "GravestoneDoji":
                    self.GravestoneDoji = self.I(GravestoneDoji_wrapper)

        # Example of adding ATR if needed in your strategy, adjust accordingly
        # self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.ATR_LEVEL)
    def next(self):
        # Example placeholder logic for determining if conditions are met for buy/sell
        # based on active candlestick patterns.
        for pattern, is_active in self.CANDLESTICK_INDICATORS.items():
            if not is_active:
                continue  # Skip this pattern if it's not active
            
            # Dynamically get the indicator attribute based on the pattern name
            indicator = getattr(self, pattern, None)

            # Ensure the indicator exists before proceeding
            if indicator is not None:
                latest_indicator_value = indicator[-1]
            
            #if not self.use_rsi:
            RSI_buy=True
            RSI_sell=True
            if self.USE_RSI:
                if self.rsi>self.OVERBOUTH_RSI:
                    RSI_buy=False
                elif self.rsi< self.OVERSELL_RSI:
                    RSI_sell=False
                
            if RSI_buy:
                
                if pattern in ["BullishEngulfing", "Hammer","InvertedHammer", "DragonFlyDoji"] and latest_indicator_value == 100 and len(self.trades) == 0:
                    # This is a simplified logic; you'll need to adjust according to your strategy's requirements
                    #if self.condition_for_buying_based_on(pattern):
                    if self.ema - self.data.Close > self.DEVIATION*self.data.Close/100:
                        
                        SL = min(self.data.Low[-self.BACK_CANDLES:])
                        TP = self.data.Close[-1] + (self.data.Close[-1] - SL) * self.REWARD
                        #SIZE_=int(self.SIZE*1_000_000/abs(self.data.Close[-1] -SL)/100)
                        aval_lots=int(self.equity/self.data.Close[-1])
                        trade_size=int(self.equity*self.RISKPCT/(abs(self.data.Close[-1] -SL)))#
                        SIZE2= trade_size if trade_size<= aval_lots else aval_lots
                    
#                         print('-------------------------------------------------------')
#                         print(self.rsi[-1])
#                         print(f'available lots: {aval_lots}')
#                         print(f'trade_size: {trade_size}')
#                         print(f'Size is:{SIZE2}')
#                         print(f'equity is:{self.equity}')
                        self.buy(sl=SL, tp=TP, size=SIZE2)
                        if self.position.is_short:
                            self.position.close()
            if RSI_sell:
                if pattern in ["BearishEngulfing", "ShootingStar","HangingMan", "GravestoneDoji"] and latest_indicator_value == -100 and len(self.trades) == 0:
                    # Again, this is simplified logic
                    #if self.condition_for_selling_based_on(pattern):
                    if self.data.Close -self.ema > self.DEVIATION*self.data.Close/100:
                        
                        SL = max(self.data.High[-self.BACK_CANDLES:])
                        TP = self.data.Close[-1] - (SL - self.data.Close[-1]) * self.REWARD
                        #SIZE_=int(self.SIZE*1_000_000/abs(self.data.Close[-1] -SL)/100)
                        aval_lots=int(self.equity/self.data.Close[-1])
                        trade_size=int(self.equity*self.RISKPCT/(abs(self.data.Close[-1] -SL)))##
                        SIZE2= trade_size if trade_size< aval_lots else aval_lots
#                         print('-------------------------------------------------------')
#                         print(self.rsi[-1])
#                         print(f'available lots: {aval_lots}')
#                         print(f'trade_size: {trade_size}')
#                         print(f'Size is:{SIZE2}')
#                         print(f'equity is:{self.equity}')
                        self.sell(sl=SL, tp=TP, size=SIZE2)## size= (balance*riskpct)/(slatr*contractsize)
                        if self.position.is_long:
                            self.position.close()
                        
        if self.position.is_long:
            if self.data.Close > self.ema:
                self.position.close()
        if self.position.is_short:
            if self.data.Close < self.ema:
                self.position.close()
                        
def run_backtests(symbols, intervals, start_date, end_date,capital, riskpct, fees,
                          ema,
                          candlestick_indicators,
                        deviation,
                         use_rsi, rsi, overbought_rsi, oversell_rsi,back_candles, reward, optimization, maximize):
    all_results = {}

    for symbol in symbols:
        for interval in intervals:
            print(f'\n\n------------------------{symbol}----------------------------------')
            print(f'      -----------------{interval}-----------------')
            df_bt = load_and_filter_data(symbol, interval, start_date, end_date)
            
            if df_bt.empty:
                print(f"No data for {symbol} {interval}")
                continue

            try:
                df = apply_candlestick_patterns(df_bt, ema_window=ema, candlestick_indicators=candlestick_indicators)
                print(df)
                input_df, stat_df, Trades, heatmap = MR_backtest(df, symbol, interval,
                            capital, riskpct, fees,
                          ema,
                          candlestick_indicators,
                        deviation,
                         use_rsi, rsi, overbought_rsi, oversell_rsi,                                     
                         back_candles, reward, 
                          optimization, maximize)

                all_results[(symbol, interval)] = {
                    'input_df': input_df,
                    'stat_df': stat_df,
                    'trades': Trades,
                    'heatmap': heatmap
                    }
            except Exception as e:
                print(f"Failed to backtest {symbol} {interval}: {e}")
    
    print("Backtesting complete.")
    return all_results    
                        
                        
def MR_backtest(data, symbol, interval,
                          capital= None, riskpct=None, fees=None,
                          ema=None,
                          candlestick_indicators=None,
                        deviation= None,
                use_rsi=None, rsi=None, overbought_rsi=None, oversell_rsi=None,
                         back_candles= None, reward= None, 
                          optimization=None, maximize='Return [%]'):
    # Dynamically set the EMA values if provided
    if riskpct is not None:
        MeanReversion.RISKPCT= riskpct
 
    if ema is not None:
        MeanReversion.EMA = ema
        
    if candlestick_indicators is not None:
        MeanReversion.CANDLESTICK_INDICATORS = candlestick_indicators
    if deviation is not None:
        MeanReversion.DEVIATION= deviation
        
        
    if use_rsi is not None:
        MeanReversion.USE_RSI= use_rsi
    if rsi is not None:
        MeanReversion.RSI= rsi
    if overbought_rsi is not None:
        MeanReversion.OVERBOUGHT_RSI= overbought_rsi
    if oversell_rsi is not None:
        MeanReversion.OVERSELL_RSI= oversell_rsi
#     if use_sl is not None:
#         MeanReversion.use_sl= use_sl
#     if use_tp is not None:
#         MeanReversion.use_tp= use_tp
    if back_candles is not None:
        MeanReversion.BACK_CANDLES= back_candles
    if reward is not None:
        MeanReversion.REWARD= reward

        
    bt = Backtest(data, MeanReversion, cash=capital, margin=1/10, commission= fees)
    
    # Define the columns for the empty DataFrame
    columns = ['ema','deviation', 'sl_back_candles', 'reward', 'optimization',
              'dev_opt','sl_opt', 'tp_opt']
    # Create an empty DataFrame with the specified columns
    input_df = pd.DataFrame(columns=columns)

    if optimization['optimization']:
        stats, heatmap= bt.optimize(
        DEVIATION= (0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.5, 3) if optimization['deviation'] else deviation,
        BACK_CANDLES= range(1, 10, 1) if optimization['back_candles'] else back_candles,
        REWARD= (1,1.5, 2, 2.5, 3) if optimization['reward'] else reward,   
        maximize= maximize,
        #constraint = lambda param: 5 < (param.LONG_EMA - param.SHORT_EMA) < 50,
        #max_tries=100,
        return_heatmap= True
        ) 
        print(f'\n summary bactesting result of {symbol}_{interval}\n')
        #print(stats)
       
        # After obtaining the values, either from optimization results or direct parameters
        values = {
            'ema': stats._strategy.LONG_EMA if optimization.get('ema') else ema,
            'deviation':stats._strategy.DEVIATION if optimization.get('deviation') else deviation,
            'sl_back_candles':stats._strategy.BACK_CANDLES if optimization.get('back_candles') else back_candles,
            'reward': stats._strategy.REWARD if optimization.get('reward') else reward, 
            'optimization': optimization.get('optimization', False),
            'dev_opt': optimization.get('deviation', False),
            'sl_opt': optimization.get('back_candles', False),
            'tp_opt': optimization.get('reward', False)
        }

        # Convert values dictionary to a DataFrame row
        row_df = pd.DataFrame([values])  # Creates a DataFrame with a single row
        # Append this row to input_df
        input_df = pd.concat([input_df, row_df], ignore_index=True)
    
    else:
        heatmap=None
        stats = bt.run()
        print(f'\n summary backtesting result of {symbol}_{interval}\n')
        #print(stats)
        #return stats, None
        
        values = {
        'ema': ema,
        'deviation': deviation,
        'sl_back_candles': back_candles,
        'reward': reward,
        'optimization': optimization.get('optimization', False),
        'deviation_opt': False,  # Assuming you want to keep track if trend_ema was considered
        'sl_opt': False,  # Track if stop loss was considered
        'tp_opt': False  # Track if take profit was considered
        }

        # Convert values dictionary to a DataFrame row
        row_df = pd.DataFrame([values])  # Creates a DataFrame with a single row

        # Append this row to input_df
        input_df = pd.concat([input_df, row_df], ignore_index=True)
    ## Excute data
    Trades=stats._trades
    Trades['Duration']= Trades['Duration'].astype(str)
    # Calculate 'used_amount'
    Trades['used_amount'] = abs(Trades['Size']) * Trades['EntryPrice']
    # Assign 'SL' and 'TP' based on 'PnL'
    Trades['SL'] = np.where(Trades['PnL'] < 0, abs(Trades['ExitPrice']-Trades['EntryPrice']), abs(Trades['ExitPrice']-Trades['EntryPrice'])/input_df['reward'][0])
    #Trades['TP'] = np.where(Trades['PnL'] > 0, Trades['ExitPrice'], None)
    Trades['TP'] = np.where(Trades['PnL'] < 0, abs(Trades['ExitPrice']-Trades['EntryPrice'])*input_df['reward'][0], abs(Trades['ExitPrice']-Trades['EntryPrice']))
    # Calculate cumulative PnL
    Trades['Cumulative PnL'] = Trades['PnL'].cumsum()
    
    # Assuming 'df' is your DataFrame containing the Trades data
    new_order = ['Size', 'used_amount', 'EntryBar', 'ExitBar', 'EntryPrice', 'ExitPrice', 'SL', 'TP', 'PnL', 'Cumulative PnL', 'ReturnPct', 'EntryTime', 'ExitTime', 'Duration']

    # Reorder the DataFrame according to 'new_order'
    Trades = Trades[new_order]
    
    # Shift the 'data' index by one period
    data_shifted = data.shift(1)
    # Assuming 'data_shifted' has already been created from 'data'
    data_shifted.index = pd.to_datetime(data_shifted.index)
    # Convert 'EntryTime' in 'Trades' to datetime if it's not already
    Trades['EntryTime'] = pd.to_datetime(Trades['EntryTime'])
    # Perform the asof merge
    Trades = pd.merge_asof(Trades.sort_values('EntryTime'), data_shifted, left_on='EntryTime', right_index=True)
    ## Additional analysis of Trades
    #Trades= Trade_manage(Trades, capital)
    #summary_trades = summarize_trades(Trades)
    #print(summary_trades)
     # Convert stat to DataFrame
    stat_df = pd.DataFrame(stats).T
    stat_df['symbol']= symbol
    stat_df['interval']= interval
    stat_df = stat_df.rename(columns={'Duration': 'full_Duration'})
    stat_df['Start']= stat_df['Start'].astype(str)
    stat_df['End']= stat_df['End'].astype(str)
    stat_df['full_Duration']= stat_df['full_Duration'].astype(str)
    stat_df['Max. Drawdown Duration']= stat_df['Max. Drawdown Duration'].astype(str)
    stat_df['Avg. Drawdown Duration']= stat_df['Avg. Drawdown Duration'].astype(str)
    stat_df['Max. Trade Duration']= stat_df['Max. Trade Duration'].astype(str)
    stat_df['Avg. Trade Duration']= stat_df['Avg. Trade Duration'].astype(str)
    # Move column 'symbol' to the first position
    cols = stat_df.columns.tolist()  # Get the list of column names
    cols = ['symbol'] + [col for col in cols if col != 'symbol']  # Rearrange the column names
    stat_df = stat_df[cols]  # Reorder the DataFrame based on the new column order
    stat_df=stat_df.iloc[:, :-2]
    #stat_df = stat_df.rename(columns={'_strategy': 'ema_level'})
    stat_df['optimization']= optimization['optimization']
    
    ## Calculate Expected Value 
    #Riskamount=riskpct*capital#
    EV= stat_df['Win Rate [%]']/100*input_df['reward'] - (1-(stat_df['Win Rate [%]']/100))*1 
    stat_df['EV [%]']=round(EV[0],2) *100 ### EV*Riskamount>> profit average per trade 
    stat_df['RR [%]']=stat_df['EV [%]']*stat_df['# Trades']/100
    #EV ## EV*Risk
    
    # Calculate the breakeven win rate
    breakeven_WinRate = 1 / (1 + input_df['reward'])* 100
    stat_df['breakeven_WinRate [%]']= round(breakeven_WinRate[0],2)
    
    stat_df= stat_df.T
    print(stat_df)
    # Assuming you want to use values from the first row to name the file
    row = input_df.iloc[0]  # First row
    
    
    # Inside your function, after the optimization and result processing
    symbol_dir_trades = os.path.join('Results', symbol, interval, 'trades')
    symbol_dir_plots = os.path.join('Results', symbol, interval, 'plots')

    # Create the directories if they don't exist
    os.makedirs(symbol_dir_trades, exist_ok=True)
    os.makedirs(symbol_dir_plots, exist_ok=True)
    
    # Construct file name
    optimization_status = "Optimized" if row['optimization'] else "NotOptimized"
    
    # Create a file name containing all active candles separated by '_'
    active_candles = [key for key, value in candlestick_indicators.items() if value]
    active_cname = "_".join(active_candles)

    file_name = f"{symbol}_{interval}_{active_cname}_EMA{row['ema']}_dev{row['deviation']}_sl{row['sl_back_candles']}_tp{row['reward']}_{'Optimized' if row['optimization'] else 'NotOptimized'}.xlsx"

    
    # Now, when saving the Excel and plot files, use the full paths
    excel_file_path = os.path.join(symbol_dir_trades, file_name)
    plot_file_path = os.path.join(symbol_dir_plots, file_name.replace('.xlsx', '.png'))  # Assuming plot function saves as .png

    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        # Write each DataFrame to a different worksheet
        input_df.to_excel(writer, sheet_name='Input_setting')
        Trades.to_excel(writer, sheet_name='Trades')
        #summary_trades.to_excel(writer, sheet_name='summary_trades')
        stat_df.to_excel(writer, sheet_name='results')
        if heatmap is not None:
            heatmap= heatmap.sort_values(ascending=False)
            heatmap= pd.DataFrame(heatmap)
            heatmap.to_excel(writer, sheet_name='heatmap')
            
    bt.plot(resample=False,filename= plot_file_path)

    return input_df, stat_df, Trades, heatmap

def load_and_filter_data(symbol, interval, start_date, end_date):
    # Construct the file path
    data_dir = 'DATA/data'
    file_name = f"{symbol}_{interval}.csv"
    file_path = os.path.join(data_dir, file_name)
    
    # Initialize an empty DataFrame
    df_bt = pd.DataFrame()
    
    # Check if the file exists
    if os.path.isfile(file_path):
        print(f"Data file {file_name} found. Loading data.")
        
        # Load the data
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        print('----------------------Full Data----------------------')
        print(df)
        print('-----------------------------------------------------')
        
        # Ensure start_date and end_date are in the correct format
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check if start_date is before end_date
        if start_date >= end_date:
            print("start_date must be earlier than end_date.")
            return df_bt  # Return empty DataFrame
        
        # Check if the dates are within the DataFrame's range
        if start_date < df.index.min() or end_date > df.index.max():
            print("Start_date and end_date must be within the range of the DataFrame.")
            return df_bt  # Return empty DataFrame
        
        # Filter the DataFrame based on the start and end dates
        df_bt = df.loc[start_date:end_date]
        print('----------------Backtesting Data---------------------')
        print(df_bt)
        
    else:
        print(f"Data file {file_name} not found. Please download it before proceeding.")
    
    return df_bt  # This will return either the filtered DataFrame or an empty DataFrame if not found or any check fails


def trend_ema_trend(df, window=26):
    """
    Calculate the Slow Exponential Moving Average (EMA).

    :param df: DataFrame containing 'close'.
    :param window: The period for the EMA calculation, default is 26.
    :return: Pandas Series with the Slow EMA values.
    """
    close = df['Close'].values
    slow_ema = talib.EMA(close, timeperiod=window)
    return pd.Series(slow_ema, index=df.index)

def bullish_engulfing_candle(df):
    """
    Detect Bullish Engulfing Candle Pattern.

    :param df: DataFrame containing 'open', 'high', 'low', 'close'.
    :return: Pandas Series with 100 where Bullish Engulfing Candle pattern is detected, 0 otherwise.
    """
    engulfing = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
    integer = (engulfing > 0).astype(int) * 100  # Keep 100 where bullish, 0 otherwise
    return integer

def bearish_engulfing_candle(df):
    """
    Detect Bearish Engulfing Candle Pattern.

    :param df: DataFrame containing 'open', 'high', 'low', 'close'.
    :return: Pandas Series with -100 where Bearish Engulfing Candle pattern is detected, 0 otherwise.
    """
    engulfing = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
    integer = (engulfing < 0).astype(int) * -100  # Keep -100 where bearish, 0 otherwise
    return integer


def Hammer_candle(df): ## Bullish
    
    integer = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    return integer  

def InvertedHammer_candle(df): ## Bullish
    integer = talib.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    return integer  

def HangingMan_candle(df): ## Bullish
    integer = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
    return integer 

def ShootingStar_candle(df): ## Bearish
    integer = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    return integer  


# #########
# def Doji_candle(df): ## Neutral
    
#     integer = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
#     return integer 

def DragonflyDoji_candle(df): ## Bullish
    
    integer = talib.CDLDRAGONFLYDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    return integer 

def GravestoneDoji_candle(df): ##Bearish
    
    integer = talib.CDLGRAVESTONEDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    return integer 





def apply_candlestick_patterns(df, ema_window=100, candlestick_indicators=None):
    if candlestick_indicators is None:
        candlestick_indicators = {
            "BullishEngulfing": False,
            "Hammer": False,
            "InvertedHammer": False,
            "DragonFlyDoji": False,
            
            "BearishEngulfing": False,
            "ShootingStar": False,
            "HanginMan":False,
            "GravestoneDoji": False
        }

    # Calculate EMA and drop rows with NaN values
    df['ema'] = trend_ema_trend(df, window=ema_window)
    df.dropna(inplace=True)

    # Apply candlestick patterns based on the indicators dictionary
    for pattern, is_active in candlestick_indicators.items():
        if is_active:
            if pattern == "BullishEngulfing": 
                df['BullishEngulfing'] = bullish_engulfing_candle(df)
            elif pattern == "BearishEngulfing":
                df['BearishEngulfing'] = bearish_engulfing_candle(df)
            elif pattern == "Hammer":
                df['Hammer'] = Hammer_candle(df)
            elif pattern == "InvertedHammer":
                df['InvertedHammer'] = InvertedHammer_candle(df)
            elif pattern == "DragonFlyDoji":
                df['DragonFlyDoji'] = DragonflyDoji_candle(df)
            elif pattern == "ShootingStar":
                df['ShootingStar'] = ShootingStar_candle(df)
            elif pattern == "HangingMan":
                df['HangingMan'] = HangingMan_candle(df)
            elif pattern == "GravestoneDoji":
                df['GravestoneDoji'] = GravestoneDoji_candle(df)
            # Add additional patterns as needed

    return df


# Define the names of the directories to be created
directories = ['Results', 'Results/PLOTS', 'Results/TRADES']

# Loop through the directory names and create them if they don't exist
for dir_name in directories:
    # Construct the full path for the directory
    full_path = os.path.join(dir_name)
    # Check if the directory exists, and if not, create it
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Directory '{dir_name}' created at: {full_path}")
    else:
        print(f"Directory '{dir_name}' already exists at: {full_path}")
        

def Trade_manage(Trades, init_capital):
    # Calculate cumulative PnL
    Trades['Cumulative PnL'] = Trades['PnL'].cumsum()

    # Calculate cumulative return by adding the initial capital to the cumulative PnL
    Trades['Cumulative Return'] = init_capital + Trades['Cumulative PnL']

    # Calculate ReturnPct2 from cumulative Return
    Trades['ReturnPct2'] = Trades['PnL'] / Trades['Cumulative Return']

    # Mark losses and set gains to NaN
    Trades['Losses'] = Trades['ReturnPct'].apply(lambda x: x if x < 0 else np.nan)

    # Mark losses and set gains to NaN
    Trades['Losses2'] = Trades['ReturnPct2'].apply(lambda x: x if x < 0 else np.nan)

    # Initialize a new column for cumulative drawdown with zeros
    Trades['CumulativeDrawdown'] = 0

    # Temporary variable to hold the running sum of losses
    running_loss = 0

    # Iterate through each trade to calculate cumulative drawdown
    for index, row in Trades.iterrows():
        if row['ReturnPct2'] < 0:
            running_loss += row['ReturnPct2']
            Trades.at[index, 'CumulativeDrawdown'] = running_loss
        else:
            running_loss = 0
            Trades.at[index, 'CumulativeDrawdown'] = np.nan

    # Initialize the 'Drawdowns' column with NaNs
    Trades['Drawdowns'] = np.nan

    # Variables to hold the cumulative drawdown and last index during a losing streak
    cumulative_drawdown = 0
    last_loss_index = None

    # Iterate through each trade to assign drawdowns
    for index, row in Trades.iterrows():
        if row['ReturnPct2'] < 0:
            cumulative_drawdown += row['ReturnPct2']
            last_loss_index = index
        else:
            if last_loss_index is not None:
                Trades.at[last_loss_index, 'Drawdowns'] = cumulative_drawdown
                cumulative_drawdown = 0
                last_loss_index = None

    # Handle the last streak of losses, if any
    if last_loss_index is not None:
        Trades.at[last_loss_index, 'Drawdowns'] = cumulative_drawdown

    return Trades

def summarize_trades(trades_df):
    # Define a dictionary to hold summary statistics
    summary = {
        'Total Trades': len(trades_df),
        'Profitable Trades': len(trades_df[trades_df['PnL'] > 0]),
        'Losing Trades': len(trades_df[trades_df['PnL'] < 0]),
        'Win Rate': len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        #'Total PnL': trades_df['PnL'].sum(),
        #'Average PnL': trades_df['PnL'].mean(),
        #'Median PnL': trades_df['PnL'].median(),
        #'PnL Standard Deviation': trades_df['PnL'].std(),
        'Max PnL': trades_df['PnL'].max(),
        'Min PnL': trades_df['PnL'].min(),
        'Min losses': trades_df['Losses'].max(),
        'Min losses': trades_df['Losses'].min(),
        'Average losses': trades_df['Losses'].mean(),
        'Average losses2': trades_df['Losses2'].mean(),
        'Final Cumulative Return': trades_df['Cumulative Return'].iloc[-1] if len(trades_df) > 0 else np.nan,
        'Average ReturnPct2': trades_df['ReturnPct2'].mean(),
        'Max Drawdown': trades_df['Drawdowns'].min(), 
        'Min Drawdown': trades_df['Drawdowns'].max(),
         'Average Drawdown': trades_df['Drawdowns'].mean()
        

    }

    # Convert the summary dictionary into a DataFrame for nice formatting
    summary_Trades = pd.DataFrame(summary, index=[0])

    return summary_Trades


