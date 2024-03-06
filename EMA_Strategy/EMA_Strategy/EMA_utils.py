from DATA.Utils import *

import datetime
import pandas_ta as ta
import pandas as pd
import talib as ta
import seaborn as sns
import matplotlib.pyplot as plt
from backtesting import Backtest
from backtesting import Strategy
from backtesting.lib import crossover, plot_heatmaps
import seaborn as sns
import matplotlib.pyplot as plt


class EMAcrossover(Strategy):
    
    SIZE= 10
    #intsize= 0.99
    SHORT_EMA = 1  # Default value
    LONG_EMA = 1   # Default value
    
    USE_TREND_EMA = True  # Default value
    TREND_EMA= 1   # Default value
    
    use_sl= False
    use_tp= False
    BACK_CANDLES= 1  # Default value
    REWARD= 1   # Default value
    
    
    ATR_LEVEL= 1   # Default value
    
#     optimization= {'optimization': False
#               ,'short_ema':False, 'long_ema':False, 'trend_ema':False, 'back_candles':False, 'reaward':False}
    #loss_averg= 0.02
    def init(self):
        self.short_ema = self.I(ta.EMA, self.data.Close, self.SHORT_EMA)
        self.long_ema = self.I(ta.EMA, self.data.Close, self.LONG_EMA)
        if self.USE_TREND_EMA:
            self.trend_ema = self.I(ta.EMA, self.data.Close, self.TREND_EMA)
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close,timeperiod=self.ATR_LEVEL)
        #self.back_candles=BACK_CANDLES
    def next(self):
        # Use trend_ema conditionally
        if not self.USE_TREND_EMA:
            if crossover(self.short_ema, self.long_ema):
                SL= None if not self.use_sl else min( self.data.Low[-self.BACK_CANDLES:])
                TP= None if not self.use_tp else self.data.Close[-1] + (self.data.Close[-1]- SL)*self.REWARD
                self.buy(sl=SL, tp=TP, size=self.SIZE)
                if self.position.is_short:
                    self.position.close()
            elif crossover(self.long_ema, self.short_ema):
                SL= None if not self.use_sl else max( self.data.High[-self.BACK_CANDLES:])
                TP= None if not self.use_tp else self.data.Close[-1] - (SL - self.data.Close[-1])*self.REWARD
                self.sell(sl=SL, tp=TP, size=self.SIZE)
                if self.position.is_long:
                    self.position.close()
                    
        ####################################################################
        if self.USE_TREND_EMA:
            if self.data.Close> self.trend_ema:
                if self.position.is_short:
                    self.position.close()
            elif self.data.Close < self.trend_ema:
                if self.position.is_long:
                    self.position.close()
                
            if self.data.Close> self.trend_ema:
                if crossover(self.short_ema, self.long_ema):
                    SL= None if not self.use_sl else min( self.data.Low[-self.BACK_CANDLES:])
                    TP= None if not self.use_tp else self.data.Close[-1] + (self.data.Close[-1]- SL)*self.REWARD
                    self.buy(sl=SL, tp=TP, size=self.SIZE)
                elif crossover(self.long_ema, self.short_ema):
                    self.position.close()
            elif self.data.Close< self.trend_ema:
                if crossover(self.long_ema, self.short_ema):
                    SL= None if not self.use_sl else max( self.data.High[-self.BACK_CANDLES:])
                    TP= None if not self.use_tp else self.data.Close[-1] - (SL - self.data.Close[-1])*self.REWARD
                    self.sell(sl=SL, tp=TP, size=self.SIZE)
                elif crossover(self.short_ema, self.long_ema):
                    self.position.close()

                    
def ema_backtest_strategy(data, symbol, interval,
                          size=None,
                          short_ema=None, long_ema=None, trend_ema=None, use_trend_ema=None,
                          atr_level= None,
                          use_sl=None, use_tp=None, back_candles= None, reward= None, 
                          optimization=None, maximize='Return [%]'):
    # Dynamically set the EMA values if provided
    if size is not None:
        EMAcrossover.SIZE= size
        
    if short_ema is not None:
        EMAcrossover.SHORT_EMA = short_ema
    if long_ema is not None:
        EMAcrossover.LONG_EMA = long_ema

    if use_trend_ema is not None:
        EMAcrossover.USE_TREND_EMA= use_trend_ema
    if trend_ema is not None:
        EMAcrossover.TREND_EMA= trend_ema

    
    if use_sl is not None:
        EMAcrossover.use_sl= use_sl
    if use_tp is not None:
        EMAcrossover.use_tp= use_tp
    if back_candles is not None:
        EMAcrossover.BACK_CANDLES= back_candles
    if reward is not None:
        EMAcrossover.REWARD= reward

    if atr_level is not None:
        EMAcrossover.ATR_LEVEL= atr_level
#     if optimization is not None: 
#         EMAcrossover.optimization= optimization
    bt = Backtest(data, EMAcrossover, cash=100_000, margin=1/1, commission=.0005)
    
    # Define the columns for the empty DataFrame
    columns = ['short_ema', 'long_ema', 'trend_ema', 'sl_back_candles', 'reward', 'optimization',
              'Trend_opt','sl_opt', 'tp_opt']
    # Create an empty DataFrame with the specified columns
    input_df = pd.DataFrame(columns=columns)

    if optimization['optimization']:
        stats, heatmap= bt.optimize(
        SHORT_EMA= range(5, 30, 1) if optimization['short_ema'] else short_ema,
        LONG_EMA= range(10,35, 1) if optimization['long_ema'] else long_ema,
        TREND_EMA= range(40,200, 5) if optimization['trend_ema'] else trend_ema,
        BACK_CANDLES= range(3, 20, 1) if optimization['back_candles'] else back_candles,
        REWARD= range(0.5,2, 0.2) if optimization['reward'] else reward,   
        ATR_LEVEL= 14,
        maximize= maximize,
        constraint = lambda param: 5 < (param.LONG_EMA - param.SHORT_EMA) < 50,
        return_heatmap= True
        ) 
        print(f'\n summary bactesting result of {symbol}_{interval}\n')
        print(stats)
        
        # After obtaining the values, either from optimization results or direct parameters
        values = {
            'short_ema': stats._strategy.SHORT_EMA if optimization.get('short_ema') else short_ema,
            'long_ema': stats._strategy.LONG_EMA if optimization.get('long_ema') else long_ema,
            'trend_ema': stats._strategy.TREND_EMA if optimization.get('trend_ema') else trend_ema,
            'sl_back_candles': stats._strategy.BACK_CANDLES if optimization.get('back_candles') else back_candles,
            'reward': stats._strategy.REWARD if optimization.get('reward') else reward,
            'optimization': optimization.get('optimization', False),
            'Trend_opt': optimization.get('trend_ema', False),
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
        print(stats)
        #return stats, None
        
        values = {
        'short_ema': short_ema,
        'long_ema': long_ema,
        'trend_ema': trend_ema if use_trend_ema else None,
        'sl_back_candles': back_candles if use_sl else None,
        'reward': reward if use_tp else None,
        'optimization': optimization.get('optimization', False),
        'Trend_opt': False,  # Assuming you want to keep track if trend_ema was considered
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
    # Merge stat DataFrame to the left of other_data DataFrame
    
    # Assuming you want to use values from the first row to name the file
    row = input_df.iloc[0]  # First row
    # Construct file name
    optimization_status = "Optimized" if row['optimization'] else "NotOptimized"
    file_name = f"{symbol}_{interval}_EMA_{row['short_ema']}_{row['long_ema']}_{row['trend_ema']}_sl{row['sl_back_candles']}_tp{row['reward']}_{'Optimized' if row['optimization'] else 'NotOptimized'}.xlsx"

    with pd.ExcelWriter(f'Results/TRADES/{file_name}', engine='openpyxl') as writer:
        # Write each DataFrame to a different worksheet
        input_df.to_excel(writer, sheet_name='Input_setting')
        Trades.to_excel(writer, sheet_name='Trades')
        stat_df.to_excel(writer, sheet_name='results')
        if heatmap is not None:
            heatmap=pd.DataFrame(heatmap)
            heatmap.to_excel(writer, sheet_name='heatmap')
            
    bt.plot(resample=False,filename= f'Results/PLOTS/{file_name}')

    return input_df, stat_df, Trades, heatmap


# Trades['ReturnPct']= Trades['ReturnPct']*100 
# # Calculate Losses Column
# Trades['Losses'] = Trades['ReturnPct'].apply(lambda x: x if x < 0 else np.nan)

# # Initialize a new column for cumulative drawdown with zeros
# Trades['CumulativeDrawdown'] = 0

# # Temporary variable to hold the running sum of losses
# running_loss = 0

# # Iterate through each trade
# for index, row in Trades.iterrows():
#     # If the PnL is negative, add it to the running loss and set the drawdown
#     if row['ReturnPct'] < 0:
#         running_loss += row['ReturnPct']
#         Trades.at[index, 'CumulativeDrawdown'] = running_loss
#     # If the PnL is positive, reset the running loss and set the drawdown to NaN
#     else:
#         running_loss = 0
#         Trades.at[index, 'CumulativeDrawdown'] = np.nan
        
# # Initialize the 'Drawdowns' column with NaNs
# Trades['Drawdowns'] = np.nan
# # Temporary variables to hold the cumulative drawdown and last index during a losing streak
# cumulative_drawdown = 0
# last_loss_index = None
# # Iterate through each trade
# for index, row in Trades.iterrows():
#     if row['ReturnPct'] < 0:
#         # If the trade is a loss, update the cumulative drawdown and the last index
#         cumulative_drawdown += row['ReturnPct']
#         last_loss_index = index
#     else:
#         # If the trade is a win and there was a previous losing streak
#         if last_loss_index is not None:
#             # Set the last row of the losing streak with the cumulative drawdown
#             Trades.at[last_loss_index, 'Drawdowns'] = cumulative_drawdown
#             # Reset cumulative drawdown and last index for the next streak
#             cumulative_drawdown = 0
#             last_loss_index = None
# # If the last trade(s) were losses, we need to set the drawdown for the last one
# if last_loss_index is not None:
#     Trades.at[last_loss_index, 'Drawdowns'] = cumulative_drawdown