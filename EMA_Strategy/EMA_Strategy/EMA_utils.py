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
import openpyxl


# Define the names of the directories to be created
directories = ['Results']#, 'Results/PLOTS', 'Results/TRADES']
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


def run_backtests(symbols, intervals, start_date, end_date, capital, size, fees, short_ema, long_ema, trend_ema, use_trend_ema, atr_level, use_sl, use_tp, back_candles, reward, use_exit_emaclose, optimization, maximize):
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
                input_df, stats, trades, summary_trades, heatmap = ema_backtest_strategy(
                    df_bt, symbol, interval,
                    capital, size, fees, short_ema, long_ema, trend_ema, use_trend_ema,
                    atr_level, use_sl, use_tp, back_candles, reward, use_exit_emaclose,
                    optimization, maximize
                )
                all_results[(symbol, interval)] = {
                    'input_df': input_df,
                    'stats': stats,
                    'trades': trades,
                    'summary_trades': summary_trades,
                    'heatmap': heatmap
                }
            except Exception as e:
                print(f"Failed to backtest {symbol} {interval}: {e}")
    
    print("Backtesting complete.")
    return all_results    



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
    
    use_exit_emaclose=False
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
            if self.position.is_long:
                if self.data.Close < self.trend_ema:
                    self.position.close()
            elif self.position.is_short:
                if self.data.Close> self.trend_ema:
                    self.position.close()
                
            if self.short_ema >= self.trend_ema or self.long_ema >= self.trend_ema:
                if crossover(self.short_ema, self.long_ema) or crossover(self.short_ema, self.trend_ema):
                    SL= None if not self.use_sl else min( self.data.Low[-self.BACK_CANDLES:])
                    TP= None if not self.use_tp else self.data.Close[-1] + (self.data.Close[-1]- SL)*self.REWARD
                    self.buy(sl=SL, tp=TP, size=self.SIZE)
                elif crossover(self.long_ema, self.short_ema) or crossover(self.trend_ema, self.short_ema):
                    self.position.close()
            elif self.short_ema <= self.trend_ema or self.long_ema <= self.trend_ema:
                if crossover(self.long_ema, self.short_ema) and crossover(self.trend_ema, self.short_ema):
                    SL= None if not self.use_sl else max( self.data.High[-self.BACK_CANDLES:])
                    TP= None if not self.use_tp else self.data.Close[-1] - (SL - self.data.Close[-1])*self.REWARD
                    self.sell(sl=SL, tp=TP, size=self.SIZE)
                elif crossover(self.short_ema, self.long_ema) or crossover(self.short_ema, self.trend_ema):
                    self.position.close()
        #############################
        if self.use_exit_emaclose:
            if self.position.is_long:
                 if self.data.Close < self.short_ema:
                        self.position.close()
            if self.position.is_short:
                 if self.data.Close > self.short_ema:
                        self.position.close()
                    

def ema_backtest_strategy(data, symbol, interval,
                          capital=None, size=None, fees= None,
                          short_ema=None, long_ema=None, trend_ema=None, use_trend_ema=None,
                          atr_level= None,
                          use_sl=None, use_tp=None, back_candles= None, reward= None, 
                          use_exit_emaclose=None,
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
    
    if use_exit_emaclose is not None:
        EMAcrossover.use_exit_emaclose= use_exit_emaclose
    if atr_level is not None:
        EMAcrossover.ATR_LEVEL= atr_level
#     if optimization is not None: 
#         EMAcrossover.optimization= optimization
    bt = Backtest(data, EMAcrossover, cash=capital, margin=1/1, commission= fees)
    
    # Define the columns for the empty DataFrame
    columns = ['symbol','interval', 'short_ema', 'long_ema', 'trend_ema','exit_emaclose', 'sl_back_candles', 'reward', 'optimization',
              'Trend_opt','sl_opt', 'tp_opt']
    # Create an empty DataFrame with the specified columns
    input_df = pd.DataFrame(columns=columns)

    if optimization['optimization']:
        stats, heatmap= bt.optimize(
        SHORT_EMA= range(5, 30, 2) if optimization['short_ema'] else short_ema,
        LONG_EMA= range(10,45, 2) if optimization['long_ema'] else long_ema,
        TREND_EMA= range(50,200, 5) if optimization['trend_ema'] else trend_ema,
        BACK_CANDLES= range(10, 20, 1) if optimization['back_candles'] else back_candles,
        REWARD= (1, 1.1, 1.2, 1.5, 1.8, 2) if optimization['reward'] else reward,   
        ATR_LEVEL= 14,
        maximize= maximize,
        constraint = lambda param: 5 < (param.LONG_EMA - param.SHORT_EMA) < 50,
        max_tries=100,
        return_heatmap= True
        ) 
        print(f'\n summary bactesting result of {symbol}_{interval}\n')
        print(stats)
        
        # After obtaining the values, either from optimization results or direct parameters
        values = {
            'symbol': symbol,
            'interval':interval,
            'short_ema': stats._strategy.SHORT_EMA if optimization.get('short_ema') else short_ema,
            'long_ema': stats._strategy.LONG_EMA if optimization.get('long_ema') else long_ema,
            'trend_ema': False if not use_trend_ema else (stats._strategy.TREND_EMA if optimization.get('trend_ema') else trend_ema),
            'exit_emaclose':use_exit_emaclose,
            'sl_back_candles': False if not use_sl else (stats._strategy.BACK_CANDLES if optimization.get('back_candles') else back_candles),
            'reward': False if not use_tp else (stats._strategy.REWARD if optimization.get('reward') else reward), 
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
        'symbol': symbol,
        'interval':interval,
        'short_ema': short_ema,
        'long_ema': long_ema,
        'trend_ema': trend_ema if use_trend_ema else False,
        'exit_emaclose':use_exit_emaclose,
        'sl_back_candles': back_candles if use_sl else False,
        'reward': reward if use_tp else False,
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
    Trades= Trade_manage(Trades, capital)
    Trades['Duration']= Trades['Duration'].astype(str)
    summary_trades = summarize_trades(Trades)
    print(summary_trades)
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
    
    # Inside your function, after the optimization and result processing
    symbol_dir_trades = os.path.join('Results', symbol, interval, 'trades')
    symbol_dir_plots = os.path.join('Results', symbol, interval, 'plots')

    # Create the directories if they don't exist
    os.makedirs(symbol_dir_trades, exist_ok=True)
    os.makedirs(symbol_dir_plots, exist_ok=True)
    
    
    # Construct file name
    optimization_status = "Optimized" if row['optimization'] else "NotOptimized"
    # Adjust the file_name to include only the naming part, without the path
    file_name = f"EMA_{row['short_ema']}_{row['long_ema']}_{row['trend_ema']}_exitemac{row['exit_emaclose']}_sl{row['sl_back_candles']}_tp{row['reward']}_{optimization_status}.xlsx"

    # Now, when saving the Excel and plot files, use the full paths
    excel_file_path = os.path.join(symbol_dir_trades, file_name)
    plot_file_path = os.path.join(symbol_dir_plots, file_name.replace('.xlsx', '.png'))  # Assuming plot function saves as .png

    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        # Write each DataFrame to a different worksheet
        input_df.to_excel(writer, sheet_name='Input_setting')
        Trades.to_excel(writer, sheet_name='Trades')
        summary_trades.to_excel(writer, sheet_name='summary_trades')
        stat_df.to_excel(writer, sheet_name='results')
        if heatmap is not None:
            heatmap= heatmap.sort_values(ascending=False)
            heatmap= pd.DataFrame(heatmap)
            heatmap.to_excel(writer, sheet_name='heatmap')
            
    # When plotting, use the new plot_file_path
    bt.plot(resample=False, filename=plot_file_path)

    return input_df, stat_df, Trades,summary_trades, heatmap
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
