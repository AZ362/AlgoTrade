#%matplotlib inline
import quantstats as qs
from Utils import *
# Usage
downloader = MT5DataDownloader()
data = downloader.get_data(symbol, interval, days_back)# Usage
print(data)

# Calculate the returns for OHLC columns
returns = data[['open', 'high', 'low', 'close']].pct_change().dropna()
returns_descriptive_stats = returns.describe()

print('------------------------------------------------------------------------------------')
print('returns descriptive stats: ')
print(returns_descriptive_stats)


# Visualize the last 'x' candles for returns
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

for i, col in enumerate(['open', 'high', 'low', 'close']):
    axs[i].plot(returns[col], label='{} Return'.format(col))
    axs[i].legend(loc='upper left')

fig.suptitle('{} Candles Returns'.format(symbol))
plt.show()


# Generate various plots (example: cumulative returns and drawdown)
print(qs.plots.snapshot(returns['close'], title='Benchmark Performance Snapshot'))
# Generate a full report
# report_name = f'Benchmark_report_{symbol}_{interval}.html'#
# # Generate a full report
# qs.reports.html(returns['close'], output=report_name, title=f'Benchmark Analysis Report - {symbol}, {interval}')
# default_report_name = 'quantstats-tearsheet.html'
# if os.path.exists(default_report_name):
#     os.rename(default_report_name, report_name)
