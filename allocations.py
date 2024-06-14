import yfinance as yf
import pandas as pd
from pypfopt.hierarchical_portfolio import HRPOpt
import warnings

warnings.filterwarnings("ignore", message="The default fill_method='pad' in DataFrame.pct_change is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message ="max_sharpe transforms the optimization problem so additional objectives may not work as expected", category=UserWarning)


from predictions import clean

def opt(tickers, start_date, end_date):
    stock_data = get_data(tickers, start_date=start_date, end_date=end_date)
    
    hrp = HRPOpt(returns=stock_data)
    weights = hrp.optimize()
    weights_series = pd.Series(weights)
    weights_series = weights_series[weights_series != 0]
    return weights_series


def get_data(tickers, start_date, end_date):
    data = yf.download(tickers=tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    return data


def get_series(tickers, start_date, end_date):
    weights_series = opt(tickers, start_date=start_date, end_date=end_date)
    return weights_series

def get_weights(release_date, start_date, end_date):
    tickers = clean(release_date)
    return get_series(list(tickers), start_date, end_date)


