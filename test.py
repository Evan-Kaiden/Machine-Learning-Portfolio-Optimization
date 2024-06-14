import pandas as pd
import yfinance as yf
import numpy as np

from tqdm import tqdm
import warnings

import os
import joblib
from keras.models import load_model

from pypfopt.hierarchical_portfolio import HRPOpt

from alpaca_trade_api.rest import REST

warnings.filterwarnings("ignore", message="Series.__getitem__")
warnings.filterwarnings("ignore", message="The default fill_method='pad' in DataFrame.pct_change is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message ="max_sharpe transforms the optimization problem so additional objectives may not work as expected", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

tickers = pd.read_csv('data/tick.csv')['Ticker']

SECRET = 'Iy7JxSTLqhSuxDBRNGynOUzoupNggBbVZ9U4Mbim'
KEY = 'PK8T7NEZ2OWMID429A0E'
ENDPOINT_URL = 'https://paper-api.alpaca.markets'

def calculate_rsi(data, period):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]  

def calculate_macd_histogram(data, short_period=12, long_period=26, signal_period=9):
    ema_short = data['Close'].ewm(span=short_period, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_period, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_histogram.iloc[-1]

def calculate_rvi(data, period):
    close_open_diff = data['Close'] - data['Open']
    high_low_diff = data['High'] - data['Low']
    rvi = (close_open_diff.rolling(window=period).mean() / high_low_diff.rolling(window=period).mean()).iloc[-1]
    return rvi

def calculate_roc(data, period):
    roc = ((data['Close'].iloc[-1] - data['Close'].iloc[-period]) / data['Close'].iloc[-period]) * 100
    return roc


def calculate_atr(data, period):
    true_range = data['High'] - data['Low']
    true_range = true_range.combine(data['High'] - data['Close'].shift(), max)
    true_range = true_range.combine(data['Close'].shift() - data['Low'], max)
    atr = true_range.rolling(window=period).mean()
    return atr.iloc[-1]

def calculate_rolling_mean_price(data, period):
    rolling_mean_price_change = data['Close'].rolling(window=period).mean()
    return rolling_mean_price_change.iloc[-1]

def calculate_rolling_mean_volume(data, period):
    rolling_mean_volume = data['Volume'].rolling(window=period).mean()
    return rolling_mean_volume.iloc[-1]


def strip(df, is_bs = False):
    if not is_bs:
        df.index = df.index.map(lambda x: x.strip() if isinstance(x, str) else x)
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x) 
        df.set_index(df.name, inplace=True)
        df = df.drop(['name', 'ttm'], axis=1).T
        df = df.applymap(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)
        df.astype(float)
    else:
        df.index = df.index.map(lambda x: x.strip() if isinstance(x, str) else x)
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x) 
        df.set_index(df.name, inplace=True)
        df = df.drop(['name'], axis=1).T
        df = df.applymap(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)
        df.astype(float)
    return df
def calculate_rsi(data, period):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]  

def calculate_macd_histogram(data, short_period=12, long_period=26, signal_period=9):
    ema_short = data['Close'].ewm(span=short_period, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_period, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_histogram.iloc[-1]

def calculate_rvi(data, period):
    close_open_diff = data['Close'] - data['Open']
    high_low_diff = data['High'] - data['Low']
    rvi = (close_open_diff.rolling(window=period).mean() / high_low_diff.rolling(window=period).mean()).iloc[-1]
    return rvi

def calculate_roc(data, period):
    roc = ((data['Close'].iloc[-1] - data['Close'].iloc[-period]) / data['Close'].iloc[-period]) * 100
    return roc


def calculate_atr(data, period):
    true_range = data['High'] - data['Low']
    true_range = true_range.combine(data['High'] - data['Close'].shift(), max)
    true_range = true_range.combine(data['Close'].shift() - data['Low'], max)
    atr = true_range.rolling(window=period).mean()
    return atr.iloc[-1]

def calculate_rolling_mean_price(data, period):
    rolling_mean_price_change = data['Close'].rolling(window=period).mean()
    return rolling_mean_price_change.iloc[-1]

def calculate_rolling_mean_volume(data, period):
    rolling_mean_volume = data['Volume'].rolling(window=period).mean()
    return rolling_mean_volume.iloc[-1]

def load_data(n, BS, CF, IS, ticker, Date):

    returnOnAssets = IS['Net Income'] / BS['Total Assets']
    returnOnEquity = IS['Net Income'] / BS['Stockholders Equity']
    cashRatio = BS['Cash And Cash Equivalents'] / BS['Current Liabilities']
    returnOnCapitalEmployed = IS['EBIT'] / (BS['Total Assets'] - BS['Current Liabilities'])
    equityRatio =  BS['Stockholders Equity'] / BS['Total Assets']
    netProfitMargin = IS['Net Income'] / IS['Total Revenue']
    OperatingProfitMargin = IS['Operating Income'] / IS['Total Revenue']
    grossProfitMargin = (IS['Total Revenue'] - IS['Cost Of Revenue']) / IS['Total Revenue']
    debtRatio = BS['Total Debt'] / BS['Total Assets']

    start = pd.to_datetime(Date[n])

    IncomeDelta = IS['Net Income'].values[n]/IS['Net Income'].values[n + 1]   
    AssetsDelta = BS['Total Assets'].values[n]/BS['Total Assets'].values[n + 1]
    GrossMarginDelta = (IS['Gross Profit'].values[n] / IS['Total Revenue'].values[n]) / (IS['Gross Profit'].values[n + 1] / IS['Total Revenue'].values[n + 1])
    accruals = CF['Operating Cash Flow'].values[n] / BS['Total Assets'].values[n] - returnOnAssets.values[n]

    stock_data = yf.download(ticker, start=(start - pd.Timedelta(weeks=65)), end=start, progress=False, show_errors=False)

    ma_30 = calculate_rolling_mean_price(stock_data, period=30)
    ma_60 = calculate_rolling_mean_price(stock_data, period=60)
    ma_120 = calculate_rolling_mean_price(stock_data, period=120)
    ma_240 = calculate_rolling_mean_price(stock_data, period=240)

    va_30 = calculate_rolling_mean_volume(stock_data, period=30)
    va_60 = calculate_rolling_mean_volume(stock_data, period=60)
    va_120 = calculate_rolling_mean_volume(stock_data, period=120)
    va_240 = calculate_rolling_mean_volume(stock_data, period=240)

    macd_histogram_short = calculate_macd_histogram(stock_data, short_period=12, long_period=26, signal_period=9)
    macd_histogram_medium = calculate_macd_histogram(stock_data, short_period=24, long_period=52, signal_period=18)
    macd_histogram_long = calculate_macd_histogram(stock_data, short_period=48, long_period=104, signal_period=36)
    macd_hisstogram_longest = calculate_macd_histogram(stock_data, short_period=144, long_period=312, signal_period=108)

    rsi_30 = calculate_rsi(stock_data, period=30)
    rsi_60 = calculate_rsi(stock_data, period=60)
    rsi_120 = calculate_rsi(stock_data, period=120)
    rsi_240 = calculate_rsi(stock_data, period=240)

    rvi_30 = calculate_rvi(stock_data, period=30)
    rvi_60 = calculate_rvi(stock_data, period=60)
    rvi_120 = calculate_rvi(stock_data, period=120)
    rvi_240 = calculate_rvi(stock_data, period=240)

    roc_30 = calculate_roc(stock_data, period=30)
    roc_60 = calculate_roc(stock_data, period=60)
    roc_120 = calculate_roc(stock_data, period=120)
    roc_240 = calculate_roc(stock_data, period=240)

    atr_30 = calculate_atr(stock_data, period=30)
    atr_60 = calculate_atr(stock_data, period=60)
    atr_120 = calculate_atr(stock_data, period=120)
    atr_240 = calculate_atr(stock_data, period=240)

    data =  [returnOnAssets.values[n], returnOnEquity.values[n], cashRatio.values[n],
        returnOnCapitalEmployed.values[n], equityRatio.values[n],
        netProfitMargin.values[n], OperatingProfitMargin.values[n], grossProfitMargin.values[n],
        debtRatio.values[n], IncomeDelta, AssetsDelta, GrossMarginDelta, accruals,
        rsi_30, rsi_60, rsi_120, rsi_240, macd_histogram_short, macd_histogram_medium, macd_histogram_long, macd_hisstogram_longest,
        rvi_30, rvi_60, rvi_120, rvi_240, roc_30, roc_60, roc_120, roc_240,  atr_30, atr_60, atr_120, atr_240, ma_30, ma_60, ma_120, ma_240,
        va_30, va_60, va_120, va_240, Date[n]]

    return data


def return_data():
    data = {}
    n_total = 0


    for ticker in tqdm(tickers):
        try:
            n_total += 1

            ticker_obj = yf.Ticker(ticker)
            BS = ticker_obj.quarterly_balance_sheet.T
            IS = ticker_obj.quarterly_financials.T
            CF = ticker_obj.quarterly_cashflow.T
            Date = BS.index

            stock_data = load_data(0, BS, CF, IS, ticker, Date)

            data[ticker] = stock_data
        except: continue




    df = pd.DataFrame(data).T
    df.columns = [
            'returnOnAssets', 'returnOnEquity', 'cashRatio',  'returnOnCapitalEmployed', 'equityRatio','netProfitMargin', 'OperatingProfitMargin','grossProfitMargin','debtRatio', 'IncomeDelta', 'AssetsDelta', 'GrossMarginDelta', 'accruals', 
            'rsi_30', 'rsi_60', 'rsi_120', 'rsi_240', 'macd_histogram_short', 'macd_histogram_medium', 'macd_histogram_long', 'macd_histogram_longest',
            'rvi_30', 'rvi_60', 'rvi_120', 'rvi_240', 'roc_30', 'roc_60', 'roc_120', 'roc_240',  'atr_30', 'atr_60', 'atr_120', 'atr_240', 'ma_30', 'ma_60', 'ma_120', 'ma_240',
            'va_30', 'va_60', 'va_120', 'va_240', 'date'
                            ]
    df = df.dropna()
    
    print(f'{df.shape[0]} succesful out of {n_total} total: {df.shape[0]/n_total}% success rate')

    return df


def load_stocks(Data, keras_model):
    pred = keras_model.predict(Data)
    pred = pred.flatten()
    pred = (pred > 0.75).astype(int)
    cleaned = Data[pred == 1]
    return cleaned.index


def clean(current_date):
    # Load Encoder Models
    model_names = os.listdir('encoding_models')
    models = [load_model('encoding_models/' + name) for name in model_names if name != '.DS_Store']

    # Load Neural Network for predictions
    keras_model = load_model('predictive_models/fnnwa.h5')

    # Load Data Scaler
    scaler = joblib.load('predictive_models/data_scaler.joblib')

    # Get Data
    data = return_data()

    # Define Encoder Model Dict
    models_dict = {'RSI' : [models[4], ['rsi_30', 'rsi_60', 'rsi_120', 'rsi_240']], 
            'MACD' : [models[3],['macd_histogram_short', 'macd_histogram_medium', 'macd_histogram_long', 'macd_histogram_longest']], 
            'RVI' : [models[6], ['rvi_30', 'rvi_60', 'rvi_120', 'rvi_240']], 
            'ROC' : [models[0], ['roc_30', 'roc_60', 'roc_120', 'roc_240']], 
            'ATR' :[models[5], ['atr_30', 'atr_60', 'atr_120', 'atr_240']], 
            'MA' :[models[7], ['ma_30', 'ma_60', 'ma_120', 'ma_240']], 
            'VA' : [models[1], ['va_30', 'va_60', 'va_120', 'va_240']],
            'F' : [models[8], ['returnOnAssets', 'returnOnEquity', 'cashRatio', 'returnOnCapitalEmployed', 'equityRatio', 'netProfitMargin', 'OperatingProfitMargin', 'grossProfitMargin', 'debtRatio', 'accruals']],
            'FD' : [models[2], ['IncomeDelta', 'AssetsDelta', 'GrossMarginDelta']]}

    # Define Data that the model will see
    data['date'] = pd.to_datetime(data['date'])

    # Ensure the current_date is also in datetime format
    current_date = pd.to_datetime(current_date)
    # Filter the DataFrame based on the date and drop the 'date' column
    model_data = data[data['date'] == current_date].drop('date', axis=1)
    print(model_data.shape)
    # Scale Data
    scaled_data = pd.DataFrame(
        scaler.transform(model_data),
        columns = model_data.columns
    )

    # Encode Data
    encoded_data = []
    for value in models_dict.values():
        encoded_data.append(value[0].predict(scaled_data[value[1]]).flatten())

    encoded_df = pd.DataFrame(data = np.array(encoded_data).T, columns=models_dict.keys())

    df = pd.concat([encoded_df, scaled_data], axis=1)

    df = df.set_index(data.index)

    # Stock Selection 
    stocks = load_stocks(df, keras_model)

    return stocks



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



api = REST(key_id=KEY,secret_key=SECRET,base_url=ENDPOINT_URL)

def get_portfolio_value():
    account = api.get_account()
    return float(account.equity)

def calc_shares(weights_series):
    shares = []

    portfolio_value = get_portfolio_value()

    for stock, weight in weights_series.items():
        ticker = yf.Ticker(stock)
        share_price = ticker.history(period='1d')['Close'][0]
        shares.append([stock, round((portfolio_value*weight)/share_price, 2)])
    return shares

def enter_positions(shares):
    for ticker, qty in shares:
        api.submit_order(symbol=ticker, qty=qty, side='buy', time_in_force='day')

def close_positions():
    positions = api.list_positions()
    for position in positions:
        if float(position.qty) > 0:
            api.close_position(symbol=position.symbol)



def main():
    dates = pd.read_csv('dates.csv')
    date = dates.iloc[0].values

    dates = dates.iloc[1:]
    dates.to_csv('dates.csv', index=False)

    close_positions()

    weight_dist = get_weights(date[0], date[2], date[1])
    share_dist = calc_shares(weight_dist)
    print(share_dist)
    enter_positions(share_dist)

if __name__ == '__main__':
    main()