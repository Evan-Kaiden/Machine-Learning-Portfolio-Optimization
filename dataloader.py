import pandas as pd
import yfinance as yf
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

tickers = pd.read_csv('data/tick.csv')['Ticker']


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