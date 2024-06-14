import warnings
import yfinance as yf
from alpaca_trade_api.rest import REST
from config import ENDPOINT_URL, KEY, SECRET


warnings.filterwarnings("ignore", message="Series.__getitem__")

ENDPOINT_URL = ENDPOINT_URL
KEY = KEY
SECRET = SECRET

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
