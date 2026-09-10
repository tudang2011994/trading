import yfinance
import pandas
from datetime import datetime

def is_trading_day() -> bool:
    return  datetime.today().weekday() > 5

def get_symbols():
    return 0

def get_latest_symbol_data(symbol):
    return 0

def remove_old_data():
    return 0

def main():
    is_trading_day()
    symbols = get_symbols()
    for symbol in symbols:
        get_latest_symbol_data(symbol)
        remove_old_data()
    return 0

if __name__ == "__main__":
    main()