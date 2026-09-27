import yfinance as yf
import pandas
from datetime import datetime

def is_trading_day(date) -> bool:
    #return  date > 5
    return True

def get_symbols() -> list[str]:
    return ['AAPL','TSLA']

def get_latest_symbol_data(symbol):
    
    df = yf.download(symbol, period ="1d", interval="5m")
    df.to_parquet(f'{symbol}_1d.parquet')


def main():
    current_date = datetime.today()

    # Check is there trading day today
    if not is_trading_day(current_date.weekday()):
        return 0
    
    # List all symbols
    symbols = get_symbols()

    # Get new symbols data
    for symbol in symbols:
        get_latest_symbol_data(symbol)
    return 0

if __name__ == "__main__":
    main()