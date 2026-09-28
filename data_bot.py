import yfinance as yf
import pandas
from datetime import datetime
from pathlib import Path
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
import os


BASE_PATH = Path.cwd()
DATA_PATH = BASE_PATH / 'data'
 
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')

client = StockHistoricalDataClient(
    api_key= API_KEY, 
    secret_key=SECRET_KEY
    )


def is_trading_day(date) -> bool:
    #return  date > 5
    return True

def get_symbols() -> list[str]:
    return ['AAPL','TSLA']

def get_latest_symbol_data(symbol) -> None:

    #df = yf.download(symbol, period ="1d", interval="5m")
    #df.to_parquet(f'{symbol}_1d.parquet')
    return 0


def get_symbol_data_alpaca(symbol) -> None:
    
    request = StockBarsRequest(
        symbol_or_symbols= symbol,
        timeframe= TimeFrame(
            5,
            TimeFrameUnit.Minute
        ),
    )

    bar = client.get_stock_bars(request)

    bar.df.to_parquet(f'{DATA_PATH/symbol}.parquet')
    


def main():
    current_date = datetime.today()

    # Check is there trading day today
    if not is_trading_day(current_date.weekday()):
        return 0
    
    # List all symbols
    symbols = get_symbols()

    # Get new symbols data
    for symbol in symbols:
        get_symbol_data_alpaca(symbol)
    return 0

if __name__ == "__main__":
    main()