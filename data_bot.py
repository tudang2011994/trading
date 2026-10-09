import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
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

def get_latest_symbol_data(symbols) -> None:

    df = yf.download(symbols, period ="1d", interval="5m")
    df.to_parquet(f'{symbols}_1d.parquet')


    return 0


def download_symbols_data_alpaca(symbols,start_time, end_time) -> None:
    
    # Build request object
    request = StockBarsRequest(
        symbol_or_symbols= symbols,
        timeframe= TimeFrame(
            5,
            TimeFrameUnit.Minute
        ),
        start = start_time,
        end_time = end_time,
        feed='iex'
    )

    # Fetch data from alpaca
    try:
        bar = client.get_stock_bars(request)
        df = bar.df
    except:
        print("error")

    # Data cleaning
    df = df.reset_index()

    # Identical name of columns
    df = df.rename(
        columns={
            "symbol": "symbol",
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "trade_count": "trade_count",
            "vwap": "vwap",
        }
    )

    # Store latest stock data
    output_file = DATA_PATH/f'data.parquet'
    df.to_parquet(output_file)
    print(f'Download completed data',bar.df.shape)

    return df

# Clean raw data get from alpaca api
def clean_symbols_data(df: pd.DataFrame ) -> pd.DataFrame:

    # Copy new dataframe
    df = df.copy()

    print(f'Number of empty value {df.isna().sum()}')
    print(f'Number otal dupplicate value {df.duplicated().sum()}')

    df.drop_duplicates()

    return df

# Get Features Engineering 
def apply_features_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    

    return df
    


def main():
    current_date = datetime.today()

    # Check is there trading day today
    if not is_trading_day(current_date.weekday()):
        return 0
    
    # List all symbols
    symbols = get_symbols()


    start_time = datetime.now() - timedelta(days=10)
    end_time = datetime.now()
    # Get new symbols data
    df = download_symbols_data_alpaca(symbols, start_time, end_time)

    # Clean raw data 
    df = clean_symbols_data(df)

    # Features Engineering
    df = apply_features_engineering(df)

    return 0

if __name__ == "__main__":
    main()