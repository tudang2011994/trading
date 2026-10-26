import pandas as pd

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    g = df.groupby('symbol',group_keys= False)

    # Return features
    df['return_1'] = g['close'].pct_change(1)

    df['return_3'] = g['close'].pct_change(3)

    df['return_6'] = g['close'].pct_change(6)

    df['return_12'] = g['close'].pct_change(12)

    return df

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Range pct
    df['range_pct'] = (
        (df['high'] - df['low']) / df['close']
    )

    # Body pct
    df['body_pct'] = (
        (df['open'] - df['close']).abs() / df['close']
    )

    # Upper wick pct
    df['upper_wick_pct'] = (
        (df['high'] - df[['open','close']].max(axis = 1))
        / df['close']
    )

    # Lower wick pct
    df['lower_wick_pct'] = (
        (df[['open','close']].min(axis = 1) - df['low'])
        / df['close']
    )

    return df


# Add Trending Feature SMA 
def add_treding_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    g = df.groupby('symbol', group_keys = False)

    # Add sma 20
    df['sma20'] = g['close'].transform( lambda x : x.rolling(20).mean())

    # Price vs sma 20
    df['price_vs_sma20'] = (
        df['close'] / df['sma20'] - 1
    )

    # Add ema 20
    df['ema20'] = g['close'].ewm(span = 20, adjust = False).mean()

    # Add ema spread 
    df['ema_spread'] = (
        df['ema20'] / df['sma20'] - 1
    )

    # Add rsi features
    df['price_change'] = g['close'].diff()
    df['gain'] = df['price_change'].clip(lower = 0)
    df['loss'] = (-df['price_change']).clip(lower = 0)
    df['avg_gain_14'] = df['gain'].rolling(14).mean()
    df['avg_loss_14'] = df['loss'].rolling(14).mean()
    rs = df["avg_gain_14"] / df["avg_loss_14"]
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Add mcid features
    df["ema12"] = g["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    df["ema12"] = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    df["macd"] = df["ema12"] - df["ema26"]
    
    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    return df

    # Add volatility features
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    g = df.groupby('symbol', group_keys = False)

    # Add Feature True Range
    previous_close = g["close"].shift(1)
    
    tr1 = df['high'] - df['low']
    tr2 = (previous_close - df['high']).abs()
    tr3 = (previous_close - df['low']).abs()

    df['true_range'] = pd.concat([tr1, tr2, tr3], axis= 1).max( axis = 1)

    df["atr_14"] = df["true_range"].ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    df["atr_pct"] = (
        df["atr_14"] / df["Close"]
    )

    # Add Volume Features
    df["volume_sma20"] = (
        g["Volume"].rolling(20).mean()
    )

    df["volume_ratio"] = (
        df["Volume"] / df["volume_sma20"]
    )

    return df
    


def add_features_engineering(df: pd.DataFrame):

    df = add_price_features(df)
    df = add_candle_features(df)
    df = add_treding_features(df)
    df = add_volatility_features(df)

    return df