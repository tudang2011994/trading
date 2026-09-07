"""One daily directional prediction and trade execution for a single Schwab symbol.

Default operation is paper trading.  Set PAPER_TRADING=false only after supplying
SCHWAB_ACCESS_TOKEN and SCHWAB_ACCOUNT_HASH, reviewing Schwab's current API terms,
and testing the order path in a non-production account.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier


SYMBOL = os.getenv("TICKER", "SPY").upper()
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
INITIAL_CASH = float(os.getenv("INITIAL_PAPER_BALANCE", "100000"))
TRADE_DOLLARS = float(os.getenv("TRADE_DOLLARS", "1000"))
LOG_PATH = Path(os.getenv("TRADE_LOG_PATH", "trades.log"))
STATE_PATH = Path(os.getenv("PAPER_STATE_PATH", "paper_state.json"))
SCHWAB_BASE_URL = "https://api.schwabapi.com"

LOG_COLUMNS = [
    "timestamp", "symbol", "action", "price", "quantity", "cash",
    "position_shares", "portfolio_value", "prediction", "mode",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class SchwabClient:
    """Minimal Schwab REST client using an access token supplied by the operator."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("SCHWAB_API_KEY", "")
        self.secret = os.environ.get("SCHWAB_SECRET", "")
        self.access_token = os.environ.get("SCHWAB_ACCESS_TOKEN", "")
        self.session = requests.Session()
        if not self.access_token:
            refresh_token = os.environ.get("SCHWAB_REFRESH_TOKEN", "")
            if not (self.api_key and self.secret and refresh_token):
                raise RuntimeError(
                    "Set SCHWAB_ACCESS_TOKEN, or set SCHWAB_API_KEY, SCHWAB_SECRET, "
                    "and SCHWAB_REFRESH_TOKEN from Schwab's approved OAuth flow."
                )
            token_response = self.session.post(
                f"{SCHWAB_BASE_URL}/v1/oauth/token",
                auth=(self.api_key, self.secret),
                data={"grant_type": "refresh_token", "refresh_token": refresh_token},
                timeout=30,
            )
            token_response.raise_for_status()
            self.access_token = token_response.json()["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})

    def price_history(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        response = self.session.get(
            f"{SCHWAB_BASE_URL}/marketdata/v1/pricehistory",
            params={
                "symbol": symbol,
                "periodType": "year",
                "period": 1,
                "frequencyType": "daily",
                "frequency": 1,
                "startDate": int(start.timestamp() * 1000),
                "endDate": int(end.timestamp() * 1000),
            },
            timeout=30,
        )
        response.raise_for_status()
        candles = response.json().get("candles", [])
        if not candles:
            raise RuntimeError(f"Schwab returned no candle data for {symbol}.")
        frame = pd.DataFrame(candles)
        frame["datetime"] = pd.to_datetime(frame["datetime"], unit="ms", utc=True)
        return frame.sort_values("datetime").reset_index(drop=True)

    def get_account(self, account_hash: str) -> dict:
        response = self.session.get(
            f"{SCHWAB_BASE_URL}/trader/v1/accounts/{account_hash}",
            params={"fields": "positions"}, timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def place_market_order(self, account_hash: str, symbol: str, instruction: str, quantity: int) -> None:
        order = {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {"symbol": symbol, "assetType": "EQUITY"},
            }],
        }
        response = self.session.post(
            f"{SCHWAB_BASE_URL}/trader/v1/accounts/{account_hash}/orders",
            json=order, timeout=30,
        )
        response.raise_for_status()

    def close(self) -> None:
        self.session.close()


def add_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI(14), SMA(20), and MACD(12, 26, 9) from closing prices."""
    result = candles.copy()
    close = result["close"].astype(float)
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    result["rsi"] = 100 - (100 / (1 + rs))
    result["sma"] = close.rolling(20).mean()
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    result["macd"] = fast - slow
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["target"] = (close.shift(-1) > close).astype(int)
    return result


def predict_direction(candles: pd.DataFrame) -> tuple[str, float]:
    data = add_indicators(candles)
    features = ["rsi", "sma", "macd", "macd_signal"]
    # Exclude the final row because its next-day target is not known.
    training = data.iloc[:-1].dropna(subset=features + ["target"])
    latest = data.iloc[[-1]].dropna(subset=features)
    if len(training) < 60 or latest.empty or training["target"].nunique() < 2:
        logger.warning("Insufficient usable training data; choosing HOLD.")
        return "HOLD", 0.5
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42, n_jobs=1,
    )
    model.fit(training[features], training["target"])
    probability_up = float(model.predict_proba(latest[features])[0, 1])
    # A neutral band avoids trading weak model opinions.
    if probability_up >= 0.55:
        return "BUY", probability_up
    if probability_up <= 0.45:
        return "SELL", probability_up
    return "HOLD", probability_up


def load_paper_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"cash": INITIAL_CASH, "shares": 0}


def save_paper_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_log(record: dict) -> None:
    needs_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
    with LOG_PATH.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(record)


def paper_trade(action: str, price: float, prediction: float) -> dict:
    state = load_paper_state()
    shares = int(state["shares"])
    cash = float(state["cash"])
    quantity = 0
    executed_action = "HOLD"
    if action == "BUY":
        quantity = min(int(TRADE_DOLLARS // price), int(cash // price))
        if quantity > 0:
            cash -= quantity * price
            shares += quantity
            executed_action = "BUY"
    elif action == "SELL" and shares > 0:
        quantity = shares
        cash += quantity * price
        shares = 0
        executed_action = "SELL"
    state = {"cash": round(cash, 2), "shares": shares}
    save_paper_state(state)
    return make_record(executed_action, price, quantity, cash, shares, prediction, "paper")


def live_trade(client: SchwabClient, action: str, price: float, prediction: float) -> dict:
    account_hash = os.environ.get("SCHWAB_ACCOUNT_HASH", "")
    if not account_hash:
        raise RuntimeError("SCHWAB_ACCOUNT_HASH is required for live trading.")
    account = client.get_account(account_hash)
    balances = account.get("securitiesAccount", {}).get("currentBalances", {})
    positions = account.get("securitiesAccount", {}).get("positions", [])
    existing = next((p for p in positions if p.get("instrument", {}).get("symbol") == SYMBOL), None)
    shares = int(float(existing.get("longQuantity", 0))) if existing else 0
    quantity = 0
    executed_action = "HOLD"
    if action == "BUY":
        available = float(balances.get("availableFunds", 0))
        quantity = min(int(TRADE_DOLLARS // price), int(available // price))
        if quantity > 0:
            client.place_market_order(account_hash, SYMBOL, "BUY", quantity)
            shares += quantity
            executed_action = "BUY"
    elif action == "SELL" and shares > 0:
        quantity = shares
        client.place_market_order(account_hash, SYMBOL, "SELL", quantity)
        shares = 0
        executed_action = "SELL"
    cash = float(balances.get("cashBalance", 0))
    portfolio = float(balances.get("liquidationValue", cash + shares * price))
    return make_record(executed_action, price, quantity, cash, shares, prediction, "live", portfolio)


def make_record(action: str, price: float, quantity: int, cash: float, shares: int,
                prediction: float, mode: str, portfolio: float | None = None) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": SYMBOL,
        "action": action, "price": f"{price:.2f}", "quantity": quantity,
        "cash": f"{cash:.2f}", "position_shares": shares,
        "portfolio_value": f"{portfolio if portfolio is not None else cash + shares * price:.2f}",
        "prediction": f"{prediction:.4f}", "mode": mode,
    }


def run() -> None:
    client: SchwabClient | None = None
    try:
        client = SchwabClient()
        end = datetime.now(timezone.utc)
        candles = client.price_history(SYMBOL, end - timedelta(days=400), end)
        price = float(candles.iloc[-1]["close"])
        action, prediction = predict_direction(candles)
        record = paper_trade(action, price, prediction) if PAPER_TRADING else live_trade(client, action, price, prediction)
        append_log(record)
        logger.info("Logged %s %s at $%.2f (up probability %.2f%%)", record["action"], SYMBOL, price, prediction * 100)
    finally:
        if client is not None:
            client.close()
            logger.info("Schwab HTTP session closed.")


if __name__ == "__main__":
    run()
