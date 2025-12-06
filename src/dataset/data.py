# dataset/data.py
import pandas as pd
import numpy as np
import yfinance as yf
import os

class DataLoader:
    """Handles downloading, merging, and feature engineering for Gold + Macro data."""

    def __init__(self, start="2000-08-30", end="2025-01-01", interval="1d"):
        self.start = start
        self.end = end
        self.interval = interval
        self.cache_path = os.path.join(os.getcwd(), "yf_cache", "gold_macro_features.csv")

    def _calc_rsi(self, series, window=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        gain = up.rolling(window).mean()
        loss = down.rolling(window).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def fetch_data(self):
        """Downloads gold + macro data and returns engineered DataFrame."""
        print("📡 Downloading data from Yahoo Finance...")
        macro_raw = yf.download(
            ["DX=F", "^TNX", "CL=F", "^GSPC", "^VIX"],
            start=self.start, end=self.end, interval=self.interval, group_by="ticker"
        )

        macro = pd.DataFrame({
            "DollarIndex": macro_raw["DX=F"]["Close"],
            "US10Y": macro_raw["^TNX"]["Close"],
            "Oil": macro_raw["CL=F"]["Close"],
            "SP500": macro_raw["^GSPC"]["Close"],
            "VIX": macro_raw["^VIX"]["Close"]
        }).dropna()

        gold = yf.download("GC=F", start=self.start, end=self.end, interval=self.interval)[
            ["Open", "High", "Low", "Close", "Volume"]
        ]
        gold.index.name = "Date"
        gold = gold.sort_index()

        df = gold.merge(macro, left_index=True, right_index=True, how="inner").dropna()
        df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Volatility"] = df["LogReturn"].rolling(20).std()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA200"] = df["Close"].rolling(200).mean()
        df["Momentum_10"] = df["Close"] / df["MA20"] - 1
        df["Momentum_50"] = df["Close"] / df["MA200"] - 1
        df["ROC_4w"] = df["Close"].pct_change(20)
        df["DollarReturn"] = np.log(df["DollarIndex"] / df["DollarIndex"].shift(1))
        df["YieldChange"] = df["US10Y"].diff()
        df["RSI"] = self._calc_rsi(df["Close"])
        df["Gold_to_Dollar"] = df["Close"] / df["DollarIndex"]
        df["Gold_to_Yield"] = df["Close"] / (df["US10Y"] + 1e-6)
        df["Gold_to_SPX"] = df["Close"] / df["SP500"]
        df["Lag1"] = df["LogReturn"].shift(1)
        df["Lag5"] = df["LogReturn"].rolling(5).mean()
        df["Vol_5"] = df["LogReturn"].rolling(5).std()
        df["RSI_Change"] = df["RSI"].diff()
        df["DollarLag"] = df["DollarReturn"].shift(1)
        df["YieldLag"] = df["YieldChange"].shift(1)
        df["Future_Return_10d"] = np.log(df["Close"].shift(-10) / df["Close"])
        df["Target"] = (df["Future_Return_10d"] > 0).astype(int)
        df = df.dropna()

        print("✅ Data loaded. Shape:", df.shape)
        return df
