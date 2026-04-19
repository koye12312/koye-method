"""Step 1 of the Koye Method pipeline: download TSLA hourly OHLCV bars."""
from core.data_fetcher import download_hourly

TICKER = "TSLA"
PERIOD = "2y"   # Yahoo allows up to ~730 days of hourly history

if __name__ == "__main__":
    df = download_hourly(TICKER, period=PERIOD)
    print(f"\nFirst rows of {TICKER}:")
    print(df.head(3))
    print(f"\nLast rows of {TICKER}:")
    print(df.tail(3))
    print(f"\nNext step: python 02_label_regimes.py")
