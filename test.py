import requests
import time
import sqlite3

# Path to your SQLite database file
db_path = 'trading_data.db'

# Establish connection
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables with necessary columns, including 'timestamp'
try:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            timeframe INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            indicator_name TEXT NOT NULL,
            value REAL,
            timeframe INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL,
            amount REAL,
            timestamp TEXT NOT NULL
        )
    ''')

    # Commit changes
    conn.commit()
    print("Database and tables initialized successfully.")

except Exception as e:
    print(f"Error creating tables: {e}")

# Close the connection
finally:
    conn.close()

"""
# Kraken API endpoint for ticker information
ticker_url = "https://api.kraken.com/0/public/Ticker"
ohlc_url = "https://api.kraken.com/0/public/OHLC"

# Get ticker information for all pairs
response = requests.get(ticker_url)
data = response.json()

# Extract volume and pairs
volumes = []
for pair, stats in data['result'].items():
    volume = float(stats['v'][1])  # Take the 24h volume
    volumes.append((pair, volume))

# Sort by volume in descending order and get top 50 tickers
volumes_sorted = sorted(volumes, key=lambda x: x[1], reverse=True)
top_50_tickers = volumes_sorted[:50]


# Function to fetch OHLC data for a specific pair and interval
def fetch_ohlc(pair, interval):
    ohlc_params = {
        'pair': pair,
        'interval': interval  # 1 = 1 minute, 5 = 5 minutes
    }
    response = requests.get(ohlc_url, params=ohlc_params)
    return response.json()


# Fetch OHLC data for 1-minute and 5-minute intervals for the top 50 tickers
for pair, volume in top_50_tickers:
    print(f"Fetching data for: {pair}")

    # Fetch 1-minute OHLC data
    ohlc_1min = fetch_ohlc(pair, 1)

    # Fetch 5-minute OHLC data
    ohlc_5min = fetch_ohlc(pair, 5)

    # Process or save the OHLC data
    print(f"1-min OHLC data for {pair}: {ohlc_1min}")
    print(f"5-min OHLC data for {pair}: {ohlc_5min}")

    # Pause between requests to avoid rate-limiting
    time.sleep(2)
"""