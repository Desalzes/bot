# data_manager.py

import os
import json
import requests
import logging
import pandas as pd
import time
import ta  # Ensure this library is installed: pip install ta


class DataManager:
    def __init__(self, parameter_agent, verbose=True):
        self.parameter_agent = parameter_agent
        self.verbose = verbose
        self.data_dir = 'data/historical_data'
        self.data = {}

    def get_top_volume_tickers(self, limit=50):
        """
        Fetches the top 'limit' tickers from Kraken sorted by 24-hour trading volume.
        """
        try:
            url = "https://api.kraken.com/0/public/AssetPairs"
            response = requests.get(url)
            data = response.json()

            if data['error']:
                logging.error(f"Kraken API Error: {data['error']}")
                return []

            asset_pairs = data['result']
            volumes = []
            pair_names = list(asset_pairs.keys())
            batch_size = 20

            for i in range(0, len(pair_names), batch_size):
                batch_pairs = pair_names[i:i + batch_size]
                batch_pairs_str = ','.join(batch_pairs)

                ticker_url = "https://api.kraken.com/0/public/Ticker"
                params = {'pair': batch_pairs_str}
                response = requests.get(ticker_url, params=params)
                ticker_data = response.json()

                if ticker_data['error']:
                    logging.error(f"Kraken API Error: {ticker_data['error']}")
                    continue

                for pair, stats in ticker_data['result'].items():
                    volume = float(stats['v'][1])
                    volumes.append((pair, volume))

                time.sleep(1)

            volumes_sorted = sorted(volumes, key=lambda x: x[1], reverse=True)
            top_ticker_symbols = [self.map_kraken_to_symbol(pair) for pair, _ in volumes_sorted[:limit]]
            return top_ticker_symbols

        except Exception as e:
            logging.exception(f"An error occurred while fetching top volume tickers: {e}")
            return []

    def map_kraken_to_symbol(self, kraken_pair):
        """
        Maps Kraken's asset pair names to standard symbols.
        """
        reverse_mapping = {
            'XXBTZUSD': 'BTCUSD',
            'XETHZUSD': 'ETHUSD',
            # Add other mappings as needed
        }
        return reverse_mapping.get(kraken_pair, kraken_pair)

    def fetch_ohlcv(self, symbol, interval=1, since=None):
        """
        Fetches OHLCV data for a given symbol from Kraken.
        """
        try:
            kraken_pair = self.map_symbol_to_kraken(symbol)
            url = "https://api.kraken.com/0/public/OHLC"
            params = {'pair': kraken_pair, 'interval': interval}
            if since:
                params['since'] = since

            response = requests.get(url, params=params)
            data = response.json()

            if data['error']:
                logging.error(f"Kraken API Error for pair {kraken_pair}: {data['error']}")
                return pd.DataFrame()

            result_pair = list(data['result'].keys())[0]
            ohlcv = data['result'][result_pair]
            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df[['open', 'high', 'low', 'close', 'vwap', 'volume']] = df[['open', 'high', 'low', 'close', 'vwap', 'volume']].astype(float)
            return df

        except Exception as e:
            logging.exception(f"Failed to fetch OHLCV data for {symbol}: {e}")
            return pd.DataFrame()

    def map_symbol_to_kraken(self, symbol):
        symbol_mapping = {
            'BTCUSD': 'XXBTZUSD',
            'ETHUSD': 'XETHZUSD',
            # Add other mappings as needed
        }
        return symbol_mapping.get(symbol, symbol)

    def fetch_and_save_data(self, symbol, interval):
        """
        Fetches OHLCV data for a symbol and interval and saves it to a JSON file.
        Returns True if successful, False otherwise.
        """
        try:
            ohlc_data = self.fetch_ohlcv(symbol, interval)
            if ohlc_data.empty:
                logging.error(f"Failed to fetch data for {symbol} at {interval}m interval.")
                return False

            symbol_dir = os.path.join(self.data_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            file_path = os.path.join(symbol_dir, f"{interval}m.json")

            ohlc_data.reset_index(inplace=True)
            ohlc_data['time'] = ohlc_data['time'].astype(str)
            data_to_save = ohlc_data.to_dict(orient='records')
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            logging.info(f"Fetched and saved data for {symbol} at {interval}m interval to {file_path}.")
            return True
        except Exception as e:
            logging.exception(f"Failed to fetch and save data for {symbol} at {interval}m interval: {e}")
            return False

    def get_data(self, symbol, interval):
        key = (symbol, interval)
        if key not in self.data:
            file_path = os.path.join(self.data_dir, symbol, f"{interval}m.json")
            if not os.path.exists(file_path):
                logging.warning(f"Data file {file_path} not found. Fetching data...")
                success = self.fetch_and_save_data(symbol, interval)
                if not success:
                    logging.error(f"Failed to fetch data for {symbol} at {interval}m interval.")
                    return pd.DataFrame()

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                df = self.compute_technical_indicators(df)
                self.data[key] = df
                if self.verbose:
                    logging.info(f"Loaded data for {symbol} at {interval}m interval.")
            except Exception as e:
                logging.exception(f"Failed to load or process data from {file_path}: {e}")
                return pd.DataFrame()
        return self.data[key]

    def compute_technical_indicators(self, df):
        try:
            rsi = ta.momentum.RSIIndicator(close=df['close'], window=14)
            df['RSI'] = rsi.rsi()

            sma5 = ta.trend.SMAIndicator(close=df['close'], window=5)
            df['SMA_5'] = sma5.sma_indicator()
            sma10 = ta.trend.SMAIndicator(close=df['close'], window=10)
            df['SMA_10'] = sma10.sma_indicator()

            ema5 = ta.trend.EMAIndicator(close=df['close'], window=5)
            df['EMA_5'] = ema5.ema_indicator()
            ema10 = ta.trend.EMAIndicator(close=df['close'], window=10)
            df['EMA_10'] = ema10.ema_indicator()

            macd = ta.trend.MACD(close=df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()

            bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
            df['Bollinger_upper'] = bollinger.bollinger_hband()
            df['Bollinger_middle'] = bollinger.bollinger_mavg()
            df['Bollinger_lower'] = bollinger.bollinger_lband()

            atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['ATR'] = atr.average_true_range()
            return df
        except Exception as e:
            logging.exception(f"Failed to compute technical indicators: {e}")
            return df