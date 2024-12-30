import logging
import requests
import krakenex
from functools import lru_cache
import pandas as pd


class MarketDataService:
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.api = krakenex.API()
        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=100)
    def get_ohlcv(self, symbol, interval, since=None):
        """Fetches OHLCV data with caching and rate limiting"""
        self.rate_limiter.wait()
        try:
            params = {'pair': symbol, 'interval': interval}
            if since:
                params['since'] = since

            response = self.api.query_public('OHLC', params)
            if response.get('error'):
                self.logger.error(f"Kraken API Error: {response['error']}")
                return None

            return response['result']
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {e}")
            return None

    def get_top_volume_pairs(self, limit=10):
        """Fetches top volume pairs with proper rate limiting"""
        self.rate_limiter.wait()
        try:
            response = self.api.query_public('AssetPairs')
            if response.get('error'):
                self.logger.error(f"Kraken API Error: {response['error']}")
                return []

            usd_pairs = [
                pair for pair, details in response['result'].items()
                if details.get('quote') == 'ZUSD'
            ]

            volumes = []
            batch_size = 20
            for i in range(0, len(usd_pairs), batch_size):
                self.rate_limiter.wait()
                batch = usd_pairs[i:i + batch_size]
                ticker_response = self.api.query_public('Ticker', {'pair': ','.join(batch)})

                if ticker_response.get('error'):
                    self.logger.error(f"Kraken API Error: {ticker_response['error']}")
                    continue

                for pair, stats in ticker_response['result'].items():
                    try:
                        volume = float(stats['v'][1])
                        volumes.append((pair, volume))
                    except (KeyError, ValueError) as e:
                        self.logger.warning(f"Error processing volume for {pair}: {e}")

            return [pair for pair, _ in sorted(volumes, key=lambda x: x[1], reverse=True)[:limit]]
        except Exception as e:
            self.logger.error(f"Error fetching top volume pairs: {e}")
            return []
