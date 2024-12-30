import os
import asyncio
import logging
from datetime import datetime
import signal
import sys
import pandas as pd
import requests

from agents.database_agent import DatabaseAgent
from rate_limiter import RateLimiter
from market_data_service import MarketDataService
from strategy_manager import StrategyManager
from trading_stats import TradingStats
from agents.parameter_agent import ParameterAgent
from agents.trading_bot_agent import TradingBotAgent
from indicators import Indicators


def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_bot.log'),
            logging.StreamHandler()
        ]
    )


async def execute_trades(trading_bot, market_data, strategy, trading_stats):
    """Execute trades based on current strategy and market data"""
    try:
        for pair, data in market_data.items():
            if not data:
                continue

            # Get latest candle
            latest_candle = data[-1]

            # Calculate indicators
            indicators = Indicators(strategy.get('indicator_params', {}))
            analyzed_data = indicators.calculate_indicators(
                pd.DataFrame([latest_candle]),
                timeframe="1m"
            )

            # Apply strategy rules
            rsi = analyzed_data['rsi'].iloc[-1]
            macd = analyzed_data['macd'].iloc[-1]
            macd_signal = analyzed_data['macd_signal'].iloc[-1]

            # Get strategy thresholds
            rsi_buy = strategy['rsi_thresholds']['buy']
            rsi_sell = strategy['rsi_thresholds']['sell']

            # Make trading decision
            if rsi < rsi_buy and macd > macd_signal:
                confidence = strategy['position_sizing']['base_size']
                await trading_bot.execute_trade(pair, "BUY", confidence, latest_candle)
            elif rsi > rsi_sell and macd < macd_signal:
                confidence = strategy['position_sizing']['base_size']
                await trading_bot.execute_trade(pair, "SELL", confidence, latest_candle)

            # Update statistics
            trade_info = trading_bot.get_last_trade_info(pair)
            if trade_info:
                trading_stats.update_stats(pair, trade_info)
    except Exception as e:
        logging.error(f"Error executing trades: {e}")


async def shutdown(signal, loop):
    """Cleanup function for graceful shutdown"""
    logging.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    [task.cancel() for task in tasks]
    logging.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


async def main():
   logger = logging.getLogger(__name__)
   logger.info("Initializing trading bot components...")

   # Initialize components
   rate_limiter = RateLimiter(calls_per_second=1, calls_per_minute=60)
   market_data_service = MarketDataService(rate_limiter)
   database_agent = DatabaseAgent()
   parameter_agent = ParameterAgent(database_agent=database_agent)
   strategy_manager = StrategyManager(llm_model='ollama/0xroyce/plutus')
   trading_stats = TradingStats()
   trading_bot = TradingBotAgent(
       initial_balance=10000,
       parameter_agent=parameter_agent,
       database_agent=database_agent
   )

   logger.info("Starting main trading loop...")

   try:
       while True:
           # Get top trading pairs
           top_pairs = market_data_service.get_top_volume_pairs(limit=10)
           logger.info(f"Top trading pairs: {top_pairs}")

           # Update market data
           market_data = {}
           for pair in top_pairs:
               ohlcv = market_data_service.get_ohlcv(pair, interval=1)
               if ohlcv:
                   market_data[pair] = ohlcv

           # Update trading strategy
           overall_stats = trading_stats.get_overall_stats()
           strategy = await strategy_manager.update_strategy(market_data, overall_stats)

           if strategy:
               logger.info("Trading with updated strategy parameters")
               await execute_trades(trading_bot, market_data, strategy, trading_stats)

           # Log current performance
           current_stats = trading_stats.get_overall_stats()
           logger.info(
               f"Performance: {current_stats['total_trades']} trades, "
               f"P/L: ${current_stats['total_profit_loss']:.2f}, "
               f"Win Rate: {current_stats['win_rate'] * 100:.1f}%"
           )

           await asyncio.sleep(60)  # Wait for next iteration

   except asyncio.CancelledError:
       logger.info("Main loop cancelled, shutting down...")
   except Exception as e:
       logger.error(f"Unexpected error: {e}", exc_info=True)
       raise
   finally:
       await trading_bot.close()
       await strategy_manager.close()
       logger.info("Trading bot shutdown complete.")


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting trading bot...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, exiting...")
        sys.exit(0)