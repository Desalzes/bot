# agents/trading_bot_agent.py

import os
import logging
import pandas as pd
import time
import json
from datetime import datetime
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from crewai import Agent, Task, Crew, LLM

class TradingBotAgent:
    def __init__(self, parameter_agent, database_agent, initial_balance=10000, config_file='config.json',
                 log_file='../logs/iteration_logs.txt', model_dir='../models'):
        """
        Initializes the TradingBotAgent with specified parameters.

        Args:
            parameter_agent (ParameterAgent): Instance of ParameterAgent.
            database_agent (DatabaseAgent): Instance of DatabaseAgent.
            initial_balance (float): Starting balance for trading.
            config_file (str): Path to the configuration JSON file.
            log_file (str): Path to the iteration log file.
            model_dir (str): Directory to save/load ML models.
        """

        # Initialize the language model with your specified setup
        self.llm = LLM(
            model='ollama/llama3',  # Specify the model name
            base_url='http://localhost:11435',  # Update base URL if needed
            verbose=True
        )

        # Check if LLM is initialized
        if self.llm is None:
            raise ValueError("Failed to initialize LLM: LLM object is None.")

        logging.info("LLM initialized successfully.")

        # Initialize Crew AI Agent attributes
        self.agent = Agent(
            role="Adaptive Market Strategist",
            goal="""
                Primary Goal:
                - Maximize net profit by intelligently executing trades based on comprehensive market analysis.

                Secondary Goals:
                - Manage risk effectively to maintain acceptable drawdowns and align with risk tolerance levels.
                - Continuously adapt trading strategies in response to real-time market conditions and predictive insights.
                - Ensure all trading activities comply with relevant financial regulations and ethical standards.
            """,
            backstory="""
                You are an experienced Market Navigator AI with a rich history of simulated trading experiences across various market cycles, including bull markets, bear markets, and periods of high volatility. You have learned from past successes and failures, refining your strategies over time. You actively collaborate with other specialized agents, such as the ParameterAgent and DatabaseAgent, leveraging historical data and predictive analytics to inform your decisions. Your adaptive strategies and risk management skills make you a proficient and intelligent participant in the financial markets.
            """,
            llm=self.llm
        )

        self.parameter_agent = parameter_agent
        self.database_agent = database_agent  # Direct reference to DatabaseAgent
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.portfolio = {}  # e.g., {'BTCUSD': amount}
        self.trade_history = []
        self.data = {}  # e.g., {'BTCUSD': {1: DataFrame, 5: DataFrame}}
        self.verbose = True
        self.base_url = "https://api.kraken.com/0/public"  # Used for crypto trading
        self.symbols = self.get_top_kraken_tickers()  # Get top tickers by volume from Kraken
        self.timeframes = [1, 5, 15]  # Timeframes in minutes

        # Configure logging
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            force=True
        )

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        if self.verbose:
            logging.info("TradingBotAgent initialized with multi-timeframe analysis and direct database integration.")

    # -------------------- Utility Methods --------------------

    def get_top_kraken_tickers(self, limit=5):
        """
        Retrieve the top tickers by volume from Kraken's API.

        Args:
            limit (int): Number of top tickers to retrieve.

        Returns:
            list: List of trading symbols.
        """
        try:
            response = requests.get(f"{self.base_url}/AssetPairs")
            response.raise_for_status()
            data = response.json()['result']
            symbols = list(data.keys())[:limit]
            if self.verbose:
                logging.info(f"Top {limit} tickers: {symbols}")
            return symbols
        except Exception as e:
            logging.error(f"Error fetching top Kraken tickers: {e}")
            return []

    # -------------------- Data Fetching and Analysis --------------------

    def update_data(self):
        """
        Update price data for all symbols and timeframes.
        """
        if self.verbose:
            logging.info("Updating price data for all symbols and timeframes...")
        for symbol in self.symbols:
            data = self.fetch_price_data(symbol)
            if data:
                self.data[symbol] = data
            else:
                logging.warning(f"No data for symbol {symbol}, skipping.")

    def fetch_price_data(self, symbol):
        """
        Fetch historical price data for a symbol for multiple timeframes.

        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSD').

        Returns:
            dict: Dictionary containing DataFrames for each timeframe.
        """
        data = {}
        for interval in self.timeframes:
            df = self._fetch_data_from_database(symbol, interval)
            if not df.empty:
                data[interval] = df
            else:
                logging.warning(f"No data for symbol {symbol} at interval {interval}m.")
        return data

    def _fetch_data_from_database(self, symbol, interval):
        """
        Fetch historical price data for a symbol directly from the DatabaseAgent.

        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSD').
            interval (int): Time interval in minutes.

        Returns:
            pd.DataFrame: DataFrame containing historical price data.
        """
        try:
            df = self.database_agent.get_ohlcv_data(symbol, interval)
            if df is not None and not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.astype(float)
                return df
            else:
                logging.warning(f"No data returned for symbol {symbol} from DatabaseAgent.")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error fetching data from DatabaseAgent for {symbol} at interval {interval}m: {e}")
            return pd.DataFrame()

    def analyze_data(self, symbol):
        """
        Analyze market data and calculate technical indicators for all timeframes.

        Args:
            symbol (str): Trading symbol.
        """
        data = self.data.get(symbol, {})
        if not data:
            logging.warning(f"No data available for symbol {symbol} to analyze.")
            return

        analyzed_data = {}
        for interval, df in data.items():
            df = df.copy()
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']

            # Get parameters for this interval
            ema_short_window = self.parameter_agent.get_parameter('ema_short_window', timeframe=interval, default=5)
            ema_long_window = self.parameter_agent.get_parameter('ema_long_window', timeframe=interval, default=10)
            rsi_window = self.parameter_agent.get_parameter('rsi_window', timeframe=interval, default=14)
            bollinger_window = self.parameter_agent.get_parameter('bollinger_window', timeframe=interval, default=20)
            bollinger_dev = self.parameter_agent.get_parameter('bollinger_dev', timeframe=interval, default=2.0)

            # Calculate EMA indicators
            df['ema_short'] = EMAIndicator(close=close, window=ema_short_window).ema_indicator()
            df['ema_long'] = EMAIndicator(close=close, window=ema_long_window).ema_indicator()

            # Calculate RSI
            df['rsi'] = RSIIndicator(close=close, window=rsi_window).rsi()

            # Calculate Bollinger Bands
            bb_indicator = BollingerBands(close=close, window=bollinger_window, window_dev=bollinger_dev)
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_lower'] = bb_indicator.bollinger_lband()

            # Calculate MACD
            macd_indicator = MACD(close=close)
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()

            # Drop rows with NaN values generated by indicators
            df.dropna(inplace=True)

            analyzed_data[interval] = df

        self.data[symbol] = analyzed_data

    # -------------------- Decision Making --------------------

    def aggregate_insights(self, symbol):
        """
        Aggregate insights from different timeframes to make a cohesive trading decision.

        Args:
            symbol (str): Trading symbol.

        Returns:
            dict: Aggregated technical data from all timeframes.
        """
        technical_data = {}
        for interval, df in self.data[symbol].items():
            latest = df.iloc[-1]
            technical_data[interval] = {
                'ema_short': latest['ema_short'],
                'ema_long': latest['ema_long'],
                'rsi': latest['rsi'],
                'bb_upper': latest['bb_upper'],
                'bb_lower': latest['bb_lower'],
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                # Add more indicators as needed
            }
        return technical_data

    def make_decision(self, symbol):
        """
        Make a trading decision based on analyzed data from multiple timeframes and LLM insights.

        Args:
            symbol (str): Trading symbol.

        Returns:
            str: Suggested trade decision ('BUY', 'SELL', or 'HOLD').
        """
        technical_data = self.aggregate_insights(symbol)

        # Use LLM to make decision
        try:
            task_description = f"""
            Based on the following technical analysis data for {symbol} across multiple timeframes, make a trade decision.

            Technical Data:
            {json.dumps(technical_data, indent=4)}

            Consider the alignment of indicators across timeframes and provide a decision of 'BUY', 'SELL', or 'HOLD', along with a brief explanation.

            Note: Timeframes are in minutes. For example, '1' represents the 1-minute timeframe.
            """

            task = Task(
                description=task_description,
                expected_output="Provide a trade decision and a brief explanation.",
                agent=self.agent
            )
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                verbose=True
            )
            result = crew.kickoff()

            # Process LLM output to extract decision
            llm_output = result.final_answer.strip().upper()
            logging.info(f"Trade decision for {symbol}: {llm_output}")

            # Extract 'BUY', 'SELL', or 'HOLD' from LLM output
            if 'BUY' in llm_output:
                decision = 'BUY'
            elif 'SELL' in llm_output:
                decision = 'SELL'
            else:
                decision = 'HOLD'

            return decision
        except Exception as e:
            logging.error(f"Error making trade decision for {symbol}: {e}")
            return "HOLD"

    # -------------------- Trade Execution --------------------

    def execute_trade(self, symbol, decision):
        """
        Execute a trade based on the decision.

        Args:
            symbol (str): Trading symbol.
            decision (str): 'BUY', 'SELL', or 'HOLD'.
        """
        latest_price = self.data[symbol][self.timeframes[0]]['close'].iloc[-1]  # Use the latest price from the shortest timeframe

        if decision == "BUY" and self.balance > latest_price:
            # Buy
            amount = (self.balance * 0.1) / latest_price  # Use 10% of balance
            self.portfolio[symbol] = self.portfolio.get(symbol, 0) + amount
            self.balance -= amount * latest_price
            trade = {
                'symbol': symbol,
                'action': 'buy',
                'price': latest_price,
                'amount': amount,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.trade_history.append(trade)
            self.store_trade(trade)
            if self.verbose:
                logging.info(f"Bought {amount:.6f} {symbol} at ${latest_price:.2f}")
        elif decision == "SELL" and self.portfolio.get(symbol, 0) > 0:
            # Sell
            amount = self.portfolio[symbol]
            self.balance += amount * latest_price
            self.portfolio[symbol] = 0
            trade = {
                'symbol': symbol,
                'action': 'sell',
                'price': latest_price,
                'amount': amount,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.trade_history.append(trade)
            self.store_trade(trade)
            if self.verbose:
                logging.info(f"Sold {amount:.6f} {symbol} at ${latest_price:.2f}")
        else:
            if self.verbose:
                logging.info(f"No trade executed for {symbol}. Decision: {decision}")

    def store_trade(self, trade):
        """
        Stores trade information via the DatabaseAgent.

        Args:
            trade (dict): Trade information.
        """
        try:
            self.database_agent.store_trade(trade)
            logging.info(f"Trade stored for {trade['symbol']} at {trade['timestamp']}.")
        except Exception as e:
            logging.error(f"Error storing trade for {trade['symbol']} at {trade['timestamp']}: {e}")

    # -------------------- Strategy Adjustment --------------------

    def adjust_strategy_with_llm(self):
        """
        Use the LLM to adjust trading strategies based on historical performance and market conditions.
        """
        try:
            # Retrieve historical performance
            trade_history = self.get_trade_history()
            if trade_history.empty:
                logging.warning("Trade history is empty. Skipping strategy adjustment.")
                return

            # Prepare performance summary
            kpis = self.parameter_agent.analyze_performance(trade_history)
            performance_summary = f"Performance Summary:\n{json.dumps(kpis, indent=4)}"

            # Prepare market conditions summary (e.g., volatility, trends)
            market_conditions = {
                "current_balance": self.balance,
                "portfolio": self.portfolio,
                "market_volatility": self.calculate_market_volatility(),
                # Add more market condition metrics as needed
            }
            market_conditions_text = json.dumps(market_conditions, indent=4)

            task_description = f"""
            Based on the following trading performance and current market conditions, suggest adjustments to the trading strategy to optimize profitability.

            {performance_summary}

            Market Conditions:
            {market_conditions_text}

            Please provide specific strategy adjustments and explain the rationale behind each suggestion.
            """

            task = Task(
                description=task_description,
                expected_output="Provide strategy adjustments and explanations.",
                agent=self.agent
            )
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                verbose=True
            )
            result = crew.kickoff()

            # Process LLM output to adjust strategy
            llm_output = result.final_answer.strip()
            logging.info(f"Strategy Adjustment Suggestions:\n{llm_output}")
            # Implement strategy adjustments based on LLM suggestions
            # This requires parsing the LLM output and applying the changes
        except Exception as e:
            logging.error(f"Error adjusting strategy with LLM: {e}")

    # -------------------- Utility Methods --------------------

    def calculate_market_volatility(self):
        """
        Calculate overall market volatility based on price data.

        Returns:
            float: Market volatility metric.
        """
        try:
            # Simple example: Calculate standard deviation of returns across all symbols and timeframes
            volatilities = []
            for symbol, intervals_data in self.data.items():
                for interval, df in intervals_data.items():
                    if len(df) > 1:
                        returns = df['close'].pct_change().dropna()
                        volatility = returns.std()
                        volatilities.append(volatility)
            average_volatility = sum(volatilities) / len(volatilities) if volatilities else 0.0
            return average_volatility
        except Exception as e:
            logging.error(f"Error calculating market volatility: {e}")
            return 0.0

    def get_trade_history(self):
        """
        Retrieves trade history via the DatabaseAgent.

        Returns:
            pd.DataFrame: DataFrame containing the trade history.
        """
        try:
            df = self.database_agent.get_trade_history()
            if df is not None and not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            else:
                logging.warning("No trade history returned from DatabaseAgent.")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error retrieving trade history via DatabaseAgent: {e}")
            return pd.DataFrame()

    # -------------------- Main Loop --------------------

    def run_one_cycle(self):
        """
        Run a single trading cycle.
        """
        self.update_data()
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
            self.analyze_data(symbol)
            decision = self.make_decision(symbol)
            self.execute_trade(symbol, decision)
        # Adjust strategy periodically
        self.adjust_strategy_with_llm()

    def run(self, duration_minutes=60):
        """
        Run the trading bot for a specified duration.

        Args:
            duration_minutes (int): Duration to run the bot in minutes.
        """
        if self.verbose:
            logging.info(f"Starting TradingBotAgent for {duration_minutes} minutes...")
        start_time = time.time()
        duration_seconds = duration_minutes * 60

        while time.time() - start_time < duration_seconds:
            self.run_one_cycle()
            time.sleep(60)  # Wait 1 minute before next iteration

        self.summary()

    # -------------------- Summary and Closing --------------------

    def summary(self):
        """
        Print a summary of trading performance and log it.
        """
        try:
            logging.info("\n--- Trading Summary ---")
            total_portfolio_value = self.balance
            for symbol, amount in self.portfolio.items():
                if amount > 0 and symbol in self.data:
                    latest_price = self.data[symbol][self.timeframes[0]]['close'].iloc[-1]
                    value = amount * latest_price
                    total_portfolio_value += value
                    logging.info(f"Holding {amount:.6f} {symbol}, Value: ${value:.2f}")
            logging.info(f"Final Balance: ${self.balance:.2f}")
            logging.info(f"Total Portfolio Value: ${total_portfolio_value:.2f}")
            profit = total_portfolio_value - self.initial_balance
            logging.info(f"Net Profit/Loss: ${profit:.2f}")
            logging.info("\n====================\n")
        except Exception as e:
            logging.error(f"Error during summary: {e}")

    def close(self):
        """
        Placeholder for any cleanup actions.
        """
        pass
