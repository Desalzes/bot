import os
import json
import logging
import threading
import pandas as pd
import numpy as np
import itertools
from crewai import Agent, Task, Crew, LLM

class ParameterAgent:
    def __init__(self, database_agent, trading_bot_agent=None, config_file='config.json'):
        """
        Initializes the ParameterAgent with specified parameters.

        Args:
            database_agent (DatabaseAgent): Instance of DatabaseAgent.
            trading_bot_agent (TradingBotAgent, optional): Instance of TradingBotAgent for collaboration.
            config_file (str): Path to the configuration JSON file.
        """

        # Initialize the language model with your specified setup
        self.llm = LLM(
            provider='ollama',               # Specify the provider
            model='ollama/llama3',           # Specify the model name
            base_url='http://localhost:11435',  # Update base URL if needed
            verbose=True
        )

        # Check if LLM is initialized
        if self.llm is None:
            raise ValueError("Failed to initialize LLM: LLM object is None.")

        logging.info("LLM initialized successfully.")

        # Initialize Crew AI Agent attributes
        self.agent = Agent(
            role="Strategic Parameter Optimizer and Analyst",
            goal="""
                Primary Goal:
                - Optimize technical indicator parameters to maximize trading efficiency and profitability.

                Secondary Goals:
                - Enhance overall trading strategies by identifying optimal parameter settings based on data analysis.
                - Ensure that parameter adjustments do not increase risk beyond acceptable levels.
                - Establish a feedback loop where parameter adjustments are informed by ongoing performance data.
            """,
            backstory="""
                You are an experienced Technical Analyst AI specializing in parameter optimization within financial markets. You have a history of optimizing parameters that have significantly improved trading performance. You continuously learn from both successes and failures, refining your optimization techniques over time. You have established strong collaborative relationships with other agents, understanding their needs and the impact of parameter changes on their operations. Your expertise and reliability make you a trusted resource for enhancing trading strategies.
            """,
            llm=self.llm
        )

        self.database_agent = database_agent  # Direct reference to DatabaseAgent
        self.trading_bot_agent = trading_bot_agent  # For collaboration (can be None)

        self.config_file = os.path.abspath(config_file)
        self.parameters = {}
        self.lock = threading.Lock()
        self.load_parameters()

        # Configure logging
        logging.basicConfig(
            filename='../logs/parameter_agent_logs.txt',
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

        logging.info("ParameterAgent initialized with enhanced features.")

    # -------------------- Parameter Management --------------------

    def load_parameters(self):
        """
        Loads parameters from the configuration JSON file.
        """
        with self.lock:
            try:
                with open(self.config_file, 'r') as f:
                    self.parameters = json.load(f)
                logging.info("Configuration parameters loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading configuration file {self.config_file}: {e}")
                self.parameters = self.default_parameters()
                logging.info("Using default configuration.")

    def default_parameters(self):
        """
        Returns default parameters with timeframe-specific settings.
        """
        default_params = {
            "timeframes": [1, 5, 15],
            "ema_short_window": {"1": 5, "5": 8, "15": 13},
            "ema_long_window": {"1": 13, "5": 21, "15": 34},
            "rsi_window": {"1": 14, "5": 14, "15": 14},
            "bollinger_window": {"1": 20, "5": 20, "15": 20},
            "bollinger_dev": {"1": 2.0, "5": 2.0, "15": 2.0},
            "initial_balance": 10000,
            # Add more default parameters as needed
        }
        return default_params

    def get_parameters(self):
        """
        Returns all parameters.
        """
        with self.lock:
            return self.parameters

    def get_parameter(self, key, timeframe=None, default=None):
        """
        Retrieves a single parameter by key and timeframe, returning default if not found.

        Args:
            key (str): The parameter key.
            timeframe (int or str, optional): The timeframe for which to get the parameter.
            default: The default value if the parameter is not found.

        Returns:
            The parameter value.
        """
        with self.lock:
            if timeframe is not None and key in self.parameters:
                timeframe = str(timeframe)
                param = self.parameters.get(key, {})
                return param.get(timeframe, default)
            else:
                return self.parameters.get(key, default)

    def update_parameters(self, new_parameters):
        """
        Updates parameters and writes them back to the configuration JSON file.

        Args:
            new_parameters (dict): Dictionary containing new parameter values.
        """
        with self.lock:
            try:
                self.parameters.update(new_parameters)
                with open(self.config_file, 'w') as f:
                    json.dump(self.parameters, f, indent=4)
                logging.info("Configuration parameters updated successfully.")
            except Exception as e:
                logging.error(f"Error updating configuration file {self.config_file}: {e}")

    # -------------------- Data Retrieval --------------------

    def retrieve_historical_data(self):
        """
        Retrieves historical trade data via the DatabaseAgent.

        Returns:
            pd.DataFrame: DataFrame containing historical trade data.
        """
        try:
            trade_history = self.database_agent.get_trade_history()
            if not trade_history.empty:
                trade_history['timestamp'] = pd.to_datetime(trade_history['timestamp'])
                logging.info("Trade history retrieved successfully.")
            else:
                logging.warning("Trade history is empty.")
            return trade_history
        except Exception as e:
            logging.error(f"Error retrieving historical data via DatabaseAgent: {e}")
            return pd.DataFrame()

    def retrieve_indicator_data(self, symbol, indicator_name, timeframe):
        """
        Retrieves technical indicator data via the DatabaseAgent.

        Args:
            symbol (str): Trading symbol.
            indicator_name (str): Name of the indicator.
            timeframe (int): Timeframe in minutes.

        Returns:
            pd.DataFrame: DataFrame containing indicator data.
        """
        try:
            indicator_data = self.database_agent.get_indicator_data(
                symbol=symbol,
                indicator_name=indicator_name,
                timeframe=timeframe
            )
            if not indicator_data.empty:
                indicator_data['timestamp'] = pd.to_datetime(indicator_data['timestamp'])
                logging.info(f"Indicator data retrieved for {symbol}, {indicator_name}, {timeframe}m.")
            else:
                logging.warning(f"No indicator data available for {symbol}, {indicator_name}, {timeframe}m.")
            return indicator_data
        except Exception as e:
            logging.error(f"Error retrieving indicator data via DatabaseAgent: {e}")
            return pd.DataFrame()

    # -------------------- Performance Analysis --------------------

    def analyze_performance(self, trade_history):
        """
        Analyzes trading performance and computes KPIs.

        Args:
            trade_history (pd.DataFrame): DataFrame containing historical trade data.

        Returns:
            dict: Dictionary containing KPIs.
        """
        try:
            if trade_history.empty:
                logging.warning("No trade history available for performance analysis.")
                return {}

            # Calculate profit and loss per trade
            trade_history['profit'] = trade_history.apply(
                lambda row: (row['price'] * row['amount']) if row['action'] == 'sell' else (-row['price'] * row['amount']),
                axis=1
            )

            # Initialize KPIs dictionary
            kpis = {
                'total_profit': trade_history['profit'].sum(),
                'num_trades': len(trade_history),
                'win_loss_ratio': 0,
                'roi': 0,
                'average_profit_per_trade': trade_history['profit'].mean(),
                'max_drawdown': 0
            }

            # Calculate win/loss ratio
            num_wins = len(trade_history[trade_history['profit'] > 0])
            num_losses = len(trade_history[trade_history['profit'] < 0])
            kpis['win_loss_ratio'] = num_wins / num_losses if num_losses > 0 else float('inf')

            # Calculate ROI
            initial_balance = self.parameters.get('initial_balance', 10000)
            kpis['roi'] = (kpis['total_profit'] / initial_balance) * 100

            # Calculate max drawdown
            cumulative_profit = trade_history['profit'].cumsum()
            cumulative_max = cumulative_profit.cummax()
            drawdown = cumulative_max - cumulative_profit
            kpis['max_drawdown'] = drawdown.max()

            logging.info(f"Performance KPIs: {kpis}")
            return kpis
        except Exception as e:
            logging.error(f"Error analyzing performance: {e}")
            return {}

    # -------------------- Backtesting Methods --------------------

    def backtest_parameters(self, parameters):
        """
        Backtests trading strategy using the given parameters.

        Args:
            parameters (dict): Parameters to use for backtesting.

        Returns:
            dict: Performance metrics from the backtest.
        """
        try:
            # Implement backtesting logic here
            # For this example, we'll simulate backtest performance
            performance = {
                'total_profit': np.random.uniform(-1000, 1000),
                'num_trades': np.random.randint(10, 100),
                'win_loss_ratio': np.random.uniform(0.5, 2.0),
                'roi': np.random.uniform(-10, 10),
                'average_profit_per_trade': np.random.uniform(-50, 50),
                'max_drawdown': np.random.uniform(0, 500)
            }

            logging.info(f"Backtest performance: {performance}")
            return performance
        except Exception as e:
            logging.error(f"Error during backtesting: {e}")
            return {}

    def backtest_parameters_on_data(self, parameters, data):
        """
        Backtests trading strategy using the given parameters and data.

        Args:
            parameters (dict): Parameters to use for backtesting.
            data (pd.DataFrame): Historical data to use for backtesting.

        Returns:
            dict: Performance metrics from the backtest.
        """
        # Implement backtesting logic here using the provided data
        # Placeholder performance
        performance = {
            'total_profit': np.random.uniform(-1000, 1000),
            # Add other performance metrics as needed
        }
        return performance

    # -------------------- Parameter Optimization --------------------

    def optimize_parameters(self):
        """
        Optimizes technical indicator parameters based on historical performance.

        Updates the parameters in the configuration file.
        """
        try:
            # Advanced Optimization Algorithm (e.g., Grid Search)
            best_parameters, best_performance = self.grid_search_optimization()

            # Include backtesting before applying parameter changes
            current_parameters = self.parameters.copy()
            current_performance = self.backtest_parameters(current_parameters)

            if best_performance['total_profit'] > current_performance['total_profit']:
                # Implement parameter stability check
                if self.parameter_stability_check(best_parameters):
                    # Update parameters
                    self.update_parameters(best_parameters)
                    logging.info("Parameters optimized and updated successfully.")
                else:
                    logging.info("New parameters failed stability check. Keeping existing parameters.")
            else:
                logging.info("New parameters did not improve performance. Keeping existing parameters.")

        except Exception as e:
            logging.error(f"Error optimizing parameters: {e}")

    def grid_search_optimization(self):
        """
        Performs grid search over parameter space to find optimal parameters.

        Returns:
            tuple: (best_parameters, best_performance)
        """
        try:
            # Define the parameter grid for each timeframe
            param_grid = {
                'ema_short_window': {"1": [5, 8, 13], "5": [5, 8, 13], "15": [5, 8, 13]},
                'ema_long_window': {"1": [13, 21, 34], "5": [13, 21, 34], "15": [13, 21, 34]},
                'rsi_window': {"1": [10, 14], "5": [10, 14], "15": [10, 14]},
                # Add more parameters as needed
            }

            timeframes = [str(tf) for tf in self.parameters.get('timeframes', [1, 5, 15])]

            # Generate all combinations of parameters
            keys = list(param_grid.keys())
            value_lists = []
            for key in keys:
                for tf in timeframes:
                    value_lists.append(param_grid[key][tf])

            param_combinations = list(itertools.product(*value_lists))

            best_performance = {'total_profit': -float('inf')}
            best_parameters = None

            for combination in param_combinations:
                test_params = self.parameters.copy()
                idx = 0
                for key in keys:
                    for tf in timeframes:
                        value = combination[idx]
                        test_params[key][tf] = value
                        idx += 1

                # Simulate trading with these parameters
                performance = self.backtest_parameters(test_params)
                total_profit = performance.get('total_profit', 0)

                if total_profit > best_performance['total_profit']:
                    best_performance = performance
                    best_parameters = test_params.copy()

            if best_parameters:
                logging.info(f"Best parameters found: {best_parameters}")
                return best_parameters, best_performance
            else:
                logging.warning("No optimal parameters found.")
                return self.parameters, best_performance

        except Exception as e:
            logging.error(f"Error during grid search optimization: {e}")
            return self.parameters, {'total_profit': -float('inf')}

    # -------------------- Parameter Stability and Sensitivity --------------------

    def parameter_stability_check(self, parameters):
        """
        Checks the stability of parameters over multiple periods.

        Args:
            parameters (dict): Parameters to check.

        Returns:
            bool: True if parameters are stable, False otherwise.
        """
        try:
            # Split data into multiple periods (e.g., using walk-forward analysis)
            periods = self.get_walk_forward_periods()
            performances = []

            for train_data, test_data in periods:
                # Backtest on testing data
                performance = self.backtest_parameters_on_data(parameters, test_data)
                total_profit = performance.get('total_profit', 0)
                performances.append(total_profit)

            # Analyze performance consistency
            performance_variance = np.var(performances)
            threshold = 1000  # Define a threshold for acceptable variance

            if performance_variance < threshold:
                logging.info("Parameter stability check passed.")
                return True
            else:
                logging.info("Parameter stability check failed.")
                return False

        except Exception as e:
            logging.error(f"Error during parameter stability check: {e}")
            return False

    def get_walk_forward_periods(self):
        """
        Generates periods for walk-forward analysis.

        Returns:
            list of tuples: Each tuple contains (train_data, test_data)
        """
        # Implement logic to split historical data into training and testing periods
        # For simplicity, return an empty list in this placeholder
        return []

    def sensitivity_analysis(self):
        """
        Performs sensitivity analysis on parameters to determine their impact on performance.
        """
        try:
            base_parameters = self.parameters.copy()
            sensitive_params = {}
            timeframes = [str(tf) for tf in self.parameters.get('timeframes', [1, 5, 15])]

            for key in base_parameters.keys():
                if isinstance(base_parameters[key], dict):
                    # Handle timeframe-specific parameters
                    for timeframe in timeframes:
                        value = base_parameters[key].get(timeframe)
                        if value is None:
                            continue
                        variations = [value * 0.8, value, value * 1.2]
                        performances = []
                        for var in variations:
                            test_params = self.parameters.copy()
                            test_params[key][timeframe] = var
                            performance = self.backtest_parameters(test_params)
                            total_profit = performance.get('total_profit', 0)
                            performances.append((var, total_profit))
                        sensitive_params[f"{key}_{timeframe}"] = performances
                else:
                    # Non-timeframe-specific parameters
                    value = base_parameters[key]
                    variations = [value * 0.8, value, value * 1.2]
                    performances = []
                    for var in variations:
                        test_params = self.parameters.copy()
                        test_params[key] = var
                        performance = self.backtest_parameters(test_params)
                        total_profit = performance.get('total_profit', 0)
                        performances.append((var, total_profit))
                    sensitive_params[key] = performances

            # Analyze and log sensitivity results
            logging.info("Sensitivity Analysis Results:")
            for param, results in sensitive_params.items():
                logging.info(f"{param}: {results}")
        except Exception as e:
            logging.error(f"Error during sensitivity analysis: {e}")

    # -------------------- Inter-Agent Communication --------------------

    def receive_feedback(self, feedback):
        """
        Receives feedback from other agents to inform parameter optimization.

        Args:
            feedback (dict): Feedback data containing performance metrics and observations.
        """
        # Process feedback and adjust optimization strategy accordingly
        logging.info(f"Received feedback: {feedback}")
        # Implement logic to use feedback in optimization

    def share_insights(self):
        """
        Shares optimization insights with other agents.
        """
        insights = {
            'current_parameters': self.parameters,
            # Add more insights as needed
        }
        # Assuming there's a method in TradingBotAgent to receive insights
        if self.trading_bot_agent:
            self.trading_bot_agent.receive_insights(insights)
            logging.info("Shared insights with TradingBotAgent.")

    # -------------------- Optimization Task Execution --------------------

    def perform_optimization_task(self):
        """
        Periodically perform optimization based on historical data.

        This method can be scheduled to run at desired intervals.
        """
        self.optimize_parameters()

    # -------------------- Closing Method --------------------

    def close(self):
        """
        Placeholder for any cleanup actions.
        """
        pass
