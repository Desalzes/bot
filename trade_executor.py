# trade_executor.py

import logging
import os
import json
from datetime import datetime

class TradeExecutor:
    def __init__(self, exchange_manager, parameter_agent, verbose=True):
        self.exchange_manager = exchange_manager
        self.parameter_agent = parameter_agent
        self.verbose = verbose
        self.trade_history_file = './data/trade_history.json'
        self.portfolio_file = './data/portfolio.json'
        self.balance_file = './data/balance.json'
        self.initialize_files()

    def initialize_files(self):
        # Initialize trade history file
        if not os.path.exists(self.trade_history_file):
            with open(self.trade_history_file, 'w') as f:
                json.dump([], f)
            logging.info(f"Created trade history file at {self.trade_history_file}.")
        else:
            logging.info(f"Trade history file exists at {self.trade_history_file}.")

        # Initialize portfolio file
        if not os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'w') as f:
                json.dump({}, f)
            logging.info(f"Created portfolio file at {self.portfolio_file}.")
        else:
            logging.info(f"Portfolio file exists at {self.portfolio_file}.")

        # Initialize balance file
        if not os.path.exists(self.balance_file):
            initial_balance = self.parameter_agent.get_parameter('initial_balance', 10000.0)
            with open(self.balance_file, 'w') as f:
                json.dump({'balance': initial_balance}, f)
            logging.info(f"Created balance file at {self.balance_file} with initial balance {initial_balance}.")
        else:
            logging.info(f"Balance file exists at {self.balance_file}.")

    def load_trade_history(self):
        with open(self.trade_history_file, 'r') as f:
            return json.load(f)

    def save_trade_history(self, trade_history):
        with open(self.trade_history_file, 'w') as f:
            json.dump(trade_history, f, indent=4)

    def load_portfolio(self):
        with open(self.portfolio_file, 'r') as f:
            return json.load(f)

    def save_portfolio(self, portfolio):
        with open(self.portfolio_file, 'w') as f:
            json.dump(portfolio, f, indent=4)

    def load_balance(self):
        with open(self.balance_file, 'r') as f:
            return json.load(f)['balance']

    def save_balance(self, balance):
        with open(self.balance_file, 'w') as f:
            json.dump({'balance': balance}, f, indent=4)

    def execute_trade(self, symbol, decision, ticker_data, portfolio, balance, lock, interval):
        """
        Executes a trade based on the decision ('BUY' or 'SELL').
        In paper trading mode, updates the portfolio and balance without real transactions.
        """
        try:
            with lock:
                current_price = ticker_data['close'].iloc[-1]
                max_trade_amount = self.parameter_agent.get_parameter('max_trade_amount', 100)

                if decision == "BUY":
                    # Determine how much to buy based on max_trade_amount and current balance
                    trade_amount = min(max_trade_amount, balance)
                    if trade_amount <= 0:
                        logging.info(f"Insufficient balance to buy {symbol}.")
                        return

                    # Update portfolio and balance
                    quantity = trade_amount / current_price
                    if symbol in portfolio:
                        # Update average purchase price
                        total_cost = portfolio[symbol]['purchase_price'] * portfolio[symbol]['quantity']
                        total_cost += trade_amount
                        portfolio[symbol]['quantity'] += quantity
                        portfolio[symbol]['purchase_price'] = total_cost / portfolio[symbol]['quantity']
                    else:
                        portfolio[symbol] = {'quantity': quantity, 'purchase_price': current_price}
                    balance -= trade_amount

                    if self.verbose:
                        logging.info(f"Bought {quantity:.6f} {symbol} at {current_price} USD.")

                    # Log the trade
                    trade = {
                        'symbol': symbol,
                        'action': "BUY",
                        'price': current_price,
                        'quantity': quantity,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    trade_history = self.load_trade_history()
                    trade_history.append(trade)
                    self.save_trade_history(trade_history)

                elif decision == "SELL":
                    # Determine how much to sell based on portfolio holdings
                    if symbol not in portfolio or portfolio[symbol]['quantity'] <= 0:
                        logging.info(f"No holdings to sell for {symbol}.")
                        return

                    quantity = portfolio[symbol]['quantity']
                    trade_amount = quantity * current_price
                    balance += trade_amount
                    portfolio[symbol]['quantity'] = 0
                    portfolio[symbol].pop('purchase_price', None)

                    if self.verbose:
                        logging.info(f"Sold {quantity:.6f} {symbol} at {current_price} USD.")

                    # Log the trade
                    trade = {
                        'symbol': symbol,
                        'action': "SELL",
                        'price': current_price,
                        'quantity': quantity,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    trade_history = self.load_trade_history()
                    trade_history.append(trade)
                    self.save_trade_history(trade_history)

                else:
                    logging.info(f"No trade executed for {symbol}. Decision: {decision}")

                # Update portfolio and balance
                self.save_portfolio(portfolio)
                self.save_balance(balance)

        except Exception as e:
            logging.exception(f"Failed to execute trade for {symbol}: {e}")