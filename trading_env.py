# trading_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    A custom trading environment for Reinforcement Learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=1000):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.current_step = 0

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observations: [Price, RSI, EMA_short, EMA_long, MACD, Volume]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)

        # Initialize state
        self.balance = initial_balance
        self.position = 0
        self.cost_basis = 0
        self.net_worth = initial_balance

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.cost_basis = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, 'close'],
            self.df.loc[self.current_step, 'rsi'],
            self.df.loc[self.current_step, 'ema_short'],
            self.df.loc[self.current_step, 'ema_long'],
            self.df.loc[self.current_step, 'macd'],
            self.df.loc[self.current_step, 'volume']
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        done = False
        reward = 0

        # Execute action
        if action == 1:  # Buy
            if self.balance >= current_price:
                self.position += 1
                self.balance -= current_price
                self.cost_basis = current_price
                if self.verbose:
                    print(f"Bought at {current_price}")
        elif action == 2:  # Sell
            if self.position > 0:
                self.position -= 1
                self.balance += current_price
                reward = current_price - self.cost_basis
                if self.verbose:
                    print(f"Sold at {current_price}, Reward: {reward}")

        # Update net worth
        self.net_worth = self.balance + self.position * current_price

        # Calculate reward
        if action == 2 and self.position == 0:
            reward += self.net_worth - self.initial_balance

        # Increment step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        obs = self._next_observation()
        info = {}

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Position: {self.position}')
        print(f'Net Worth: {self.net_worth}')
        print(f'Profit: {profit}')