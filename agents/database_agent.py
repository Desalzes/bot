import os
import sqlite3
import logging
import threading
from datetime import datetime
import pandas as pd
import redis
from crewai import Agent, LLM

class DatabaseAgent:
    def __init__(self, database='trading_data.db'):
        """
        Initializes the DatabaseAgent with specified parameters.

        Args:
            database (str): Path to the SQLite database file.
        """
        # Initialize the language model with your specified setup
        self.llm = LLM(
            model='ollama/llama3',
            base_url='http://localhost:11435',
            verbose=True
        )

        if self.llm is None:
            raise ValueError("Failed to initialize LLM: LLM object is None.")

        logging.info("LLM initialized successfully.")

        # Initialize Crew AI Agent attributes
        self.agent = Agent(
            role="Data Integrity Guardian and Insight Provider",
            goal="""
                Primary Goal:
                - Maintain data integrity by ensuring all data within the system is accurate, consistent, and secure, enabling reliable decision-making by other agents.

                Secondary Goals:
                - Provide actionable insights by analyzing data to generate patterns, trends, and correlations that can improve trading strategies and system performance.
                - Optimize data storage and retrieval for efficiency and scalability.
                - Facilitate knowledge sharing by providing relevant data and insights to other agents to enhance overall system performance.
            """,
            backstory="""
                You are an experienced Data Analyst AI specializing in financial systems, with a rich history of handling various data-related challenges, such as inconsistent data formats, missing values, and data corruption. You continuously update your analytical methods to incorporate the latest techniques in data science and machine learning. You have established strong collaborative relationships with other agents, understanding their data needs and preferences. Your expertise and accuracy make you a trustworthy data provider, enhancing collaboration and system efficiency.
            """,
            llm=self.llm
        )

        # Ensure the database path is absolute
        self.database = os.path.abspath(database)
        self.conn = None
        self.cursor = None
        self.create_database()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.lock = threading.Lock()

        # Configure logging
        logging.basicConfig(
            filename='database_agent_logs.txt',
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

        logging.info("DatabaseAgent initialized with enhanced features.")

    def create_database(self):
        """
        Create a SQLite database to store OHLCV data, technical indicators, and trade history.
        """
        try:
            # Connect to the database (this will create the file if it doesn't exist)
            self.conn = sqlite3.connect(self.database, check_same_thread=False)
            self.cursor = self.conn.cursor()

            # Create OHLCV data table with the timestamp column
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp TEXT,  -- Ensure 'timestamp' column is here
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    timeframe INTEGER
                )
            ''')

            # Create indicators table with the timestamp column
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp TEXT,  -- Ensure 'timestamp' column is here
                    indicator_name TEXT,
                    value REAL,
                    timeframe INTEGER
                )
            ''')

            # Create trade history table with the timestamp column
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    action TEXT,
                    price REAL,
                    amount REAL,
                    timestamp TEXT  -- Ensure 'timestamp' column is here
                )
            ''')

            # Create indices for faster retrieval
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data (timestamp)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_timestamp ON indicators (timestamp)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_history_timestamp ON trade_history (timestamp)')

            # Commit changes to the database
            self.conn.commit()
            logging.info(f"Database initialized at {self.database}.")
        except Exception as e:
            logging.error(f"Error initializing database at {self.database}: {e}")

    def store_ohlcv_data(self, symbol, timestamp, open_price, high, low, close, volume, timeframe):
        """
        Store OHLCV data in the database.

        Args:
            symbol (str): Trading symbol.
            timestamp (str): Timestamp of the data.
            open_price (float): Open price.
            high (float): High price.
            low (float): Low price.
            close (float): Close price.
            volume (float): Trading volume.
            timeframe (int): Timeframe of the data in minutes.
        """
        try:
            with self.lock:
                self.cursor.execute('''
                    INSERT INTO ohlcv_data (symbol, timestamp, open, high, low, close, volume, timeframe)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, timestamp, open_price, high, low, close, volume, timeframe))
                self.conn.commit()
            logging.info(f"Stored OHLCV data for {symbol} at {timestamp} for timeframe {timeframe}m.")
        except Exception as e:
            logging.error(f"Error storing OHLCV data for {symbol} at {timestamp}: {e}")

    def close(self):
        """
        Close the database connection.
        """
        try:
            if self.conn:
                self.conn.close()
                logging.info("Database connection closed.")
        except Exception as e:
            logging.error(f"Error closing database connection: {e}")
