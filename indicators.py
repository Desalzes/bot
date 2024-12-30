class Indicators:
   def __init__(self, parameters=None):
       self.parameters = parameters or {}

   def get_parameter(self, name, timeframe, default=None):
       if timeframe in self.parameters:
           return self.parameters[timeframe].get(name, default)
       return default

   def calculate_indicators(self, df, timeframe):
       """Calculate technical indicators"""
       if df.empty:
           return df

       try:
           close = df['close']
           high = df['high']
           low = df['low']
           volume = df['volume']

           # Get parameters for this timeframe
           ema_short_window = self.get_parameter('ema_short_window', timeframe, 12)
           ema_long_window = self.get_parameter('ema_long_window', timeframe, 26)
           rsi_window = self.get_parameter('rsi_window', timeframe, 14)

           # Calculate EMA
           df['ema_short'] = df['close'].ewm(span=ema_short_window).mean()
           df['ema_long'] = df['close'].ewm(span=ema_long_window).mean()

           # Calculate RSI
           delta = df['close'].diff()
           gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
           loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
           rs = gain / loss
           df['rsi'] = 100 - (100 / (1 + rs))

           # Calculate MACD
           exp1 = df['close'].ewm(span=12, adjust=False).mean()
           exp2 = df['close'].ewm(span=26, adjust=False).mean()
           df['macd'] = exp1 - exp2
           df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

           return df.fillna(0)

       except Exception as e:
           logging.error(f"Error calculating indicators: {e}")
           return df