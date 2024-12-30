import json
import logging
from datetime import datetime, timedelta


class StrategyManager:
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.current_strategy = None
        self.strategy_update_interval = timedelta(hours=1)
        self.last_strategy_update = None
        self.logger = logging.getLogger(__name__)

    def should_update_strategy(self):
        """Determines if it's time to update the trading strategy"""
        if not self.last_strategy_update:
            return True
        return datetime.now() - self.last_strategy_update >= self.strategy_update_interval

    async def update_strategy(self, market_data, trading_stats):
        if not self.should_update_strategy():
            return self.default_strategy

        try:
            prompt = self._create_strategy_prompt(market_data, trading_stats)
            response = await self.llm.complete(prompt)  # Changed from generate_strategy

            new_strategy = self._parse_strategy_response(response)
            if new_strategy:
                self.current_strategy = new_strategy
                self.last_strategy_update = datetime.now()
                self.logger.info("Trading strategy updated successfully")
                return self.current_strategy

            return self.default_strategy

        except Exception as e:
            self.logger.error(f"Error updating strategy: {e}")
            return self.default_strategy

    def _create_strategy_prompt(self, market_data, trading_stats):
        """Creates a prompt for the LLM to generate trading strategy"""
        return f"""
        Based on the following market conditions and trading performance:

        Market Data:
        {market_data}

        Trading Statistics:
        {trading_stats}

        Please analyze the current market conditions and suggest optimal trading parameters:
        1. RSI thresholds for overbought/oversold conditions
        2. MACD signal thresholds
        3. Volume threshold multipliers
        4. Position sizing recommendations
        5. Risk management parameters

        Provide the response in JSON format with specific numerical values.
        """

    def _parse_strategy_response(self, response):
        """Parses and validates the LLM response"""
        try:
            strategy = json.loads(response)
            required_fields = [
                'rsi_thresholds', 'macd_thresholds',
                'volume_multipliers', 'position_sizing',
                'risk_parameters'
            ]

            if all(field in strategy for field in required_fields):
                return strategy
            else:
                self.logger.error("Invalid strategy format received from LLM")
                return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing strategy JSON: {e}")
            return None

        