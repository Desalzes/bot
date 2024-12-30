import logging


class TradingStats:
    def __init__(self):
        self.stats = {}
        self.logger = logging.getLogger(__name__)

    def update_stats(self, symbol, trade_info):
        """Updates trading statistics for a symbol"""
        if symbol not in self.stats:
            self.stats[symbol] = {
                "trades": 0,
                "profit_loss": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_volume": 0.0,
                "max_drawdown": 0.0,
                "last_trade_time": None,
                "trade_history": []
            }

        stats = self.stats[symbol]
        stats["trades"] += 1
        stats["profit_loss"] += trade_info["profit_loss"]
        stats["total_volume"] += trade_info["volume"]
        stats["last_trade_time"] = trade_info["timestamp"]
        stats["trade_history"].append(trade_info)

        if trade_info["profit_loss"] > 0:
            stats["winning_trades"] += 1
        else:
            stats["losing_trades"] += 1

        # Update max drawdown
        if len(stats["trade_history"]) > 1:
            drawdown = self._calculate_drawdown(stats["trade_history"])
            stats["max_drawdown"] = min(stats["max_drawdown"], drawdown)

    def _calculate_drawdown(self, trade_history):
        """Calculates the current drawdown from trade history"""
        cumulative_pnl = [trade["profit_loss"] for trade in trade_history]
        peak = max(cumulative_pnl)
        return min(0, min(cumulative_pnl) - peak)

    def get_symbol_stats(self, symbol):
        """Returns statistics for a specific symbol"""
        return self.stats.get(symbol, {})

    def get_overall_stats(self):
        """Returns overall trading statistics"""
        overall = {
            "total_trades": 0,
            "total_profit_loss": 0.0,
            "win_rate": 0.0,
            "total_volume": 0.0,
            "max_drawdown": 0.0
        }

        for stats in self.stats.values():
            overall["total_trades"] += stats["trades"]
            overall["total_profit_loss"] += stats["profit_loss"]
            overall["total_volume"] += stats["total_volume"]
            overall["max_drawdown"] = min(overall["max_drawdown"], stats["max_drawdown"])

        if overall["total_trades"] > 0:
            total_winning = sum(s["winning_trades"] for s in self.stats.values())
            overall["win_rate"] = total_winning / overall["total_trades"]

        return overall