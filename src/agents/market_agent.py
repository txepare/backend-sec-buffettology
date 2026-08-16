# pyrefly: ignore [missing-import]
import logging
from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.tools.market_api import MarketDataAPI

logger = logging.getLogger(__name__)


class MarketDataAgent(BaseAgent):
    """
    Agente especializado en obtener cotizaciones y métricas de mercado en tiempo real.
    """

    def __init__(self):
        super().__init__(
            agent_name="MarketDataAgent",
            prompt_file="market_extractor.xml",
            temperature=0.0
        )

    def fetch_market_data(self, ticker: str) -> Dict[str, Any]:
        """
        Obtiene datos de cotización actual desde Yahoo Finance.
        """
        logger.info(f"[{self.agent_name}] Consultando datos de cotización para {ticker}...")
        return MarketDataAPI.get_market_quote(ticker)

    def get_data(self, ticker: str) -> Dict[str, Any]:
        return self.fetch_market_data(ticker)