import logging
from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.tools.sec_api import SecEdgarAPI

logger = logging.getLogger(__name__)


class SecExtractorAgent(BaseAgent):
    """
    Sub-agente responsable de extraer los datos XBRL de la SEC EDGAR.
    """

    def __init__(self):
        super().__init__(
            agent_name="SecExtractorAgent",
            prompt_file="sec_extractor.xml",
            temperature=0.0
        )

    def extract(self, ticker: str) -> Dict[str, Any]:
        logger.info(f"[{self.agent_name}] Extrayendo datos XBRL de la SEC para {ticker}...")
        facts_data = SecEdgarAPI.fetch_company_facts(ticker)
        return {
            "ticker": ticker,
            "raw_facts": facts_data
        }