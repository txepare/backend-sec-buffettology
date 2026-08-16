import logging
from typing import Dict, Any
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SectorConfiguratorAgent(BaseAgent):
    """
    Agente responsable de clasificar el sector de la empresa y
    seleccionar las métricas de Buffettology relevantes.
    """

    def __init__(self):
        super().__init__(
            agent_name="SectorConfiguratorAgent",
            prompt_file="sector_configurator.xml",
            temperature=0.0
        )

    def configure_sector_rules(self, sector_name: str, industry_name: str = "") -> Dict[str, Any]:
        """
        Mapea el sector/industria a una de las 4 categorías principales de Buffettology.
        """
        s = sector_name.lower()
        
        if "bank" in s or "financial" in s:
            category = "Banca y Servicios Financieros"
            metrics = ["ROE", "ROA", "NIM", "Efficiency_Ratio", "Tier1_Capital"]
            balance_type = "No Clasificado (Orden por Liquidez)"
        elif "reit" in s or "real estate" in s:
            category = "Inmobiliario (REITs)"
            metrics = ["FFO", "AFFO", "NAV", "Cap_Rate", "Payout_Ratio"]
            balance_type = "Clasificado / No Clasificado"
        elif "utility" in s or "utilities" in s or "energy" in s:
            category = "Servicios Públicos (Utilities) y Energía"
            metrics = ["Rate_Base_Return", "Regulated_ROE", "Interest_Coverage", "CapEx_Continuity"]
            balance_type = "Clasificado (Capital Intensivo)"
        else:
            category = "Industrial / Tecnología / Consumo"
            metrics = ["ROE", "ROIC", "Gross_Margin", "Operating_Margin", "SGA_to_Gross_Profit", "CapEx_to_Net_Income"]
            balance_type = "Clasificado (Current / Non-current)"

        logger.info(f"[{self.agent_name}] Sector categorizado como: '{category}'")

        return {
            "category": category,
            "balance_structure": balance_type,
            "metrics_to_include": metrics,
            "framework": "Buffettology"
        }

    def configure(self, sector_name: str, industry_name: str = "") -> Dict[str, Any]:
        return self.configure_sector_rules(sector_name, industry_name)