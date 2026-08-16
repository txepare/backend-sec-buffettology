import os
import logging
import pandas as pd
from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.tools.pdf_builder import PDFBuilder
from config.settings import OUTPUT_DIR

logger = logging.getLogger(__name__)


class PDFGeneratorAgent(BaseAgent):
    """
    Sub-agente responsable de coordinar la compilación del documento PDF final.
    """

    def __init__(self):
        super().__init__(
            agent_name="PDFGeneratorAgent",
            prompt_file="pdf_generator.xml",
            temperature=0.2
        )

    def generate_pdf(
        self,
        ticker: str,
        market_data: Dict[str, Any] = None,
        sec_data: Dict[str, Any] = None,
        sector_config: Dict[str, Any] = None,
        company_name: str = "",
        sector: str = "",
        analysis_data: Dict[str, Any] = None,
        **kwargs
    ) -> str:
        """
        Llama al constructor de PDF de ReportLab y compila el informe en la carpeta de salidas.
        Acepta tanto argumentos nombrados (market_data, sec_data, sector_config) como el diccionario analysis_data.
        """
        if market_data is None and analysis_data:
            market_data = analysis_data.get("market_data", {})
        if sec_data is None and analysis_data:
            sec_data = analysis_data.get("sec_data", analysis_data.get("normalized", {}))
        if sector_config is None and analysis_data:
            sector_config = analysis_data.get("sector_config", {})

        market_data = market_data or {}
        sec_data = sec_data or {}
        sector_config = sector_config or {}

        output_path = os.path.join(OUTPUT_DIR, f"{ticker.upper()}_Buffettology_Report.pdf")
        logger.info(f"[{self.agent_name}] Generando PDF en: {output_path}")

        current_price = market_data.get("current_price", 0.0)

        # Reconstruir el DataFrame a partir de los datos normalizados
        years = sec_data.get("years", [])
        aligned_series = sec_data.get("aligned_series", {})

        # Convertir a DataFrame de Pandas
        df_financials = pd.DataFrame(aligned_series, index=years).reset_index()
        df_financials.rename(columns={'index': 'Periodo Fiscal'}, inplace=True)
        if 'Periodo Fiscal' not in df_financials.columns:
            df_financials['Periodo Fiscal'] = years

        # Llamar al constructor con la firma de variables
        result_path = PDFBuilder.generate_pdf_report(
            ticker=ticker,
            current_price=current_price,
            df_financials=df_financials,
            output_pdf_path=output_path,
            sector_config=sector_config
        )

        return result_path