import os
import logging
from datetime import datetime
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
        monopoly_analysis: Dict[str, Any] = None,
        retained_earnings_analysis: Dict[str, Any] = None,
        management_analysis: Dict[str, Any] = None,
        forensic_analysis: Dict[str, Any] = None,
        **kwargs
    ) -> str:
        """
        Llama al constructor de PDF y compila el informe en la carpeta de salidas.
        """
        if market_data is None and analysis_data:
            market_data = analysis_data.get("market_data", {})
        if sec_data is None and analysis_data:
            sec_data = analysis_data.get("sec_data", analysis_data.get("normalized", {}))
        if sector_config is None and analysis_data:
            sector_config = analysis_data.get("sector_config", {})
        if monopoly_analysis is None and analysis_data:
            monopoly_analysis = analysis_data.get("monopoly_analysis", {})
        if retained_earnings_analysis is None and analysis_data:
            retained_earnings_analysis = analysis_data.get("retained_earnings_analysis", {})
        if management_analysis is None and analysis_data:
            management_analysis = analysis_data.get("management_analysis", {})
        if forensic_analysis is None and analysis_data:
            forensic_analysis = analysis_data.get("forensic_analysis", {})

        market_data = market_data or {}
        sec_data = sec_data or {}
        sector_config = sector_config or {}
        monopoly_analysis = monopoly_analysis or kwargs.get("monopoly_analysis")
        retained_earnings_analysis = retained_earnings_analysis or kwargs.get("retained_earnings_analysis")
        management_analysis = management_analysis or kwargs.get("management_analysis")
        forensic_analysis = forensic_analysis or kwargs.get("forensic_analysis")

        current_price = market_data.get("current_price", 0.0)

        # Formato de nombre: TICKER_PRECIO ACTUAL DE COTIZACIÓN_FECHA (Ej: TMDX_80.75_2026-08-05.pdf)
        try:
            price_val = float(current_price)
            price_str = f"{price_val:.2f}"
        except (ValueError, TypeError):
            price_str = str(current_price)

        date_str = datetime.now().strftime("%Y-%m-%d")
        pdf_filename = f"{ticker.upper()}_{price_str}_{date_str}.pdf"
        output_path = os.path.join(OUTPUT_DIR, pdf_filename)
        logger.info(f"[{self.agent_name}] Generando PDF en: {output_path}")

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
            sector_config=sector_config,
            monopoly_analysis=monopoly_analysis,
            retained_earnings_analysis=retained_earnings_analysis,
            management_analysis=management_analysis,
            forensic_analysis=forensic_analysis
        )

        return result_path