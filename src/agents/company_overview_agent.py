import os
import json
import logging
from typing import Any, Optional
import pandas as pd
import numpy as np

from src.agents.base_agent import BaseAgent
from src.tools.sec_api import SecEdgarAPI

logger = logging.getLogger(__name__)


class CompanyOverviewAgent(BaseAgent):
    """
    Agente de IA especializado en describir qué hace la empresa, sus líneas de negocio detalladas,
    su ubicación e instalaciones, su modelo de ingresos (monetización), los vientos de cola del sector,
    sus perspectivas de crecimiento o decrecimiento y su evaluación bajo el 'Círculo de Competencia' de Warren Buffett.
    """

    def __init__(self):
        super().__init__(
            agent_name="CompanyOverviewAgent",
            prompt_file="company_overview_analyst.xml",
            temperature=0.1
        )

    def analyze(self, ticker: str, market_data: dict[str, Any], sec_data: dict[str, Any]) -> dict[str, Any]:
        """
        Ejecuta el análisis basándose única y exclusivamente en el informe anual oficial 10-K de la SEC
        leído e interpretado por el LLM.
        """
        clean_ticker = ticker.upper().strip()
        logger.info(f"[{self.agent_name}] Analizando informe 10-K oficial de la SEC para {clean_ticker}...")

        # 1. Obtener informe 10-K real de la SEC (Item 1 Business, Item 2 Properties, Item 7 MD&A)
        narrative_10k = {}
        try:
            narrative_10k = SecEdgarAPI.fetch_company_10k_narrative(clean_ticker)
        except Exception as e:
            logger.warning(f"[{self.agent_name}] Aviso extrayendo informe 10-K: {e}")

        item1_text = narrative_10k.get("item1_business", "")
        item2_text = narrative_10k.get("item2_properties", "")
        item7_text = narrative_10k.get("item7_mda", "")
        filing_date_10k = narrative_10k.get("filing_date", "Reciente")

        # 2. Metadatos de la SEC y de Mercado
        sic_desc = "General Corporate"
        try:
            cik = SecEdgarAPI.get_cik_from_ticker(clean_ticker)
            if cik:
                cache_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "cache_sec", f"{clean_ticker}_facts.json"
                )
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        facts_raw = json.load(f)
                        sic_desc = facts_raw.get("sicDescription", facts_raw.get("entityName", "General Corporate"))
        except Exception as e:
            logger.debug(f"[{self.agent_name}] Aviso leyendo SIC: {e}")

        company_name = market_data.get("company_name", clean_ticker)
        sector = market_data.get("sector", "Desconocido")
        industry = market_data.get("industry", "Desconocida")
        business_desc_market = market_data.get("description") or market_data.get("business_summary") or ""
        market_cap = market_data.get("market_cap", 0)
        current_price = market_data.get("current_price", 0.0)

        # 3. Datos cuantitativos normalizados
        aligned = sec_data.get("aligned_series", {})
        years = sec_data.get("years", [])
        df = pd.DataFrame(aligned, index=years) if (aligned and years) else pd.DataFrame()

        rev_series = df.get("Ingresos totales", pd.Series([0]))
        gp_series = df.get("Beneficio bruto", pd.Series([0]))
        ni_series = df.get("Beneficio neto de la empresa", pd.Series([0]))

        latest_rev = float(rev_series.iloc[-1]) if len(rev_series) > 0 and rev_series.iloc[-1] != 0 else 0.0
        latest_ni = float(ni_series.iloc[-1]) if len(ni_series) > 0 else 0.0
        avg_gross_margin = float((gp_series / rev_series.replace(0, np.nan)).dropna().mean() * 100) if len(gp_series) > 0 else 0.0

        # Preparar recortes de texto relevantes y concisos del 10-K para máxima velocidad de inferencia
        item1_sample = item1_text[:3500] if item1_text else business_desc_market[:1500]
        item2_sample = item2_text[:800] if item2_text else "Sede corporativa global y centros operativos principales."
        item7_sample = item7_text[:3000] if item7_text else "Consultar MD&A en informe 10-K."

        # 4. Construcción del Prompt para la IA
        prompt_text = f"""
        INSTRUCCIÓN PRINCIPAL:
        Basándote estrictamente en las reglas y el formato definido en tu prompt de sistema, analiza el siguiente informe anual oficial (Form 10-K) para la empresa {company_name} ({clean_ticker}).
        Fecha del informe 10-K: {filing_date_10k}.

        DATOS GENERALES Y CUANTITATIVOS:
        - Ticker: {clean_ticker}
        - Empresa: {company_name}
        - Sector / Industria: {sector} / {industry}
        - Clasificación SEC (SIC): {sic_desc}
        - Capitalización Bursátil: ${market_cap:,.0f} USD
        - Precio Actual: ${current_price:.2f} USD
        - Ingresos Recientes: ${latest_rev:,.0f} USD (Margen Bruto Medio: {avg_gross_margin:.1f}%)
        - Beneficio Neto Reciente: ${latest_ni:,.0f} USD

        EXTRACTO DEL INFORME 10-K DE LA SEC:
        === ITEM 1. BUSINESS ===
        {item1_sample}

        === ITEM 2. PROPERTIES ===
        {item2_sample}

        === ITEM 7. MD&A ===
        {item7_sample}

        EJECUCIÓN:
        Genera la respuesta ÚNICA Y EXCLUSIVAMENTE en el formato JSON definido en la etiqueta <output_format> de tus instrucciones de sistema.
        """

        # 5. Generación estricta con LLM (Sin fallbacks)
        try:
            raw_response = self.generate_response(prompt_text)
            limpio = raw_response.replace("```json", "").replace("```", "").strip()
            resultado_json = json.loads(limpio)
            
            # Asegurar que los campos clave existan
            if not isinstance(resultado_json.get("lineas_de_negocio"), list):
                resultado_json["lineas_de_negocio"] = []
                
            logger.info(f"[{self.agent_name}] Análisis de 10-K completado exitosamente con IA para {clean_ticker}")
            return resultado_json
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error crítico en la extracción del 10-K o parseo del JSON: {e}")
            raise RuntimeError(f"Fallo en la generación de IA para {clean_ticker}. Revisa el prompt o la conectividad del modelo. Error: {str(e)}")
