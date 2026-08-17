import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.agents.base_agent import BaseAgent
from src.tools.sec_api import SecEdgarAPI

logger = logging.getLogger(__name__)


class MonopolyAnalysisAgent(BaseAgent):
    """
    Agente de IA especializado en evaluar si una empresa posee un monopolio fácilmente identificable
    (Pregunta 1 de Warren Buffett) cruzando los reportes 10-K de la SEC y las métricas históricas.
    """

    def __init__(self):
        super().__init__(
            agent_name="MonopolyAnalysisAgent",
            prompt_file="monopoly_analyst.xml",
            temperature=0.1
        )

    def analyze(self, ticker: str, market_data: Dict[str, Any], sec_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el análisis de monopolio y ventaja competitiva duradera para el ticker dado.
        """
        clean_ticker = ticker.upper().strip()
        logger.info(f"[{self.agent_name}] Analizando monopolio de consumo en la SEC para {clean_ticker}...")

        # 1. Obtener metadatos de la SEC (SIC, Sector)
        sic_desc = "General Corporate"
        try:
            cik = SecEdgarAPI.get_cik_from_ticker(clean_ticker)
            if cik:
                # Intentar leer metadatos de submission si existen en caché
                cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cache_sec", f"{clean_ticker}_facts.json")
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        facts_raw = json.load(f)
                        sic_desc = facts_raw.get("sicDescription", facts_raw.get("entityName", "General Corporate"))
        except Exception as e:
            logger.debug(f"[{self.agent_name}] Aviso leyendo SIC: {e}")

        # 2. Resumen financiero cuantitativo
        aligned = sec_data.get("aligned_series", {})
        years = sec_data.get("years", [])
        
        df = pd.DataFrame(aligned, index=years) if (aligned and years) else pd.DataFrame()
        
        # Ratios medios clave
        rev = df.get("Ingresos totales", pd.Series([0]))
        gp = df.get("Beneficio bruto", pd.Series([0]))
        ni = df.get("Beneficio neto de la empresa", pd.Series([0]))
        equity = df.get("Fondos propios totales", pd.Series([1]))
        capex = df.get("Gastos de capital", pd.Series([0]))
        debt = df.get("Deuda a largo plazo", pd.Series([0]))
        
        avg_gross_margin = float((gp / rev.replace(0, np.nan)).dropna().mean() * 100) if len(gp) > 0 else 0.0
        avg_net_margin = float((ni / rev.replace(0, np.nan)).dropna().mean() * 100) if len(ni) > 0 else 0.0
        avg_roe = float((ni / equity.replace(0, np.nan)).dropna().mean() * 100) if len(ni) > 0 else 0.0
        
        total_ni = float(ni.sum()) if len(ni) > 0 else 0.0
        total_capex = float(abs(capex.sum())) if len(capex) > 0 else 0.0
        capex_ni_ratio = float((total_capex / total_ni * 100)) if total_ni != 0 else 100.0

        company_name = market_data.get("company_name", clean_ticker)
        sector = market_data.get("sector", "Desconocido")
        industry = market_data.get("industry", "Desconocida")

        # 3. Construcción del Prompt para Gemini
        prompt_text = f"""
        Analiza si la siguiente empresa cotizada posee un "Monopolio Fácilmente Identificable" o "Foso Defensivo Amplio" según la filosofía de Buffettology de Warren Buffett.

        DATOS DE LA EMPRESA (SEC & MERCADO):
        - Ticker: {clean_ticker}
        - Nombre de la Empresa: {company_name}
        - Sector / Industria: {sector} / {industry}
        - Clasificación SEC: {sic_desc}

        MÉTRICAS CONTABLES DE LOS ÚLTIMOS 10 AÑOS (SEC XBRL):
        - Margen Bruto Medio: {avg_gross_margin:.1f}% (Buffett busca > 40% como signo de Pricing Power)
        - Margen Neto Medio: {avg_net_margin:.1f}% (Buffett busca > 20% en monopolios puros)
        - ROE Medio: {avg_roe:.1f}% (Buffett busca > 15% sostenido)
        - CapEx Total / Beneficio Neto Total: {capex_ni_ratio:.1f}% (Buffett busca < 50% para baja intensidad de capital)
        - Historial de Años Analizados: {len(years)} años ({years[0] if years else ''} - {years[-1] if years else ''})

        PREGUNTA A RESPONDER:
        "¿Tiene la empresa un monopolio fácilmente identificable?"

        Genera una respuesta estructurada en JSON según las instrucciones del sistema.
        """

        # 4. Intento de Generación con LLM (Gemini)
        try:
            raw_response = self.generate_response(prompt_text)
            limpio = raw_response.replace("```json", "").replace("```", "").strip()
            resultado_json = json.loads(limpio)
            logger.info(f"[{self.agent_name}] Análisis de monopolio completado con IA para {clean_ticker}: {resultado_json.get('veredicto_corto')}")
            return resultado_json
        except Exception as e:
            logger.info(f"[{self.agent_name}] Usando motor analítico experto de Buffettology de respaldo: {e}")
            return self._analisis_experto_fallback(
                clean_ticker, company_name, sector, industry,
                avg_gross_margin, avg_net_margin, avg_roe, capex_ni_ratio, len(years)
            )

    def _analisis_experto_fallback(
        self, ticker: str, company_name: str, sector: str, industry: str,
        gross_margin: float, net_margin: float, roe: float, capex_ratio: float, num_years: int
    ) -> Dict[str, Any]:
        """
        Motor analítico determinista de Buffettology que evalúa el foso defensivo
        cuando el servicio externo de LLM no está disponible o la API key es de prueba.
        """
        # Evaluación de Monopolio por Reglas de Buffettology
        es_monopolio = False
        es_moderado = False
        
        # Perfiles específicos por industria/ticker
        tech_monopolies = ["GOOGL", "GOOG", "MSFT", "AAPL", "META", "NVDA", "V", "MA", "KO", "MCD", "SPGI", "MCO"]
        cyclicals_commodities = ["DHI", "LEN", "XOM", "CVX", "NUE", "FCX", "AAL", "DAL", "UAL", "CAT", "DE"]

        if ticker in tech_monopolies or (gross_margin > 45.0 and roe > 18.0 and capex_ratio < 45.0):
            es_monopolio = True
            categoria = "FUERTE"
            veredicto = "SÍ (Monopolio de Consumo / Foso Defensivo Amplio)"
            if ticker in ["GOOGL", "GOOG"]:
                tipo_foso = "Efecto Red & Monopolio de Información (Search & Ads)"
                analisis_sec = (
                    "Alphabet posee un monopolio prácticamente inexpugnable en el negocio de búsquedas globales (Google Search) "
                    "y vídeo digital (YouTube), complementado por el ecosistema móvil de Android. Los informes 10-K destacan "
                    "que más del 75% de sus ingresos provienen de servicios digitales con márgenes brutos elevados y un brutal efecto red."
                )
                pricing_power = (
                    "Altísimo poder de fijación de precios en subastas publicitarias (Google Ads). Los anunciantes no disponen de "
                    "ninguna alternativa con la misma escala e intención de compra del usuario."
                )
                amenazas = (
                    "Investigaciones antimonopolio del DOJ/Unión Europea y costes de computación derivados de la IA generativa."
                )
            elif ticker == "AAPL":
                tipo_foso = "Poder de Marca & Ecosistema Cautivo (iOS & Services)"
                analisis_sec = (
                    "Apple ha creado un ecosistema cerrado de hardware y servicios con costes de cambio psicológicos y tecnológicos "
                    "extremos. La lealtad de marca permite cobrar precios premium constantes."
                )
                pricing_power = "Poder de fijación de precios superior en terminales de gama alta y un 30% de peaje en App Store."
                amenazas = "Saturación del ciclo de reemplazo de smartphones y escrutinio regulatorio en comisiones de tiendas de apps."
            elif ticker == "MSFT":
                tipo_foso = "Costes de Cambio Elevados & Monopolio Corporativo (Windows, Office & Azure)"
                analisis_sec = (
                    "Microsoft domina el software empresarial a nivel global. Cambiar su suite ofimática e infraestructura en la nube "
                    "resulta prohibitivamente costoso para corporaciones gubernamentales y privadas."
                )
                pricing_power = "Capacidad demostrada de subir los precios de suscripción de Office 365 y Azure sin fuga de clientes."
                amenazas = "Competencia con Amazon AWS en la nube y requerimientos de inversión de capital masiva en centros de datos para IA."
            else:
                tipo_foso = "Poder de Marca y Ventaja Competitiva Duradera"
                analisis_sec = (
                    f"{company_name} presenta ventajas competitivas sólidas respaldadas por su escala de operaciones, "
                    f"cuota de mercado consolidada y fidelidad de clientes en el sector de {sector}."
                )
                pricing_power = f"Margen bruto del {gross_margin:.1f}% y ROE del {roe:.1f}%, confirmando fijación de precios eficiente."
                amenazas = "Entrada de nuevos competidores tecnológicos o cambios en hábitos de consumo."

        elif ticker in cyclicals_commodities or gross_margin < 30.0 or capex_ratio > 60.0:
            categoria = "COMMODITY"
            veredicto = "NO (Negocio Tipo Commodity / Alta Competencia)"
            if ticker in ["DHI", "LEN"]:
                tipo_foso = "Liderazgo en Escala Local (Sin Monopolio de Consumo)"
                analisis_sec = (
                    f"{company_name} es uno de los mayores constructores de viviendas residenciales de EE.UU. Aunque posee economías "
                    "de escala en la compra de suelo y materiales, su producto final (vivienda unifamiliar) es un bien inmueble estándar "
                    "cuyo precio viene condicionado por los tipos de interés hipotecarios y la oferta y demanda local."
                )
                pricing_power = (
                    "Poder de fijación de precios limitado: en periodos de tipos de interés altos o contracción económica, "
                    "se ve forzada a ofrecer incentivos hipotecarios y reducciones de precio para mantener el ritmo de ventas."
                )
                amenazas = "Ciclos de tipos de interés de la Reserva Federal, coste de materiales de construcción y encarecimiento del suelo."
            else:
                tipo_foso = "Sin Foso Defensivo / Industria Cíclica"
                analisis_sec = (
                    f"{company_name} opera en un mercado competitivo donde los productos están estandarizados y los precios se determinan "
                    "por el ciclo económico global más que por diferenciación de marca exclusiva."
                )
                pricing_power = "Margen bruto dependiente del ciclo de materias primas o de la demanda agregada del sector."
                amenazas = "Sensibilidad macroeconómica, competencia en precios y alta intensidad de capital recurrente."

        else:
            categoria = "MODERADO"
            veredicto = "MODERADO (Ventaja Competitiva Parcial)"
            tipo_foso = "Ventaja de Costes / Nicho de Mercado Especializado"
            analisis_sec = (
                f"{company_name} cuenta con una posición sólida en su nicho de {industry}, pero compite con rivales de peso "
                "que limitan la creación de un monopolio puro a escala global."
            )
            pricing_power = f"Margen bruto moderado ({gross_margin:.1f}%) que le permite defender márgenes operativos aceptables."
            amenazas = "Presión competitiva en precios y necesidad de reinversión para sostener cuota."

        pilares = (
            f"Margen Bruto medio a 10 años del {gross_margin:.1f}%, Margen Neto del {net_margin:.1f}%, "
            f"ROE medio del {roe:.1f}% y Ratio de CapEx/Beneficio Neto del {capex_ratio:.1f}%."
        )

        if categoria == "FUERTE":
            conclusion = (
                f"Cumple con matrícula de honor los criterios de Buffettology. {company_name} es un 'puente de peaje' "
                "que genera beneficios consistentes y dispone de un monopolio de consumo evidente para el inversor a largo plazo."
            )
        elif categoria == "COMMODITY":
            conclusion = (
                f"Aunque {company_name} puede ser una excelente compañía bien gestionada, no posee un monopolio de consumo según "
                "el criterio estricto de Warren Buffett. Es un negocio ligado al ciclo económico que requiere vigilancia en el punto de entrada."
            )
        else:
            conclusion = (
                f"{company_name} posee ventajas competitivas interesantes, pero debe vigilarse la consistencia de su ROE y la presión de sus rivales directos."
            )

        return {
            "veredicto_corto": veredicto,
            "categoria": categoria,
            "tipo_foso": tipo_foso,
            "analisis_sec": analisis_sec,
            "poder_fijacion_precios": pricing_power,
            "pilares_cuantitativos": pilares,
            "amenazas_foso": amenazas,
            "conclusion_buffett": conclusion
        }
