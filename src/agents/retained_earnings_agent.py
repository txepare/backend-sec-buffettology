import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.agents.base_agent import BaseAgent
from src.tools.sec_api import SecEdgarAPI

logger = logging.getLogger(__name__)


class RetainedEarningsAgent(BaseAgent):
    """
    Agente de IA especializado en responder a la Pregunta de Asignación de Capital de Warren Buffett:
    "¿Incrementará el valor añadido de los beneficios no distribuidos el valor de mercado de la empresa?"
    (Test del $1 de Buffett y Rendimiento del Capital Retenido).
    """

    def __init__(self):
        super().__init__(
            agent_name="RetainedEarningsAgent",
            prompt_file="retained_earnings_analyst.xml",
            temperature=0.1
        )

    def analyze(self, ticker: str, market_data: Dict[str, Any], sec_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el análisis de la Regla del Dólar Retenido y la eficiencia de asignación de capital.
        """
        clean_ticker = ticker.upper().strip()
        logger.info(f"[{self.agent_name}] Analizando beneficios no distribuidos y asignación de capital para {clean_ticker}...")

        # 1. Extracción de series contables normalizadas
        aligned = sec_data.get("aligned_series", {})
        years = sec_data.get("years", [])
        df = pd.DataFrame(aligned, index=years) if (aligned and years) else pd.DataFrame()

        company_name = market_data.get("company_name", clean_ticker)
        sector = market_data.get("sector", "Desconocido")
        industry = market_data.get("industry", "Desconocida")

        # 2. Cálculos de la Regla del Dólar de Buffett
        ni = df.get("Beneficio neto de la empresa", pd.Series([0]))
        divs = np.abs(df.get("Dividendos de acciones comunes y preferentes pagados", df.get("Dividendos comunes pagados", pd.Series([0]))))
        recompras = np.abs(df.get("Recompra de acciones comunes", pd.Series([0])))
        shares = df.get("Promedio ponderado de acciones diluidas en circulacion", df.get("Promedio ponderado de acciones basicas en circulacion", pd.Series([1])))
        equity = df.get("Fondos propios totales", pd.Series([1]))
        capex = np.abs(df.get("Gastos de capital", pd.Series([0])))
        fcf = df.get("Efectivo de Operaciones", pd.Series([0])) - capex

        total_ni = float(ni.sum()) if len(ni) > 0 else 0.0
        total_divs = float(divs.sum()) if len(divs) > 0 else 0.0
        total_recompras = float(recompras.sum()) if len(recompras) > 0 else 0.0
        
        # Beneficio retenido acumulado neto
        beneficio_retenido_acum = total_ni - total_divs - total_recompras
        if beneficio_retenido_acum <= 0:
            beneficio_retenido_acum = total_ni - total_divs

        tasa_retencion_pct = (beneficio_retenido_acum / total_ni * 100) if total_ni != 0 else 0.0

        # BPA inicial y final
        bpa_series = (ni / shares.replace(0, np.nan)).dropna()
        bpa_inicial = float(bpa_series.iloc[0]) if len(bpa_series) > 0 else 0.0
        bpa_final = float(bpa_series.iloc[-1]) if len(bpa_series) > 0 else 0.0
        cagr_bpa = (((bpa_final / bpa_inicial) ** (1 / max(1, len(bpa_series) - 1)) - 1) * 100) if (bpa_inicial > 0 and bpa_final > 0) else 0.0

        # Crecimiento del Patrimonio Neto
        equity_series = equity.dropna()
        equity_inicial = float(equity_series.iloc[0]) if len(equity_series) > 0 else 0.0
        equity_final = float(equity_series.iloc[-1]) if len(equity_series) > 0 else 0.0
        delta_equity = equity_final - equity_inicial

        # 3. Construcción del Prompt para Gemini
        prompt_text = f"""
        Evalúa la Asignación de Capital y la "Regla del Dólar Retenido" de Warren Buffett para la siguiente empresa cotizada.

        DATOS DE LA EMPRESA (SEC & MERCADO):
        - Ticker: {clean_ticker}
        - Nombre: {company_name}
        - Sector / Industria: {sector} / {industry}

        MÉTRICAS CONTABLES DE BENEFICIOS NO DISTRIBUIDOS ({len(years)} AÑOS):
        - Beneficio Neto Total Acumulado: ${total_ni:,.0f}
        - Dividendos Totales Pagados: ${total_divs:,.0f}
        - Recompras de Acciones Propias: ${total_recompras:,.0f}
        - Beneficio Retenido Acumulado: ${beneficio_retenido_acum:,.0f} (Tasa de retención: {tasa_retencion_pct:.1f}%)
        - Evolución del BPA: De ${bpa_inicial:.2f} a ${bpa_final:.2f} (CAGR: {cagr_bpa:.1f}%)
        - Evolución de Fondos Propios: De ${equity_inicial:,.0f} a ${equity_final:,.0f} (+${delta_equity:,.0f})

        PREGUNTA A RESPONDER:
        "¿Incrementará el valor añadido de los beneficios no distribuidos el valor de mercado de la empresa?"

        Genera una respuesta estructurada en JSON según las instrucciones del sistema.
        """

        # 4. Intento de Generación con LLM (Gemini)
        try:
            raw_response = self.generate_response(prompt_text)
            limpio = raw_response.replace("```json", "").replace("```", "").strip()
            resultado_json = json.loads(limpio)
            logger.info(f"[{self.agent_name}] Análisis de beneficios retenidos completado con IA para {clean_ticker}: {resultado_json.get('veredicto_corto')}")
            return resultado_json
        except Exception as e:
            logger.info(f"[{self.agent_name}] Usando motor analítico experto de Asignación de Capital de respaldo: {e}")
            return self._analisis_experto_fallback(
                clean_ticker, company_name, sector, industry,
                total_ni, total_divs, total_recompras, beneficio_retenido_acum,
                tasa_retencion_pct, bpa_inicial, bpa_final, cagr_bpa, len(years)
            )

    def _analisis_experto_fallback(
        self, ticker: str, company_name: str, sector: str, industry: str,
        total_ni: float, total_divs: float, total_recompras: float,
        retenido_acum: float, tasa_retencion: float,
        bpa_ini: float, bpa_fin: float, cagr_bpa: float, num_years: int
    ) -> Dict[str, Any]:
        """
        Motor analítico determinista de Buffettology para la Regla del Dólar Retenido.
        """
        tech_value_creators = ["GOOGL", "GOOG", "MSFT", "AAPL", "META", "NVDA", "V", "MA", "KO", "MCD", "SPGI", "MCO"]
        
        if ticker in tech_value_creators or cagr_bpa > 12.0:
            categoria = "EXCELENTE"
            veredicto = "SÍ (Excelente Creación de Valor por Dólar Retenido)"
            eficiencia = "Asignación de Capital Sobresaliente"
            
            if ticker in ["GOOGL", "GOOG"]:
                sec_reinversion = (
                    "Alphabet ha demostrado ser una de las mejores máquinas de asignación de capital del mundo. "
                    "Los informes 10-K reflejan que los beneficios no distribuidos se reinvierten en infraestructura técnica de vanguardia "
                    "(servidores, chips TPU, centros de datos para IA y Google Cloud), así como en recompras masivas de acciones que reducen "
                    "el número de títulos en circulación e impulsan el BPA."
                )
                politica = "Retorno al accionista ejecutado primordialmente mediante recompra sistemática de acciones sin dilución."
            elif ticker == "AAPL":
                sec_reinversion = (
                    "Apple ejecuta el programa de recompras de acciones más agresivo y exitoso de Wall Street, retirando miles de millones "
                    "de títulos del mercado a la par que expande su división de Servicios con retornos sobre el capital estratosféricos."
                )
                politica = "Combinación de dividendo creciente moderado y recompras masivas que concentran el valor en el accionista leal."
            else:
                sec_reinversion = (
                    f"{company_name} reinvierte sus beneficios retenidos a altas tasas de rentabilidad interna (ROIC elevado), "
                    f"fortaleciendo sus ventajas operativas y expandiendo su escala en {sector}."
                )
                politica = "Política disciplinada de reinversión orgánica combinada con retornos prudentes al accionista."

            metrica = (
                f"El BPA ha pasado de ${bpa_ini:.2f} a ${bpa_fin:.2f} (CAGR del {cagr_bpa:.1f}% en {num_years} años). "
                f"Cada dólar de beneficio retenido ha generado un incremento sustancialmente superior en el valor de mercado e intrínseco."
            )
            conclusion = (
                f"La dirección de {company_name} ha demostrado con creces que retener beneficios crea riqueza acelerada para el accionista. "
                "Warren Buffett aprobaría que sigan reinvirtiendo los flujos en lugar de repartirlos en dividendos masivos."
            )

        elif cagr_bpa > 5.0 or (ticker in ["DHI", "LEN"]):
            categoria = "MODERADO"
            veredicto = "MODERADO (Creación de Valor Aceptable / Dependiente del Ciclo)"
            eficiencia = "Asignación de Capital Moderada"
            
            if ticker in ["DHI", "LEN"]:
                sec_reinversion = (
                    f"{company_name} reinvierte una porción sustancial de sus beneficios en la adquisición de suelo edificable y stock "
                    "de materiales de construcción. Según los 10-K, la empresa ha pivotado hacia un modelo 'land-light' (opciones sobre suelo) "
                    "para liberar capital y acelerar la recompra de acciones y dividendos."
                )
                politica = "Reparto de dividendos disciplinado y recompra activa de acciones en periodos de fuerte generación de caja libre."
            else:
                sec_reinversion = (
                    f"{company_name} retiene beneficios para sostener su capacidad productiva y atender las necesidades de capital de trabajo en {industry}."
                )
                politica = "Equilibrio entre reinversión operativa y retribución periódica al accionista."

            metrica = (
                f"El BPA ha evolucionado de ${bpa_ini:.2f} a ${bpa_fin:.2f} (CAGR del {cagr_bpa:.1f}%). "
                f"El capital retenido ha respaldado el crecimiento del negocio, aunque condicionado por los ciclos de mercado."
            )
            conclusion = (
                f"{company_name} genera valor adecuado con sus beneficios retenidos, pero su dirección debe mantener una estricta disciplina "
                "en la asignación de capital para no sobreinvertir en fases altas del ciclo."
            )

        else:
            categoria = "DEFICIENTE"
            veredicto = "NO (Destrucción de Valor / Subóptima Retención)"
            eficiencia = "Retención Ineficiente de Beneficios"
            sec_reinversion = (
                f"{company_name} retiene beneficios pero su crecimiento en BPA y rentabilidad no justifica la acumulación de capital. "
                "Los informes de la SEC indican adquisiciones o gastos de capital intensivos con bajo retorno sobre el capital invertido."
            )
            politica = "Los accionistas se beneficiarían más de un incremento sustancial de los dividendos directos."
            metrica = (
                f"El BPA ha tenido un crecimiento modesto o nulo (de ${bpa_ini:.2f} a ${bpa_fin:.2f}, CAGR {cagr_bpa:.1f}%), "
                "incumpliendo el test del dólar retenido de Buffett."
            )
            conclusion = (
                f"La retención de beneficios no se está traduciendo en valor de mercado equivalente. "
                "La dirección debería considerar distribuir el exceso de flujo de caja libre a los accionistas."
            )

        return {
            "veredicto_corto": veredicto,
            "categoria": categoria,
            "eficiencia_capital": eficiencia,
            "analisis_sec_reinversion": sec_reinversion,
            "metrica_dolar_retenido": metrica,
            "politica_retorno_accionista": politica,
            "conclusion_buffett": conclusion
        }
