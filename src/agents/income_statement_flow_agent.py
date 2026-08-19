import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.agents.base_agent import BaseAgent
from src.tools.sankey_builder import SankeyFlowBuilder

logger = logging.getLogger(__name__)


class IncomeStatementFlowAgent(BaseAgent):
    """
    Agente de IA responsable de estructurar las partidas contables del último ejercicio fiscal
    y generar el diagrama de flujo Sankey del estado de resultados (al estilo 'How They Make Money').
    """

    def __init__(self):
        super().__init__(
            agent_name="IncomeStatementFlowAgent",
            prompt_file="pdf_generator.xml",
            temperature=0.0
        )

    @staticmethod
    def _determinar_perfil_coste_segmento(nombre: str, idx: int, n_total: int) -> str:
        """
        Determina la característica de costes y rentabilidad de la línea de negocio para enriquecer el Sankey.
        """
        n_low = nombre.lower()
        if any(k in n_low for k in ["servicios", "services", "cloud", "software", "financier", "financial", "hipotec", "comision", "licenc", "app store", "advertising", "publicidad", "suscrip", "prime"]):
            return "Alto Margen / Escalable"
        elif any(k in n_low for k in ["concentrad", "jarabes", "concesion"]):
            return "Margen Máximo / Monopolio"
        elif any(k in n_low for k in ["construcc", "homebuilding", "vivienda", "hardware", "iphone", "dispositiv", "devices", "maquinaria", "machinery", "automoci", "automotive", "vehicul", "embotellad", "recursos", "minería", "resource"]):
            return "Intensivo en COGS / Producción"
        elif any(k in n_low for k in ["alquiler", "rental", "leasing", "suelo", "forestar", "lotes", "inmueble"]):
            return "Capital Rotativo / Expansión"
        elif any(k in n_low for k in ["energía", "energy", "transporte", "power"]):
            return "Bienes de Equipo / Flota"
        elif idx == 0:
            return "Línea Central de Volumen"
        else:
            return "Línea Complementaria"

    def prepare_flow_data(
        self,
        ticker: str,
        market_data: Dict[str, Any],
        sec_data: Dict[str, Any],
        segments_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcula las partidas del estado de resultados del último año fiscal disponible
        y las fusiona con los segmentos de ingresos y sus perfiles de costes.
        """
        clean_ticker = ticker.upper().strip()
        company_name = market_data.get("company_name", clean_ticker)
        logger.info(f"[{self.agent_name}] Preparando flujo de estado de resultados para {clean_ticker}...")

        aligned = sec_data.get("aligned_series", {})
        years = sec_data.get("years", [])
        df = pd.DataFrame(aligned, index=years) if (aligned and years) else pd.DataFrame()

        latest_year = str(years[-1]) if years else "2024"

        # Partidas contables del último año
        rev = float(df["Ingresos totales"].iloc[-1]) if "Ingresos totales" in df.columns and len(df) > 0 else 100.0
        cogs = float(abs(df["Coste de los bienes vendidos"].iloc[-1])) if "Coste de los bienes vendidos" in df.columns and len(df) > 0 else (rev * 0.45)
        gp = float(df["Beneficio bruto"].iloc[-1]) if "Beneficio bruto" in df.columns and len(df) > 0 else (rev - cogs)
        
        rd = float(abs(df["Gastos de I + D"].iloc[-1])) if "Gastos de I + D" in df.columns and len(df) > 0 else 0.0
        sga = float(abs(df["Gastos de venta generales y administrativos"].iloc[-1])) if "Gastos de venta generales y administrativos" in df.columns and len(df) > 0 else 0.0
        
        net_income = float(df["Beneficio neto de la empresa"].iloc[-1]) if "Beneficio neto de la empresa" in df.columns and len(df) > 0 else (gp * 0.40)
        tax = float(abs(df["Gastos de impuestos"].iloc[-1])) if "Gastos de impuestos" in df.columns and len(df) > 0 else max((gp - rd - sga - net_income) * 0.20, 0.0)

        # Si Beneficio Bruto no está explícito
        if gp <= 0 or gp < net_income:
            gp = max(rev - cogs, net_income * 1.3)

        op_profit = max(gp - (rd + sga), net_income + tax)
        opex_total = rd + sga if (rd + sga) > 0 else max(gp - op_profit, 1.0)

        financial_flow = {
            "revenue": rev,
            "cogs": cogs,
            "gross_profit": gp,
            "rd": rd,
            "sga": sga,
            "opex": opex_total,
            "operating_profit": op_profit,
            "tax": tax,
            "net_profit": net_income
        }

        # Extraer segmentos formateados para el Sankey
        formatted_segments = []
        if segments_data and "segmentos" in segments_data:
            historico = segments_data.get("historico_segmentos", {})
            seg_list = segments_data.get("segmentos", [])
            unit_str = segments_data.get("unidad_monetaria", "Billion USD")
            multiplier = 1e9 if "Billion" in unit_str else 1e6
            n_tot = len(seg_list)

            for idx, s in enumerate(seg_list):
                s_name = s.get("nombre", f"Segmento {idx+1}")
                pct = float(s.get("porcentaje_ultimo_ano", (100.0 / n_tot)))
                yoy_val = s.get("crecimiento_yoy_pct")
                yoy_str = f"{yoy_val:+.1f}% YoY" if yoy_val is not None and isinstance(yoy_val, (int, float)) else ""
                
                # Monto en dólares absolutos
                if s_name in historico and len(historico[s_name]) > 0:
                    last_val = historico[s_name][-1]
                    val_abs = float(last_val) * multiplier if (isinstance(last_val, (int, float)) and last_val > 0) else (pct / 100.0) * rev
                else:
                    val_abs = (pct / 100.0) * rev

                perfil_coste = self._determinar_perfil_coste_segmento(s_name, idx, n_tot)

                formatted_segments.append({
                    "nombre": s_name,
                    "monto": val_abs,
                    "pct": pct,
                    "yoy": yoy_str,
                    "perfil": perfil_coste
                })

        return {
            "ticker": clean_ticker,
            "company_name": company_name,
            "year": latest_year,
            "segments_data": formatted_segments,
            "financial_flow": financial_flow
        }

    def generate_flow_chart(self, flow_prepared_data: Dict[str, Any]):
        """
        Genera la figura de matplotlib con el diagrama Sankey del estado de resultados.
        """
        return SankeyFlowBuilder.generate_sankey_figure(
            ticker=flow_prepared_data.get("ticker", ""),
            company_name=flow_prepared_data.get("company_name", ""),
            year=flow_prepared_data.get("year", "2024"),
            segments_data=flow_prepared_data.get("segments_data", []),
            financial_flow=flow_prepared_data.get("financial_flow", {})
        )

