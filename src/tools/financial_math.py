import math
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class FinancialMath:
    """
    Motor determinista de cálculos financieros y métricas de Buffettology.
    Garantiza precisión matemática exacta sin depender de inferencias de LLM.
    """

    @staticmethod
    def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
        """
        Calcula la Tasa de Crecimiento Anual Compuesto (CAGR).
        """
        if periods <= 0 or start_value <= 0 or end_value <= 0:
            return 0.0
        try:
            return ((end_value / start_value) ** (1.0 / periods)) - 1.0
        except Exception:
            return 0.0

    @staticmethod
    def calculate_retained_earnings_return(
        eps_series: List[float], 
        dps_series: List[float]
    ) -> Dict[str, Any]:
        """
        Calcula el Retorno sobre las Utilidades Retenidas (Regla clave de Warren Buffett).
        Evalúa cuánto ha aumentado el EPS por cada dólar que la empresa retuvo y reinvirtió.
        
        Formula: (EPS_Final - EPS_Inicial) / Suma(EPS_Retenido)
        """
        if len(eps_series) < 2 or len(eps_series) != len(dps_series):
            return {
                "retained_earnings_return": 0.0,
                "total_retained_eps": 0.0,
                "eps_growth": 0.0,
                "is_excellent": False
            }

        eps_initial = eps_series[0]
        eps_final = eps_series[-1]
        eps_growth = eps_final - eps_initial

        total_retained_eps = 0.0
        for eps, dps in zip(eps_series[:-1], dps_series[:-1]):
            retained = max(0.0, eps - dps)
            total_retained_eps += retained

        if total_retained_eps <= 0:
            return {
                "retained_earnings_return": 0.0,
                "total_retained_eps": total_retained_eps,
                "eps_growth": eps_growth,
                "is_excellent": False
            }

        retained_return = eps_growth / total_retained_eps
        # En Buffettology, un retorno > 12-15% sobre utilidades retenidas indica un Moat fuerte
        is_excellent = retained_return >= 0.12

        return {
            "retained_earnings_return": round(retained_return * 100, 2),
            "total_retained_eps": round(total_retained_eps, 2),
            "eps_growth": round(eps_growth, 2),
            "is_excellent": is_excellent
        }

    @staticmethod
    def calculate_profitability_metrics(
        net_income: float,
        stockholders_equity: float,
        total_assets: float,
        operating_income: float,
        total_revenue: float,
        gross_profit: Optional[float] = None,
        total_debt: Optional[float] = None,
        cash_and_equivalents: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calcula los ratios clave de rentabilidad y eficiencia.
        """
        roe = (net_income / stockholders_equity) * 100 if stockholders_equity and stockholders_equity > 0 else 0.0
        roa = (net_income / total_assets) * 100 if total_assets and total_assets > 0 else 0.0
        operating_margin = (operating_income / total_revenue) * 100 if total_revenue and total_revenue > 0 else 0.0
        net_margin = (net_income / total_revenue) * 100 if total_revenue and total_revenue > 0 else 0.0
        gross_margin = (gross_profit / total_revenue) * 100 if gross_profit and total_revenue and total_revenue > 0 else 0.0

        # ROIC = NOPAT / (Capital Invertido)
        # NOPAT aprox = Operating Income * (1 - 21% Tax)
        nopat = operating_income * 0.79
        invested_capital = (stockholders_equity + (total_debt or 0.0)) - (cash_and_equivalents or 0.0)
        roic = (nopat / invested_capital) * 100 if invested_capital and invested_capital > 0 else 0.0

        return {
            "roe": round(roe, 2),
            "roa": round(roa, 2),
            "roic": round(roic, 2),
            "gross_margin": round(gross_margin, 2),
            "operating_margin": round(operating_margin, 2),
            "net_margin": round(net_margin, 2)
        }

    @staticmethod
    def calculate_free_cash_flow(
        operating_cash_flow: float, 
        capital_expenditures: float
    ) -> Dict[str, float]:
        """
        Calcula el Flujo de Caja Libre (FCF) e intensidad de CapEx.
        """
        # CapEx suele registrarse como un valor negativo en los estados contables
        capex_abs = abs(capital_expenditures)
        fcf = operating_cash_flow - capex_abs
        capex_ratio = (capex_abs / operating_cash_flow) * 100 if operating_cash_flow > 0 else 0.0

        return {
            "free_cash_flow": round(fcf, 2),
            "capex": round(capex_abs, 2),
            "capex_to_ocf_ratio": round(capex_ratio, 2)
        }

    @staticmethod
    def project_buffett_valuation(
        current_eps: float,
        historical_cagr: float,
        average_pe: float,
        current_price: float,
        years: int = 10
    ) -> Dict[str, Any]:
        """
        Proyecta la rentabilidad esperada a 10 años utilizando el método de Buffettology.
        """
        if current_eps <= 0 or current_price <= 0:
            return {
                "projected_eps": 0.0,
                "projected_price": 0.0,
                "expected_annual_return": 0.0,
                "margin_of_safety_price": 0.0
            }

        capped_cagr = min(max(historical_cagr, 0.02), 0.20)  # Limitar entre 2% y 20% por prudencia
        projected_eps = current_eps * ((1 + capped_cagr) ** years)
        projected_price = projected_eps * average_pe

        if projected_price <= 0:
            expected_annual_return = 0.0
        else:
            expected_annual_return = ((projected_price / current_price) ** (1.0 / years)) - 1.0

        # Precio con 30% de Margen de Seguridad
        margin_of_safety_price = (projected_price / ((1.15) ** years)) * 0.70

        return {
            "current_eps": round(current_eps, 2),
            "projected_cagr": round(capped_cagr * 100, 2),
            "projected_eps_10y": round(projected_eps, 2),
            "projected_price_10y": round(projected_price, 2),
            "expected_annual_return": round(expected_annual_return * 100, 2),
            "margin_of_safety_price": round(margin_of_safety_price, 2)
        }