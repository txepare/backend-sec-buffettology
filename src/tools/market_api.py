import logging
from typing import Dict, Any, Optional
import yfinance as yf
import requests

logger = logging.getLogger(__name__)


class MarketDataAPI:
    """
    Módulo para interactuar con APIs de datos de mercado en tiempo real.
    Utiliza yfinance para extraer cotizaciones, métricas de valoración y datos clave.
    """

    @staticmethod
    def get_market_quote(ticker: str) -> Dict[str, Any]:
        """
        Obtiene la cotización actual y métricas de mercado para un ticker dado.
        
        Args:
            ticker: Símbolo bursátil de la empresa (ej. AAPL, MSFT, JPM)
            
        Returns:
            Dict con datos del precio actual, múltiplos, capitalización, etc.
        """
        clean_ticker = ticker.upper().strip()
        logger.info(f"[Market API] Consultando datos de mercado para: {clean_ticker}")

        try:
            # Crear una sesión para camuflar la petición y evitar el Error 429 de Yahoo Finance
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            
            stock = yf.Ticker(clean_ticker, session=session)
            info = stock.info

            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            
            market_quote = {
                "ticker": clean_ticker,
                "company_name": info.get("longName", clean_ticker),
                "currency": info.get("currency", "USD"),
                "current_price": float(current_price) if current_price else 0.0,
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE") or info.get("forwardPE") or 0.0,
                "forward_pe": info.get("forwardPE", 0.0),
                "peg_ratio": info.get("pegRatio", 0.0),
                "price_to_book": info.get("priceToBook", 0.0),
                "shares_outstanding": info.get("sharesOutstanding", 0),
                "dividend_yield": info.get("dividendYield", 0.0) * 100 if info.get("dividendYield") else 0.0,
                "beta": info.get("beta", 1.0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0.0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0.0),
                "sector": info.get("sector", "Desconocido"),
                "industry": info.get("industry", "Desconocida")
            }

            logger.info(f"[Market API] Datos obtenidos exitosamente para {clean_ticker}: Precio ${market_quote['current_price']}")
            return market_quote

        except Exception as e:
            logger.error(f"[Market API] Error al obtener datos de mercado para {clean_ticker}: {str(e)}")
            return {
                "ticker": clean_ticker,
                "company_name": clean_ticker,
                "currency": "USD",
                "current_price": 0.0,
                "market_cap": 0,
                "enterprise_value": 0,
                "pe_ratio": 0.0,
                "forward_pe": 0.0,
                "peg_ratio": 0.0,
                "price_to_book": 0.0,
                "shares_outstanding": 0,
                "dividend_yield": 0.0,
                "beta": 1.0,
                "52_week_high": 0.0,
                "52_week_low": 0.0,
                "sector": "Desconocido",
                "industry": "Desconocida",
                "error": str(e)
            }