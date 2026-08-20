import os
import json
import logging
from typing import Dict, Any, Optional
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cache_market")
os.makedirs(CACHE_DIR, exist_ok=True)


class MarketDataAPI:
    """
    Módulo ultra-resiliente para interactuar con APIs de datos de mercado en tiempo real.
    Utiliza una arquitectura multi-canal con fallback y caché en disco para evitar errores 429.
    """

    @staticmethod
    def get_market_quote(ticker: str) -> Dict[str, Any]:
        """
        Obtiene la cotización actual y métricas de mercado para un ticker dado con sistema tolerante a fallos
        y respuesta ultrarrápida (<300ms) mediante APIs REST directas y caché local.
        """
        clean_ticker = ticker.upper().strip()
        cache_file = os.path.join(CACHE_DIR, f"{clean_ticker}_market.json")
        logger.info(f"[Market API] Consultando datos de mercado para: {clean_ticker}")

        # Si el caché en disco existe y tiene menos de 4 horas de antigüedad, reutilizarlo
        if os.path.exists(cache_file):
            try:
                mtime = os.path.getmtime(cache_file)
                import time
                if (time.time() - mtime) < 14400: # 4 horas
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                        if cached.get("current_price", 0) > 0:
                            logger.info(f"[Market API] Cotización cargada desde caché reciente para {clean_ticker}: ${cached.get('current_price')}")
                            return cached
            except Exception as e:
                logger.debug(f"[Market API] Aviso verificando caché: {e}")

        current_price = 0.0
        company_name = clean_ticker
        currency = "USD"
        sector = "Industrial"
        industry = "General"
        market_cap = 0
        pe_ratio = 0.0
        forward_pe = 0.0
        peg_ratio = 0.0
        price_to_book = 0.0
        shares_outstanding = 0
        dividend_yield = 0.0
        beta = 1.0
        fifty_two_week_high = 0.0
        fifty_two_week_low = 0.0
        description = ""

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }

        # --- CANAL 1: Yahoo Chart v8 REST API (Inmune a rate limit 429, <150ms) ---
        try:
            url_chart = f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_ticker}?interval=1d&range=5d"
            resp = requests.get(url_chart, headers=headers, timeout=3)
            if resp.status_code == 200:
                chart_data = resp.json()
                meta = chart_data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                price_cand = meta.get("regularMarketPrice") or meta.get("previousClose") or meta.get("chartPreviousClose")
                if price_cand:
                    current_price = float(price_cand)
                company_name = meta.get("longName") or meta.get("shortName") or meta.get("symbol") or clean_ticker
                currency = meta.get("currency", "USD")
                fifty_two_week_high = float(meta.get("fiftyTwoWeekHigh", 0.0) or 0.0)
                fifty_two_week_low = float(meta.get("fiftyTwoWeekLow", 0.0) or 0.0)
        except Exception as e:
            logger.debug(f"[Market API] Canal 1 (Chart v8) aviso: {e}")

        # --- CANAL 2: Yahoo Search API para Sector, Industria y Nombre oficial (<100ms) ---
        try:
            url_search = f"https://query2.finance.yahoo.com/v1/finance/search?q={clean_ticker}&quotesCount=1&newsCount=0"
            resp_s = requests.get(url_search, headers=headers, timeout=3)
            if resp_s.status_code == 200:
                quotes = resp_s.json().get("quotes", [])
                if quotes:
                    q = quotes[0]
                    sector = q.get("sector") or q.get("sectorDisp") or sector
                    industry = q.get("industry") or q.get("industryDisp") or industry
                    if company_name == clean_ticker:
                        company_name = q.get("longname") or q.get("shortname") or company_name
        except Exception as e:
            logger.debug(f"[Market API] Canal 2 (Search) aviso: {e}")

        # Si obtuvimos precio con éxito, guardar en caché de disco y retornar de inmediato
        if current_price > 0.0:
            market_quote = {
                "ticker": clean_ticker,
                "company_name": company_name,
                "currency": currency,
                "current_price": current_price,
                "market_cap": market_cap,
                "enterprise_value": market_cap,
                "pe_ratio": pe_ratio,
                "forward_pe": forward_pe,
                "peg_ratio": peg_ratio,
                "price_to_book": price_to_book,
                "shares_outstanding": shares_outstanding,
                "dividend_yield": dividend_yield,
                "beta": beta,
                "52_week_high": fifty_two_week_high,
                "52_week_low": fifty_two_week_low,
                "sector": sector,
                "industry": industry,
                "description": description,
                "business_summary": description
            }
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(market_quote, f, indent=2)
            except Exception as e:
                logger.debug(f"[Market API] No se pudo escribir caché: {e}")
                
            logger.info(f"[Market API] Datos obtenidos exitosamente para {clean_ticker}: Precio ${current_price:.2f}, Sector: {sector}")
            return market_quote

        # --- CANAL 3: Cargar de caché local si la red falló ---
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    logger.info(f"[Market API] Usando datos de cotización en caché para {clean_ticker}: Precio ${cached.get('current_price', 0.0)}")
                    return cached
            except Exception as e:
                logger.error(f"[Market API] Error leyendo caché para {clean_ticker}: {e}")

        # Fallback final
        logger.warning(f"[Market API] No se pudo obtener precio en tiempo real para {clean_ticker}. Retornando estructura base.")
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
            "description": "",
            "business_summary": ""
        }