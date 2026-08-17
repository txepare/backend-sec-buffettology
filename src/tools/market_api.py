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
        Obtiene la cotización actual y métricas de mercado para un ticker dado con sistema tolerante a fallos.
        """
        clean_ticker = ticker.upper().strip()
        cache_file = os.path.join(CACHE_DIR, f"{clean_ticker}_market.json")
        logger.info(f"[Market API] Consultando datos de mercado para: {clean_ticker}")

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

        # --- CANAL 1: Yahoo Chart v8 REST API (Inmune a rate limit 429) ---
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_ticker}?interval=1d&range=5d"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                chart_data = resp.json()
                meta = chart_data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                price_cand = meta.get("regularMarketPrice") or meta.get("previousClose") or meta.get("chartPreviousClose")
                if price_cand:
                    current_price = float(price_cand)
                company_name = meta.get("shortName") or meta.get("symbol") or clean_ticker
                currency = meta.get("currency", "USD")
                fifty_two_week_high = float(meta.get("fiftyTwoWeekHigh", 0.0) or 0.0)
                fifty_two_week_low = float(meta.get("fiftyTwoWeekLow", 0.0) or 0.0)
        except Exception as e:
            logger.debug(f"[Market API] Canal 1 (Chart v8) aviso: {e}")

        # --- CANAL 2: yfinance FastInfo / History ---
        try:
            stock = yf.Ticker(clean_ticker)
            if current_price == 0.0:
                fast_price = stock.fast_info.get("last_price") or stock.fast_info.get("previous_close")
                if fast_price:
                    current_price = float(fast_price)
            if market_cap == 0:
                market_cap = int(stock.fast_info.get("market_cap", 0) or 0)
            if shares_outstanding == 0:
                shares_outstanding = int(stock.fast_info.get("shares", 0) or 0)
            if fifty_two_week_high == 0.0:
                fifty_two_week_high = float(stock.fast_info.get("year_high", 0.0) or 0.0)
            if fifty_two_week_low == 0.0:
                fifty_two_week_low = float(stock.fast_info.get("year_low", 0.0) or 0.0)
        except Exception as e:
            logger.debug(f"[Market API] Canal 2 (FastInfo) aviso: {e}")

        # --- CANAL 3: yfinance stock.info (Para sector, industria y múltiplos) ---
        try:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            stock_session = yf.Ticker(clean_ticker, session=session)
            info = stock_session.info
            if info and isinstance(info, dict):
                if current_price == 0.0:
                    info_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
                    if info_price:
                        current_price = float(info_price)
                company_name = info.get("longName", company_name)
                sector = info.get("sector", sector)
                industry = info.get("industry", industry)
                pe_ratio = float(info.get("trailingPE") or info.get("forwardPE") or 0.0)
                forward_pe = float(info.get("forwardPE", 0.0) or 0.0)
                peg_ratio = float(info.get("pegRatio", 0.0) or 0.0)
                price_to_book = float(info.get("priceToBook", 0.0) or 0.0)
                div_yield = info.get("dividendYield")
                dividend_yield = float(div_yield * 100) if div_yield else 0.0
                beta = float(info.get("beta", 1.0) or 1.0)
                if market_cap == 0:
                    market_cap = int(info.get("marketCap", 0) or 0)
                if shares_outstanding == 0:
                    shares_outstanding = int(info.get("sharesOutstanding", 0) or 0)
        except Exception as e:
            logger.debug(f"[Market API] Canal 3 (Stock info) no crítico: {e}")

        # --- CANAL 4: Fallback de historial ---
        if current_price == 0.0:
            try:
                hist = stock.history(period="5d")
                if not hist.empty and "Close" in hist:
                    current_price = float(hist["Close"].iloc[-1])
            except Exception as e:
                logger.debug(f"[Market API] Canal 4 (Historial) aviso: {e}")

        # Si obtuvimos precio con éxito, guardar en caché de disco
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
                "industry": industry
            }
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(market_quote, f, indent=2)
            except Exception as e:
                logger.debug(f"[Market API] No se pudo escribir caché: {e}")
                
            logger.info(f"[Market API] Datos obtenidos exitosamente para {clean_ticker}: Precio ${current_price:.2f}, Sector: {sector}")
            return market_quote

        # --- CANAL 5: Cargar de caché local si todo lo anterior falló ---
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    logger.info(f"[Market API] Usando datos de cotización en caché para {clean_ticker}: Precio ${cached.get('current_price', 0.0)}")
                    return cached
            except Exception as e:
                logger.error(f"[Market API] Error leyendo caché para {clean_ticker}: {e}")

        # Fallback final si la red está completamente caída y no hay caché
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
            "industry": "Desconocida"
        }