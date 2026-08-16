import os
import json
import requests
from typing import Dict, Any, Optional
from config.settings import SEC_USER_AGENT, CACHE_DIR

class SecEdgarAPI:
    BASE_URL = "https://data.sec.gov"
    HEADERS = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate"
    }

    @staticmethod
    def _get_tickers_mapping() -> Dict[str, str]:
        """Obtiene y cachea el mapeo de Tickers a CIKs."""
        cache_path = os.path.join(CACHE_DIR, "company_tickers.json")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
                
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=SecEdgarAPI.HEADERS)
        response.raise_for_status()
        
        data = response.json()
        # Transformar para búsqueda rápida: {"AAPL": "0000320193", ...}
        mapping = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for k, v in data.items()}
        
        with open(cache_path, 'w') as f:
            json.dump(mapping, f)
            
        return mapping

    @staticmethod
    def get_cik_from_ticker(ticker: str) -> Optional[str]:
        """Convierte un ticker (ej. AAPL) en un CIK rellenado con ceros."""
        mapping = SecEdgarAPI._get_tickers_mapping()
        return mapping.get(ticker.upper())

    @staticmethod
    def fetch_company_facts(ticker: str) -> Dict[str, Any]:
        """
        Extrae todos los 'facts' financieros (XBRL) de una empresa.
        Retorna los datos y los guarda en caché local.
        """
        cik = SecEdgarAPI.get_cik_from_ticker(ticker)
        if not cik:
            raise ValueError(f"Ticker {ticker} no encontrado en la base de datos de la SEC.")

        cache_path = os.path.join(CACHE_DIR, f"{ticker}_facts.json")
        if os.path.exists(cache_path):
            print(f"[SEC API] Cargando datos de {ticker} desde caché local...")
            with open(cache_path, 'r') as f:
                return json.load(f)

        url = f"{SecEdgarAPI.BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
        print(f"[SEC API] Descargando datos de la SEC para {ticker}...")
        
        response = requests.get(url, headers=SecEdgarAPI.HEADERS)
        response.raise_for_status()
        data = response.json()

        with open(cache_path, 'w') as f:
            json.dump(data, f)

        return data
