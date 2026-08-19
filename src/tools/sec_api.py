import os
import re
import json
import logging
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, List
from config.settings import SEC_USER_AGENT, CACHE_DIR

logger = logging.getLogger(__name__)


class SecEdgarAPI:
    BASE_URL = "https://data.sec.gov"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
    HEADERS = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate"
    }

    @staticmethod
    def _get_tickers_mapping() -> Dict[str, str]:
        """Obtiene y cachea el mapeo de Tickers a CIKs."""
        cache_path = os.path.join(CACHE_DIR, "company_tickers.json")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=SecEdgarAPI.HEADERS, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        # Transformar para búsqueda rápida: {"AAPL": "0000320193", ...}
        mapping = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for k, v in data.items()}
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)
            
        return mapping

    @staticmethod
    def get_cik_from_ticker(ticker: str) -> Optional[str]:
        """Convierte un ticker (ej. AAPL) en un CIK rellenado con ceros (10 dígitos)."""
        mapping = SecEdgarAPI._get_tickers_mapping()
        return mapping.get(ticker.upper().strip())

    @staticmethod
    def fetch_company_facts(ticker: str) -> Dict[str, Any]:
        """
        Extrae todos los 'facts' financieros (XBRL) de una empresa.
        Retorna los datos y los guarda en caché local.
        """
        clean_ticker = ticker.upper().strip()
        cik = SecEdgarAPI.get_cik_from_ticker(clean_ticker)
        if not cik:
            raise ValueError(f"Ticker {clean_ticker} no encontrado en la base de datos de la SEC.")

        cache_path = os.path.join(CACHE_DIR, f"{clean_ticker}_facts.json")
        if os.path.exists(cache_path):
            logger.info(f"[SEC API] Cargando facts XBRL de {clean_ticker} desde caché local...")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        url = f"{SecEdgarAPI.BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
        logger.info(f"[SEC API] Descargando facts XBRL de la SEC para {clean_ticker}...")
        
        response = requests.get(url, headers=SecEdgarAPI.HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        return data

    @staticmethod
    def fetch_submissions(ticker: str) -> Dict[str, Any]:
        """
        Extrae el listado de presentaciones (submissions) oficiales de la SEC para un CIK dado.
        """
        clean_ticker = ticker.upper().strip()
        cik = SecEdgarAPI.get_cik_from_ticker(clean_ticker)
        if not cik:
            raise ValueError(f"Ticker {clean_ticker} no encontrado en la base de datos de la SEC.")

        cache_path = os.path.join(CACHE_DIR, f"{clean_ticker}_submissions.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        url = f"{SecEdgarAPI.BASE_URL}/submissions/CIK{cik}.json"
        logger.info(f"[SEC API] Consultando presentaciones oficiales (Submissions) para {clean_ticker}...")
        response = requests.get(url, headers=SecEdgarAPI.HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return data

    @staticmethod
    def _extract_10k_sections_from_html(html_content: bytes) -> Dict[str, str]:
        """
        Parsea el archivo HTML del informe 10-K y extrae las secciones narrativas clave:
        - Item 1: Business (Actividad, productos, servicios, divisiones, clientes, distribución)
        - Item 2: Properties (Sede central, instalaciones, fábricas)
        - Item 7: MD&A (Discusión de la dirección, vientos de cola, dinámicas de mercado, perspectivas de crecimiento)
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Eliminar scripts, estilos y encabezados que añaden ruido
        for tag in soup(['script', 'style', 'head', 'noscript', 'svg']):
            tag.decompose()
        
        text = soup.get_text(separator=' \n ')
        
        # Normalización de caracteres y saltos de línea
        text = re.sub(r'[\xa0\u200b\t\r]+', ' ', text)
        # Unir palabras cortadas por saltos de línea accidentales (ej. B\nUSINESS -> BUSINESS)
        text = re.sub(r'([A-Z])\s*\n\s*([A-Z]{2,})', r'\1\2', text)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n +', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        def find_positions(pattern: str) -> List[int]:
            return [m.start() for m in re.finditer(pattern, text, re.IGNORECASE)]

        item1_starts = find_positions(r'(?:PART\s+I[\s\n]+)?(?:ITEM|Item)\s+1[\.\:\s\-]+(?:BUSINESS|Business|Description\s+of\s+Business|General)')
        item1a_starts = find_positions(r'(?:ITEM|Item)\s+1A[\.\:\s\-]+(?:RISK\s+FACTORS|Risk\s+Factors|Risk)')
        item2_starts = find_positions(r'(?:ITEM|Item)\s+2[\.\:\s\-]+(?:PROPERTIES|Properties)')
        item7_starts = find_positions(r'(?:ITEM|Item)\s+7[\.\:\s\-]+(?:MANAGEMENT|Management)')
        item7a_starts = find_positions(r'(?:ITEM|Item)\s+7A[\.\:\s\-]+(?:QUANTITATIVE|Quantitative)')
        item8_starts = find_positions(r'(?:ITEM|Item)\s+8[\.\:\s\-]+(?:FINANCIAL\s+STATEMENTS|Financial\s+Statements)')

        def extract_span(start_candidates: List[int], end_candidates: List[int], min_len: int = 1200, max_len: int = 50000) -> str:
            # Buscar la coincidencia real del cuerpo (después de la tabla de contenidos)
            for s in reversed(start_candidates):
                for e in end_candidates:
                    if e > s + min_len:
                        span_len = e - s
                        if span_len <= max_len:
                            return text[s:e].strip()
                        else:
                            return text[s:s + max_len].strip()
            
            # Fallback en caso de que los delimitadores de fin no coincidan exactamente
            if start_candidates:
                last_s = start_candidates[-1]
                return text[last_s:last_s + max_len].strip()
            return ""

        item1_business = extract_span(item1_starts, item1a_starts + item2_starts, min_len=1000, max_len=45000)
        item7_mda = extract_span(item7_starts, item7a_starts + item8_starts, min_len=1000, max_len=45000)
        item2_properties = extract_span(item2_starts, item7_starts, min_len=200, max_len=10000)

        # Si Item 1 o Item 7 no pudieron aislarse con los delimitadores específicos, tomar ventana amplia
        if not item1_business and item1_starts:
            item1_business = text[item1_starts[-1]:item1_starts[-1] + 30000].strip()
        if not item7_mda and item7_starts:
            item7_mda = text[item7_starts[-1]:item7_starts[-1] + 30000].strip()

        return {
            "item1_business": item1_business,
            "item2_properties": item2_properties,
            "item7_mda": item7_mda
        }

    @staticmethod
    def fetch_company_10k_narrative(ticker: str) -> Dict[str, Any]:
        """
        Descarga y procesa el texto íntegro del informe 10-K más reciente presentado ante la SEC.
        Almacena en caché el texto procesado de las secciones narrativas clave.
        """
        clean_ticker = ticker.upper().strip()
        cache_path = os.path.join(CACHE_DIR, f"{clean_ticker}_10k_narrative.json")

        if os.path.exists(cache_path):
            logger.info(f"[SEC API] Cargando texto del informe 10-K para {clean_ticker} desde caché local...")
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if cached_data.get("item1_business") or cached_data.get("item7_mda"):
                        return cached_data
            except Exception as e:
                logger.debug(f"[SEC API] Aviso leyendo caché 10-K: {e}")

        cik = SecEdgarAPI.get_cik_from_ticker(clean_ticker)
        if not cik:
            logger.warning(f"[SEC API] No se encontró CIK para {clean_ticker}.")
            return {"ticker": clean_ticker, "item1_business": "", "item2_properties": "", "item7_mda": ""}

        try:
            submissions = SecEdgarAPI.fetch_submissions(clean_ticker)
            recent = submissions.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])

            # Buscar el informe 10-K más reciente
            idx_10k = None
            for idx, form in enumerate(forms):
                if form == "10-K":
                    idx_10k = idx
                    break

            if idx_10k is None:
                logger.warning(f"[SEC API] No se encontró ningún informe 10-K reciente en las presentaciones de {clean_ticker}.")
                return {"ticker": clean_ticker, "item1_business": "", "item2_properties": "", "item7_mda": ""}

            accession_number = recent["accessionNumber"][idx_10k]
            accession_nodash = accession_number.replace("-", "")
            primary_doc = recent["primaryDocument"][idx_10k]
            filing_date = recent["filingDate"][idx_10k]
            report_date = recent.get("reportDate", [""])[idx_10k] if idx_10k < len(recent.get("reportDate", [])) else ""
            cik_int = str(int(cik))

            doc_url = f"{SecEdgarAPI.ARCHIVES_URL}/{cik_int}/{accession_nodash}/{primary_doc}"
            logger.info(f"[SEC API] Descargando informe oficial 10-K de {clean_ticker} ({filing_date}) desde SEC EDGAR: {doc_url}...")

            doc_resp = requests.get(doc_url, headers=SecEdgarAPI.HEADERS, timeout=30)
            doc_resp.raise_for_status()

            # Parsear secciones del informe HTML
            logger.info(f"[SEC API] Extrayendo secciones (Item 1 Business, Item 2 Properties, Item 7 MD&A) de {clean_ticker}...")
            sections = SecEdgarAPI._extract_10k_sections_from_html(doc_resp.content)

            result = {
                "ticker": clean_ticker,
                "cik": cik,
                "accession_number": accession_number,
                "primary_document": primary_doc,
                "filing_date": filing_date,
                "report_date": report_date,
                "document_url": doc_url,
                "item1_business": sections.get("item1_business", ""),
                "item2_properties": sections.get("item2_properties", ""),
                "item7_mda": sections.get("item7_mda", "")
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

            logger.info(
                f"[SEC API] Informe 10-K extraído con éxito para {clean_ticker} "
                f"(Item 1: {len(result['item1_business'])} caracteres, Item 7: {len(result['item7_mda'])} caracteres)"
            )
            return result

        except Exception as e:
            logger.error(f"[SEC API] Error descargando o extrayendo informe 10-K para {clean_ticker}: {e}")
            return {
                "ticker": clean_ticker,
                "cik": cik,
                "item1_business": "",
                "item2_properties": "",
                "item7_mda": "",
                "error": str(e)
            }

