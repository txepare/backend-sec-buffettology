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

    _directory_loaded: bool = False
    _ticker_map: Dict[str, Dict[str, str]] = {}
    _norm_title_map: Dict[str, Dict[str, str]] = {}
    _companies_list: List[Dict[str, str]] = []
    _resolved_cache: Dict[str, Dict[str, str]] = {}

    @staticmethod
    def _normalize_company_name(name: str) -> str:
        """Normaliza el nombre de una empresa para comparaciones insensibles a puntuación y sufijos legales."""
        name = name.lower()
        name = re.sub(r'[\.,/\\#\$\%\^\&\*\;\:\{\}\=\_\`\~\(\)\-\[\]\"\'\+]', ' ', name)
        suffixes = {
            'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 
            'ltd', 'limited', 'plc', 'llc', 'lp', 'group', 'holdings', 
            'holding', 'class a', 'class b', 'class c', 'com', 'new', 'de', 'the', 'sa', 'nv'
        }
        tokens = [w for w in name.split() if w not in suffixes]
        return ' '.join(tokens).strip()

    @classmethod
    def _load_directory(cls, force_reload: bool = False) -> None:
        """Carga y prepara el directorio completo de empresas (Tickers, CIK y Nombres) en memoria."""
        if cls._directory_loaded and not force_reload:
            return

        cache_path_full = os.path.join(CACHE_DIR, "company_tickers_full.json")
        cache_path_simple = os.path.join(CACHE_DIR, "company_tickers.json")

        raw_data = None
        if not force_reload and os.path.exists(cache_path_full):
            try:
                with open(cache_path_full, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            except Exception as e:
                logger.warning(f"[SEC API] Error leyendo caché completo: {e}")

        if not raw_data:
            try:
                url = "https://www.sec.gov/files/company_tickers.json"
                logger.info("[SEC API] Descargando catálogo oficial de empresas desde la SEC...")
                response = requests.get(url, headers=cls.HEADERS, timeout=12)
                response.raise_for_status()
                raw_data = response.json()

                with open(cache_path_full, 'w', encoding='utf-8') as f:
                    json.dump(raw_data, f, indent=2)
            except Exception as e:
                logger.error(f"[SEC API] No se pudo descargar el directorio SEC: {e}")
                # Fallback al archivo simple si existe
                if os.path.exists(cache_path_simple):
                    with open(cache_path_simple, 'r', encoding='utf-8') as f:
                        simple_map = json.load(f)
                        raw_data = {
                            str(i): {"ticker": k, "cik_str": v, "title": k}
                            for i, (k, v) in enumerate(simple_map.items())
                        }

        if not raw_data:
            raw_data = {}

        cls._ticker_map.clear()
        cls._norm_title_map.clear()
        cls._companies_list.clear()

        simple_export = {}
        for k, v in raw_data.items():
            ticker = str(v.get("ticker", "")).upper().strip()
            if not ticker:
                continue
            title = str(v.get("title", "")).strip()
            cik = str(v.get("cik_str", "")).zfill(10)
            norm_title = cls._normalize_company_name(title)

            item = {
                "ticker": ticker,
                "title": title,
                "cik": cik,
                "norm_title": norm_title
            }
            cls._companies_list.append(item)
            cls._ticker_map[ticker] = item
            # Soportar variantes de tickers con puntos/guiones (ej. BRK.B / BRK-B)
            cls._ticker_map[ticker.replace('.', '-')] = item
            cls._ticker_map[ticker.replace('-', '.')] = item

            if norm_title and norm_title not in cls._norm_title_map:
                cls._norm_title_map[norm_title] = item

            simple_export[ticker] = cik

        # Guardar mapeo simple compatible
        try:
            with open(cache_path_simple, 'w', encoding='utf-8') as f:
                json.dump(simple_export, f, indent=2)
        except Exception:
            pass

        cls._directory_loaded = True
        logger.info(f"[SEC API] Directorio SEC inicializado con {len(cls._companies_list)} empresas.")

    @classmethod
    def _get_tickers_mapping(cls) -> Dict[str, str]:
        """Obtiene y cachea el mapeo de Tickers a CIKs (Compatibilidad)."""
        cls._load_directory()
        return {item["ticker"]: item["cik"] for item in cls._companies_list}

    @classmethod
    def resolve_company(cls, query: str) -> Optional[Dict[str, str]]:
        """
        Resuelve de forma ultra-rápida un término de búsqueda (Ticker o Nombre de Empresa)
        a su ficha canónica en la SEC {ticker, title, cik}.
        """
        if not query or not str(query).strip():
            return None

        cls._load_directory()
        clean_q = str(query).strip()
        upper_q = clean_q.upper()
        cache_key = upper_q

        if cache_key in cls._resolved_cache:
            return cls._resolved_cache[cache_key]

        # 1. Coincidencia exacta de Ticker (O(1))
        if upper_q in cls._ticker_map:
            res = cls._ticker_map[upper_q]
            cls._resolved_cache[cache_key] = res
            return res

        std_q = upper_q.replace('.', '-')
        if std_q in cls._ticker_map:
            res = cls._ticker_map[std_q]
            cls._resolved_cache[cache_key] = res
            return res

        # 2. Coincidencia exacta de Nombre Normalizado (O(1))
        norm_q = cls._normalize_company_name(clean_q)
        if norm_q and norm_q in cls._norm_title_map:
            res = cls._norm_title_map[norm_q]
            cls._resolved_cache[cache_key] = res
            return res

        # 3. Coincidencia de Prefijo de Nombre (ej. "Berkshire" -> Berkshire Hathaway, "Costco" -> Costco Wholesale)
        if norm_q and len(norm_q) >= 3:
            for c in cls._companies_list:
                if c["norm_title"].startswith(norm_q) and len(c["norm_title"]) <= len(norm_q) + 20:
                    cls._resolved_cache[cache_key] = c
                    return c

        # 4. Coincidencia por Contención de Subcadena / Palabras clave
        if norm_q and len(norm_q) >= 4:
            for c in cls._companies_list:
                if norm_q in c["norm_title"]:
                    cls._resolved_cache[cache_key] = c
                    return c

        # 5. Fallback con Yahoo Finance Search (para nombres coloquiales como Google -> GOOGL, Facebook -> META)
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={clean_q}&quotesCount=5&newsCount=0"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
            if resp.status_code == 200:
                quotes = resp.json().get("quotes", [])
                for q in quotes:
                    sym = q.get("symbol", "").upper().replace('.', '-')
                    if sym in cls._ticker_map:
                        res = cls._ticker_map[sym]
                        cls._resolved_cache[cache_key] = res
                        logger.info(f"[SEC API] '{clean_q}' resuelto vía Yahoo Finance a {res['ticker']} ({res['title']})")
                        return res
        except Exception as e:
            logger.debug(f"[SEC API] Aviso en búsqueda Yahoo para '{clean_q}': {e}")

        return None

    @classmethod
    def resolve_ticker(cls, query: str) -> str:
        """
        Retorna el Ticker oficial para una búsqueda (ticker o nombre).
        Si no se encuentra, retorna el query limpio en mayúsculas como fallback.
        """
        company = cls.resolve_company(query)
        if company:
            return company["ticker"]
        return query.upper().strip()

    @classmethod
    def search_companies(cls, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Busca empresas coincidentes para sugerencias rápidas o autocompletado.
        """
        if not query or not str(query).strip():
            return []

        cls._load_directory()
        clean_q = str(query).strip().upper()
        norm_q = cls._normalize_company_name(clean_q)

        results = []
        seen_tickers = set()

        # Coincidencias de Ticker prioritarias
        for c in cls._companies_list:
            if c["ticker"].startswith(clean_q):
                results.append({"ticker": c["ticker"], "title": c["title"], "cik": c["cik"]})
                seen_tickers.add(c["ticker"])
                if len(results) >= limit:
                    return results

        # Coincidencias de Nombre
        if norm_q:
            for c in cls._companies_list:
                if c["ticker"] in seen_tickers:
                    continue
                if c["norm_title"].startswith(norm_q) or norm_q in c["norm_title"]:
                    results.append({"ticker": c["ticker"], "title": c["title"], "cik": c["cik"]})
                    seen_tickers.add(c["ticker"])
                    if len(results) >= limit:
                        return results

        return results

    @classmethod
    def get_cik_from_ticker(cls, ticker_or_name: str) -> Optional[str]:
        """Convierte un ticker o nombre de empresa en un CIK rellenado con ceros (10 dígitos)."""
        company = cls.resolve_company(ticker_or_name)
        if company:
            return company["cik"]
        return None

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

