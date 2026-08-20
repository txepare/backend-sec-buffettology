import logging
import os
import threading
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Importamos tu orquestador y la API de la SEC
from src.agents.orchestrator import OrchestratorAgent
from src.tools.sec_api import SecEdgarAPI

# 1. Seguridad: Cargamos las variables de entorno desde el archivo .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# 2. Instanciamos la aplicación FastAPI
app = FastAPI(
    title="SEC Buffettology API",
    description="API de Análisis Fundamental y Valoración Buffettology automatizada con IA",
    version="2.0.0"
)

# Permitir CORS para cualquier cliente móvil o web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-carga del directorio de empresas en segundo plano al iniciar
@app.on_event("startup")
def precalentar_directorio_sec():
    def _warmup():
        try:
            logger.info("[Startup] Precargando catálogo de empresas de la SEC en memoria...")
            SecEdgarAPI._load_directory()
            logger.info("[Startup] Catálogo de empresas precargado y listo.")
        except Exception as e:
            logger.warning(f"[Startup] Aviso precargando catálogo: {e}")

    threading.Thread(target=_warmup, daemon=True).start()

# Ruta de prueba y salud
@app.get("/")
def ruta_raiz():
    return {
        "status": "online",
        "mensaje": "El servidor de Análisis Financiero Buffettology está activo y listo.",
        "version": "2.0.0"
    }

# Endpoint de búsqueda y autocompletado en tiempo real
@app.get("/buscar")
def buscar_empresas(q: str = Query(..., min_length=1, description="Ticker o nombre de la empresa"), limit: int = 5):
    """
    Busca al instante empresas en el catálogo oficial de la SEC por Ticker o Nombre.
    """
    resultados = SecEdgarAPI.search_companies(q, limit=limit)
    return {
        "query": q,
        "total": len(resultados),
        "resultados": resultados
    }

# 3. Ruta principal: Acepta Ticker O Nombre de Empresa (ej. AAPL, Apple, Coca-Cola, MSFT, Google)
@app.get("/analizar/{empresa_o_ticker}")
def analizar_empresa(empresa_o_ticker: str):
    query_limpio = empresa_o_ticker.strip()
    logger.info(f"Recibida solicitud de análisis para: '{query_limpio}'")
    
    # 1. Identificar y resolver Ticker oficial
    company_info = SecEdgarAPI.resolve_company(query_limpio)
    if company_info:
        ticker_oficial = company_info["ticker"]
        company_title = company_info["title"]
        logger.info(f"'{query_limpio}' resuelto con éxito -> Ticker: {ticker_oficial} | {company_title}")
    else:
        ticker_oficial = query_limpio.upper()
        company_title = ticker_oficial
        logger.info(f"No se encontró coincidencia directa para '{query_limpio}', usando '{ticker_oficial}' como Ticker directo.")

    try:
        # Iniciamos el orquestador optimizado multi-hilo
        orchestrator = OrchestratorAgent()
        ruta_pdf = orchestrator.run_analysis(ticker_oficial)
        
        # Verificamos que el agente realmente haya creado el archivo
        if os.path.exists(ruta_pdf):
            logger.info(f"PDF generado con éxito ({os.path.getsize(ruta_pdf)} bytes). Enviando al cliente...")
            return FileResponse(
                ruta_pdf, 
                media_type="application/pdf", 
                filename=f"Reporte_Valor_{ticker_oficial}.pdf",
                headers={
                    "X-Resolved-Ticker": ticker_oficial,
                    "X-Company-Name": company_title,
                    "Access-Control-Expose-Headers": "X-Resolved-Ticker, X-Company-Name, Content-Disposition"
                }
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail="El análisis concluyó pero no se encontró el PDF en la ruta esperada."
            )
            
    except Exception as e:
        logger.error(f"Error durante el análisis de {ticker_oficial}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))