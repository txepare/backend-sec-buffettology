import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Importamos tu orquestador actual desde la carpeta src
from src.agents.orchestrator import OrchestratorAgent

# 1. Seguridad: Cargamos las variables de entorno desde el archivo .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# 2. Instanciamos la aplicación FastAPI (Nuestro "camarero")
app = FastAPI(title="SEC Buffettology API")

# Ruta de prueba para saber que el servidor está vivo
@app.get("/")
def ruta_raiz():
    return {"mensaje": "El servidor de Análisis Financiero está activo y esperando peticiones."}

# 3. Ruta principal: La ventanilla de atención para la app móvil
@app.get("/analizar/{ticker}")
def analizar_empresa(ticker: str):
    ticker_limpio = ticker.upper().strip()
    logger.info(f"Recibida solicitud de análisis para: {ticker_limpio}")
    
    try:
        # Iniciamos a tu orquestador para procesar la empresa
        orchestrator = OrchestratorAgent()
        ruta_pdf = orchestrator.run_analysis(ticker_limpio)
        
        # Verificamos que el agente realmente haya creado el archivo
        if os.path.exists(ruta_pdf):
            logger.info("PDF encontrado, enviando al cliente móvil.")
            # Enviamos el archivo de vuelta a la aplicación móvil
            return FileResponse(
                ruta_pdf, 
                media_type="application/pdf", 
                filename=f"Reporte_Valor_{ticker_limpio}.pdf"
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail="El análisis terminó, pero no se encontró el PDF en la ruta esperada."
            )
            
    except Exception as e:
        logger.error(f"Error durante la ejecución del análisis de {ticker_limpio}: {str(e)}", exc_info=True)
        # Si algo falla, avisamos a la app móvil
        raise HTTPException(status_code=500, detail=str(e))