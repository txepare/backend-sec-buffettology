import logging
import os
import json
import pandas as pd
from typing import Dict, Any

from src.agents.market_agent import MarketDataAgent
from src.agents.sec_agent import SecExtractorAgent
from src.agents.normalizer_agent import GAAPNormalizerAgent
from src.agents.sector_agent import SectorConfiguratorAgent
from src.agents.validator_agent import ValidationAgent
from src.agents.pdf_agent import PDFGeneratorAgent
from config.settings import OUTPUT_DIR

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self):
        self.market_agent = MarketDataAgent()
        self.sec_agent = SecExtractorAgent()
        self.normalizer_agent = GAAPNormalizerAgent()
        self.sector_agent = SectorConfiguratorAgent()
        self.validator_agent = ValidationAgent()
        self.pdf_agent = PDFGeneratorAgent()

    def run_analysis(self, ticker: str) -> str:
        logger.info(f"=== [Orchestrator] Iniciando pipeline de análisis para {ticker} ===")
        
        # Paso 1: Datos de Mercado
        market_data = self.market_agent.get_data(ticker)
        
        # Bucle de Extracción y Validación (Máximo 3 intentos)
        max_intentos = 3
        intento_actual = 1
        datos_validados = False
        normalized = {}
        
        while intento_actual <= max_intentos and not datos_validados:
            logger.info(f"--- [Orchestrator] Ciclo de Extracción y Validación (Intento {intento_actual}/{max_intentos}) ---")
            
            # Paso 2: Extracción SEC
            raw_sec = self.sec_agent.extract(ticker)
            
            # Paso 3: Normalización y Síntesis Matemática
            normalized = self.normalizer_agent.normalize(raw_sec)
            
            # Paso 4: Validación por el Agente Auditor
            respuesta_auditoria = self.validator_agent.evaluate(normalized)
            
            try:
                # Asegurarnos de que procesamos el JSON estrictamente
                if isinstance(respuesta_auditoria, str):
                    limpio = respuesta_auditoria.replace("```json", "").replace("```", "").strip()
                    auditoria_json = json.loads(limpio)
                else:
                    auditoria_json = respuesta_auditoria
                    
                estado = auditoria_json.get("estado_auditoria", "RECHAZADO")
                motivos = auditoria_json.get("motivos", [])
                
                if estado == "APROBADO":
                    logger.info(f"[Orchestrator] ¡Validación Exitosa del Auditor! {motivos}")
                    datos_validados = True
                else:
                    logger.warning(f"[Orchestrator] Auditoría fallida: {motivos}")
                    logger.warning(f"[Orchestrator] Partidas a corregir identificadas: {auditoria_json.get('partidas_a_corregir', [])}")
                    intento_actual += 1
                    
            except Exception as e:
                logger.error(f"[Orchestrator] Error procesando el veredicto del auditor: {e}")
                intento_actual += 1

        if not datos_validados:
            logger.warning(f"=== [Orchestrator] ADVERTENCIA: Se agotaron los {max_intentos} intentos. Se continuará con los últimos datos generados. ===")

        # Paso 5: Exportar datos extraídos a CSV
        years = normalized.get("years", [])
        try:
            aligned_series = normalized.get("aligned_series", {})
            if years and aligned_series:
                df_export = pd.DataFrame(aligned_series, index=years)
                df_export.index.name = "Periodo Fiscal"
                csv_path = os.path.join(OUTPUT_DIR, f"{ticker}_extracted_historical.csv")
                
                df_export.T.to_csv(csv_path, sep=';', decimal=',')
                logger.info(f"[{self.__class__.__name__}] Datos históricos validados guardados en CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"[{self.__class__.__name__}] No se pudo generar el CSV intermedio: {str(e)}")

        # Paso 6: Configurar reglas sectoriales
        sector_name = market_data.get("sector", "Industrial")
        sector_config = self.sector_agent.configure(sector_name)
        
        # Paso 7: Generar PDF Final (32 gráficas)
        pdf_path = self.pdf_agent.generate_pdf(
            ticker=ticker,
            market_data=market_data,
            sec_data=normalized,
            sector_config=sector_config
        )
        
        logger.info(f"=== [Orchestrator] Análisis finalizado con éxito para {ticker}. Archivo generado: {pdf_path} ===")
        return pdf_path