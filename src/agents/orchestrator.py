import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import Dict, Any

from src.agents.market_agent import MarketDataAgent
from src.agents.sec_agent import SecExtractorAgent
from src.agents.normalizer_agent import GAAPNormalizerAgent
from src.agents.sector_agent import SectorConfiguratorAgent
from src.agents.validator_agent import ValidationAgent
from src.agents.company_overview_agent import CompanyOverviewAgent
from src.agents.revenue_segments_agent import RevenueSegmentsAgent
from src.agents.income_statement_flow_agent import IncomeStatementFlowAgent
from src.agents.monopoly_agent import MonopolyAnalysisAgent
from src.agents.retained_earnings_agent import RetainedEarningsAgent
from src.agents.management_agent import ManagementAlignmentAgent
from src.agents.accounting_forensic_agent import AccountingForensicAgent
from src.agents.pdf_agent import PDFGeneratorAgent
from src.tools.sec_api import SecEdgarAPI
from src.tools.cache_manager import CacheManager
from config.settings import OUTPUT_DIR

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self):
        self.market_agent = MarketDataAgent()
        self.sec_agent = SecExtractorAgent()
        self.normalizer_agent = GAAPNormalizerAgent()
        self.sector_agent = SectorConfiguratorAgent()
        self.validator_agent = ValidationAgent()
        self.overview_agent = CompanyOverviewAgent()
        self.segments_agent = RevenueSegmentsAgent()
        self.flow_agent = IncomeStatementFlowAgent()
        self.monopoly_agent = MonopolyAnalysisAgent()
        self.retained_agent = RetainedEarningsAgent()
        self.management_agent = ManagementAlignmentAgent()
        self.forensic_agent = AccountingForensicAgent()
        self.pdf_agent = PDFGeneratorAgent()

    def run_analysis(self, query: str) -> str:
        # Resolver automáticamente si se pasó un nombre de empresa o un ticker
        ticker = SecEdgarAPI.resolve_ticker(query)
        logger.info(f"=== [Orchestrator] Iniciando pipeline de análisis para '{query}' -> Ticker Oficial: {ticker} ===")
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Paso 1 y 2 en paralelo: Datos de Mercado + Extracción SEC inicial
            logger.info(f"[Orchestrator] Ejecutando extracción inicial paralela (Mercado + SEC) para {ticker}...")
            market_future = executor.submit(self.market_agent.get_data, ticker)
            sec_future = executor.submit(self.sec_agent.extract, ticker)
            
            market_data = market_future.result()
            raw_sec = sec_future.result()
            
            # Bucle de Normalización y Validación (Máximo 3 intentos)
            max_intentos = 3
            intento_actual = 1
            datos_validados = False
            normalized = {}
            
            while intento_actual <= max_intentos and not datos_validados:
                logger.info(f"--- [Orchestrator] Ciclo de Normalización y Auditoría (Intento {intento_actual}/{max_intentos}) ---")
                
                # Paso 3: Normalización y Síntesis Matemática
                normalized = self.normalizer_agent.normalize(raw_sec)
                
                # Paso 4: Validación por el Agente Auditor
                respuesta_auditoria = self.validator_agent.evaluate(normalized)
                
                try:
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
                        if intento_actual <= max_intentos:
                            raw_sec = self.sec_agent.extract(ticker)
                        
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
            
            # --- ACELERACIÓN PARALELA DE AGENTES DE IA ---
            logger.info(f"[Orchestrator] Disparando análisis cualitativo profundo en hilos paralelos...")
            
            # Lanzamos en paralelo los 4 agentes independientes de evaluación Buffett
            monopoly_future = executor.submit(
                self.monopoly_agent.analyze, ticker=ticker, market_data=market_data, sec_data=normalized
            )
            retained_future = executor.submit(
                self.retained_agent.analyze, ticker=ticker, market_data=market_data, sec_data=normalized
            )
            management_future = executor.submit(
                self.management_agent.analyze, ticker=ticker, market_data=market_data, sec_data=normalized
            )
            forensic_future = executor.submit(
                self.forensic_agent.analyze, ticker=ticker, market_data=market_data, sec_data=normalized
            )
            
            # Mientras tanto, ejecutamos el flujo secuencial de las páginas ejecutivas (Páginas 1, 2, 3)
            # Paso 7: Descripción y Modelo de Negocio de la Empresa (Página 1)
            company_overview = self.overview_agent.analyze(
                ticker=ticker,
                market_data=market_data,
                sec_data=normalized
            )

            # Paso 8: Fuentes de Ingresos y Desglose Histórico por Segmentos (Página 2)
            segments_data = self.segments_agent.analyze(
                ticker=ticker,
                market_data=market_data,
                sec_data=normalized,
                company_overview=company_overview
            )

            # Paso 9: Diagrama Sankey del Flujo del Estado de Resultados del Último Año (Página 3)
            income_flow_data = self.flow_agent.prepare_flow_data(
                ticker=ticker,
                market_data=market_data,
                sec_data=normalized,
                segments_data=segments_data
            )
            
            # Recolectamos los resultados de los 4 agentes de evaluación en paralelo
            monopoly_analysis = monopoly_future.result()
            retained_earnings_analysis = retained_future.result()
            management_analysis = management_future.result()
            forensic_analysis = forensic_future.result()

        # Paso 14: Generar PDF Final completo con todas las secciones integradas
        pdf_path = self.pdf_agent.generate_pdf(
            ticker=ticker,
            market_data=market_data,
            sec_data=normalized,
            sector_config=sector_config,
            company_overview=company_overview,
            segments_data=segments_data,
            income_flow_data=income_flow_data,
            monopoly_analysis=monopoly_analysis,
            retained_earnings_analysis=retained_earnings_analysis,
            management_analysis=management_analysis,
            forensic_analysis=forensic_analysis
        )
        
        # Paso 15: Mantenimiento y optimización automática del espacio en caché
        try:
            CacheManager.auto_clean()
        except Exception as e:
            logger.warning(f"[Orchestrator] Error en la optimización automática del caché: {e}")

        logger.info(f"=== [Orchestrator] Análisis finalizado con éxito para {ticker}. Archivo generado: {pdf_path} ===")
        return pdf_path