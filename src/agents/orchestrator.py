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

    def run_analysis(self, query: str, force_regenerate: bool = False) -> str:
        from datetime import datetime
        # Resolver automáticamente si se pasó un nombre de empresa o un ticker
        ticker = SecEdgarAPI.resolve_ticker(query)
        logger.info(f"=== [Orchestrator] Iniciando pipeline de análisis para '{query}' -> Ticker Oficial: {ticker} ===")
        
        # 0. Verificación de Caché Instantánea: Si ya existe un PDF generado hoy, entregarlo al instante (<50ms)
        if not force_regenerate and os.path.exists(OUTPUT_DIR):
            date_today = datetime.now().strftime("%Y-%m-%d")
            for fname in os.listdir(OUTPUT_DIR):
                if fname.startswith(f"{ticker}_") and fname.endswith(".pdf") and date_today in fname:
                    existing_path = os.path.join(OUTPUT_DIR, fname)
                    if os.path.exists(existing_path) and os.path.getsize(existing_path) > 20000:
                        logger.info(f"[Orchestrator] ¡PDF reciente del día encontrado en caché para {ticker}! Retornando al instante ({existing_path})")
                        return existing_path

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Paso 1 y 2 en paralelo: Datos de Mercado + Extracción SEC inicial
            logger.info(f"[Orchestrator] Ejecutando extracción inicial paralela (Mercado + SEC Facts) para {ticker}...")
            market_future = executor.submit(self.market_agent.get_data, ticker)
            sec_future = executor.submit(self.sec_agent.extract, ticker)
            narrative_future = executor.submit(SecEdgarAPI.fetch_company_10k_narrative, ticker)
            
            market_data = market_future.result()
            raw_sec = sec_future.result()
            _ = narrative_future.result() # Precarga de texto 10-K en caché para los agentes
            
            # Paso 3: Normalización y Síntesis Matemática (Determinista)
            logger.info(f"[Orchestrator] Normalizando datos contables para {ticker}...")
            normalized = self.normalizer_agent.normalize(raw_sec)
            
            # Paso 4: Validación por el Agente Auditor
            try:
                audit_res = self.validator_agent.evaluate(normalized)
                logger.info(f"[Orchestrator] Auditoría contable completada: {audit_res.get('estado_auditoria', 'OK')}")
            except Exception as e:
                logger.debug(f"[Orchestrator] Aviso en auditoría: {e}")

            # Paso 5: Exportar datos extraídos a CSV
            years = normalized.get("years", [])
            try:
                aligned_series = normalized.get("aligned_series", {})
                if years and aligned_series:
                    df_export = pd.DataFrame(aligned_series, index=years)
                    df_export.index.name = "Periodo Fiscal"
                    csv_path = os.path.join(OUTPUT_DIR, f"{ticker}_extracted_historical.csv")
                    df_export.T.to_csv(csv_path, sep=';', decimal=',')
            except Exception as e:
                logger.debug(f"[Orchestrator] No se pudo generar CSV intermedio: {e}")

            # Paso 6: Configurar reglas sectoriales
            sector_name = market_data.get("sector", "Industrial")
            sector_config = self.sector_agent.configure(sector_name)
            
            # --- ACELERACIÓN PARALELA TOTAL: TODOS LOS AGENTES DE IA SIMULTÁNEAMENTE ---
            logger.info(f"[Orchestrator] Disparando los 6 agentes de análisis cualitativo en hilos concurrentes paralelos...")
            
            overview_future = executor.submit(
                self.overview_agent.analyze, ticker=ticker, market_data=market_data, sec_data=normalized
            )
            segments_future = executor.submit(
                self.segments_agent.analyze, ticker=ticker, market_data=market_data, sec_data=normalized
            )
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
            
            # Recolectamos concurrentemente los resultados de los 6 agentes
            company_overview = overview_future.result()
            segments_data = segments_future.result()
            monopoly_analysis = monopoly_future.result()
            retained_earnings_analysis = retained_future.result()
            management_analysis = management_future.result()
            forensic_analysis = forensic_future.result()

            # Paso 9: Diagrama del Flujo del Estado de Resultados (Cálculo instantáneo en memoria)
            income_flow_data = self.flow_agent.prepare_flow_data(
                ticker=ticker,
                market_data=market_data,
                sec_data=normalized,
                segments_data=segments_data
            )

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