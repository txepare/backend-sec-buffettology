import argparse
import sys
import logging

from src.agents.orchestrator import OrchestratorAgent

# Stream logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SEC Buffettology Financial Analysis Multi-Agent System")
    parser.add_argument("ticker", type=str, help="Símbolo bursátil de la empresa (Ej. AAPL, MSFT, JPM, KO)")
    args = parser.parse_args()

    ticker = args.ticker.upper().strip()
    print(f"\n=======================================================")
    print(f"  SISTEMA MULTI-AGENTE DE ANÁLISIS FINANCIERO (SEC)   ")
    print(f"  Empresa a analizar: {ticker}")
    print(f"=======================================================\n")

    try:
        orchestrator = OrchestratorAgent()
        pdf_output = orchestrator.run_analysis(ticker)
        print(f"\n[ÉXITO] Informe PDF generado correctamente en:")
        print(f" -> {pdf_output}\n")

    except Exception as e:
        logger.error(f"Error durante la ejecución del análisis: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()