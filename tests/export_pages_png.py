import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.tools.pdf_builder import PDFBuilder, show_company_overview, show_revenue_segments_table, show_income_statement_flow
from src.agents.company_overview_agent import CompanyOverviewAgent
from src.agents.revenue_segments_agent import RevenueSegmentsAgent
from src.agents.income_statement_flow_agent import IncomeStatementFlowAgent
from src.tools.sankey_builder import SankeyFlowBuilder

def export_pngs():
    ticker = "GOOGL"
    market_data = {
        "company_name": "Alphabet Inc.",
        "current_price": 343.72,
        "market_cap": 2100000000000,
        "sector": "Technology",
        "industry": "Internet Content & Information",
        "description": "Alphabet Inc. offers products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America. It operates through Google Services, Google Cloud, and Other Bets segments."
    }
    years = ["2022", "2023", "2024"]
    sec_data = {
        "years": years,
        "aligned_series": {
            "Ingresos totales": [282836000000.0, 307394000000.0, 350000000000.0],
            "Coste de los bienes vendidos": [126203000000.0, 133332000000.0, 145000000000.0],
            "Beneficio bruto": [156633000000.0, 174062000000.0, 205000000000.0],
            "Gastos de I + D": [39500000000.0, 45427000000.0, 50000000000.0],
            "Gastos de venta generales y administrativos": [27244000000.0, 28169000000.0, 30000000000.0],
            "Beneficio neto de la empresa": [59972000000.0, 73795000000.0, 95000000000.0],
            "Gastos de impuestos": [11356000000.0, 11482000000.0, 15000000000.0]
        }
    }

    overview_agent = CompanyOverviewAgent()
    overview_data = overview_agent.analyze(ticker, market_data, sec_data)

    rev_agent = RevenueSegmentsAgent()
    segments_data = rev_agent.analyze(ticker, market_data, sec_data)

    flow_agent = IncomeStatementFlowAgent()
    flow_prepared = flow_agent.prepare_flow_data(ticker, market_data, sec_data, segments_data)

    class MockPdf:
        def __init__(self):
            self.figs = []
        def savefig(self, fig):
            self.figs.append(fig)

    mock_pdf = MockPdf()
    show_company_overview(mock_pdf, overview_data, ticker=ticker, market_data=market_data)
    show_revenue_segments_table(mock_pdf, segments_data, ticker=ticker, market_data=market_data)
    show_income_statement_flow(mock_pdf, flow_prepared, ticker=ticker, market_data=market_data)

    os.makedirs("tests/output_pngs", exist_ok=True)
    for idx, fig in enumerate(mock_pdf.figs):
        png_path = f"tests/output_pngs/page_{idx+1}.png"
        fig.savefig(png_path, dpi=150)
        print(f"Saved: {png_path}")

if __name__ == "__main__":
    export_pngs()
