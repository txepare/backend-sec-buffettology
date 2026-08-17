import os
import unittest
import pandas as pd
import matplotlib.pyplot as plt

from src.agents.revenue_segments_agent import RevenueSegmentsAgent
from src.agents.income_statement_flow_agent import IncomeStatementFlowAgent
from src.tools.sankey_builder import SankeyFlowBuilder
from src.tools.pdf_builder import PDFBuilder


class TestRevenueAndSankey(unittest.TestCase):

    def setUp(self):
        self.ticker = "AAPL"
        self.market_data = {
            "company_name": "Apple Inc.",
            "current_price": 224.50,
            "market_cap": 3400000000000,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories."
        }
        self.years = ["2022", "2023", "2024"]
        self.sec_data = {
            "years": self.years,
            "aligned_series": {
                "Ingresos totales": [394328000000.0, 383285000000.0, 391035000000.0],
                "Coste de los bienes vendidos": [223546000000.0, 214137000000.0, 210352000000.0],
                "Beneficio bruto": [170782000000.0, 169148000000.0, 180683000000.0],
                "Gastos de I + D": [26251000000.0, 29915000000.0, 31370000000.0],
                "Gastos de venta generales y administrativos": [25094000000.0, 24932000000.0, 24000000000.0],
                "Beneficio neto de la empresa": [99803000000.0, 96995000000.0, 93736000000.0],
                "Gastos de impuestos": [19300000000.0, 16741000000.0, 15500000000.0],
                "Beneficio neto de la empresa por accion": [6.11, 6.13, 6.08],
                "Ingresos totales por accion": [24.15, 24.22, 25.38],
                "Gastos de capital": [10708000000.0, 10959000000.0, 9450000000.0],
                "Efectivo de Operaciones": [122151000000.0, 110543000000.0, 118254000000.0],
                "Total de activo": [352755000000.0, 352583000000.0, 364980000000.0],
                "Total de fondos propios": [50672000000.0, 62146000000.0, 66800000000.0],
                "Deuda a largo plazo": [98959000000.0, 95281000000.0, 85760000000.0]
            }
        }

    def test_revenue_segments_agent(self):
        agent = RevenueSegmentsAgent()
        result = agent.analyze(self.ticker, self.market_data, self.sec_data)

        self.assertIn("segmentos", result)
        self.assertIn("historico_segmentos", result)
        self.assertIn("analisis_diversificacion", result)
        self.assertTrue(len(result["segmentos"]) >= 2)
        print("\n[OK] RevenueSegmentsAgent devolvió estructura JSON válida:", list(result.keys()))

    def test_sankey_flow_generation(self):
        agent = RevenueSegmentsAgent()
        segments_data = agent.analyze(self.ticker, self.market_data, self.sec_data)

        flow_agent = IncomeStatementFlowAgent()
        flow_prepared = flow_agent.prepare_flow_data(self.ticker, self.market_data, self.sec_data, segments_data)

        self.assertIn("financial_flow", flow_prepared)
        self.assertIn("segments_data", flow_prepared)

        fig = flow_agent.generate_flow_chart(flow_prepared)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        print("\n[OK] IncomeStatementFlowAgent generó la figura Sankey correctamente.")

    def test_pdf_report_pages(self):
        from src.agents.company_overview_agent import CompanyOverviewAgent

        overview_agent = CompanyOverviewAgent()
        overview_data = overview_agent.analyze(self.ticker, self.market_data, self.sec_data)

        rev_agent = RevenueSegmentsAgent()
        segments_data = rev_agent.analyze(self.ticker, self.market_data, self.sec_data)

        flow_agent = IncomeStatementFlowAgent()
        flow_prepared = flow_agent.prepare_flow_data(self.ticker, self.market_data, self.sec_data, segments_data)

        df_financials = pd.DataFrame(self.sec_data["aligned_series"], index=self.years).reset_index()
        df_financials.rename(columns={'index': 'Periodo Fiscal'}, inplace=True)

        test_pdf = os.path.join(os.path.dirname(__file__), "test_multipage_output.pdf")
        
        out_path = PDFBuilder.generate_pdf_report(
            ticker=self.ticker,
            current_price=224.50,
            df_financials=df_financials,
            output_pdf_path=test_pdf,
            sector_config={"sector": "Technology"},
            company_overview=overview_data,
            segments_data=segments_data,
            income_flow_data=flow_prepared,
            market_data=self.market_data
        )

        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)
        print(f"\n[OK] PDF multipágina generado exitosamente en: {out_path} ({os.path.getsize(out_path)} bytes)")

        # Limpiar archivo de prueba
        if os.path.exists(test_pdf):
            os.remove(test_pdf)


if __name__ == "__main__":
    unittest.main()
