import os
import unittest
import pandas as pd
from src.agents.company_overview_agent import CompanyOverviewAgent
from src.tools.pdf_builder import PDFBuilder
from config.settings import OUTPUT_DIR

class TestCompanyOverview(unittest.TestCase):
    def setUp(self):
        self.agent = CompanyOverviewAgent()
        self.mock_market_data = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Tecnología",
            "industry": "Electrónica de Consumo",
            "current_price": 220.50,
            "market_cap": 3400000000000,
            "description": "Apple designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories."
        }
        self.mock_sec_data = {
            "years": ["2020", "2021", "2022", "2023", "2024"],
            "aligned_series": {
                "Ingresos totales": [274515000000, 365817000000, 394328000000, 383285000000, 391035000000],
                "Beneficio bruto": [104956000000, 152836000000, 170782000000, 169148000000, 180683000000],
                "Beneficio neto de la empresa": [57411000000, 94680000000, 99803000000, 96995000000, 93736000000],
                "Fondos propios totales": [65339000000, 63090000000, 50672000000, 62146000000, 66879000000],
                "Gastos de capital": [-7309000000, -11085000000, -10708000000, -10959000000, -9450000000],
                "Promedio ponderado de acciones diluidas en circulacion": [17352119000, 16701272000, 16187581000, 15744231000, 15343840000]
            }
        }

    def test_overview_analysis_keys(self):
        result = self.agent.analyze("AAPL", self.mock_market_data, self.mock_sec_data)
        self.assertIsInstance(result, dict)
        self.assertIn("veredicto_comprensibilidad", result)
        self.assertIn("resumen_actividad", result)
        self.assertIn("modelo_ingresos", result)
        self.assertIn("mercado_y_clientes", result)
        self.assertIn("propuesta_valor", result)
        self.assertIn("circulo_competencia", result)
        print("\n[OK] Test Overview Analysis: result keys valid. Veredicto:", result.get("veredicto_comprensibilidad"))

    def test_pdf_generation_with_overview(self):
        overview_result = self.agent.analyze("AAPL", self.mock_market_data, self.mock_sec_data)
        
        years = self.mock_sec_data["years"]
        df_financials = pd.DataFrame(self.mock_sec_data["aligned_series"], index=years).reset_index()
        df_financials.rename(columns={'index': 'Periodo Fiscal'}, inplace=True)
        
        test_pdf_path = os.path.join(OUTPUT_DIR, "TEST_AAPL_overview_test.pdf")
        
        pdf_path = PDFBuilder.generate_pdf_report(
            ticker="AAPL",
            current_price=220.50,
            df_financials=df_financials,
            output_pdf_path=test_pdf_path,
            sector_config={"sector_name": "Tecnología"},
            company_overview=overview_result,
            market_data=self.mock_market_data
        )
        
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)
        print(f"\n[OK] Test PDF Report generated successfully at: {pdf_path} (Size: {os.path.getsize(pdf_path)} bytes)")

if __name__ == "__main__":
    unittest.main()
