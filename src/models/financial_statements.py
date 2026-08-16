from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class AnnualFinancials(BaseModel):
    year: str = Field(description="Año fiscal, ej. '2023'")
    total_revenue: float = Field(default=0.0)
    net_income: float = Field(default=0.0)
    diluted_eps: float = Field(default=0.0)
    operating_income: float = Field(default=0.0)
    gross_profit: float = Field(default=0.0)
    sga_expense: float = Field(default=0.0)
    rd_expense: float = Field(default=0.0)
    
    # Balance
    total_assets: float = Field(default=0.0)
    total_liabilities: float = Field(default=0.0)
    stockholders_equity: float = Field(default=0.0)
    total_debt: float = Field(default=0.0)
    cash_and_equivalents: float = Field(default=0.0)
    
    # Flujo de Caja
    operating_cash_flow: float = Field(default=0.0)
    capital_expenditures: float = Field(default=0.0)
    free_cash_flow: float = Field(default=0.0)

class ExtractedFinancialData(BaseModel):
    ticker: str
    company_name: str
    available_years_count: int
    annual_reports: List[AnnualFinancials]
    currency: str = "USD"