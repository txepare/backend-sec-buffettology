from pydantic import BaseModel, Field
from typing import List, Dict

class ChartDefinition(BaseModel):
    chart_id: int
    title: str
    metric_x: str
    metric_y: str

class SectorConfiguration(BaseModel):
    sector_name: str = Field(description="Nombre del sector principal")
    applied_framework: str = Field(default="Buffettology")
    balance_structure: str = Field(description="Tipo de clasificación del balance (Ej. Clasificado, Orden de liquidez)")
    metrics_to_compute: List[str] = Field(description="Lista de métricas clave a evaluar para este sector")
    debt_categories: List[str] = Field(description="Categorías principales de deuda a vigilar")
    chart_definitions: List[ChartDefinition]