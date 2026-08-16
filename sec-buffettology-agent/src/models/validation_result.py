from pydantic import BaseModel, Field
from typing import List

class ValidationResult(BaseModel):
    validation_status: str = Field(description="PASSED o FAILED")
    errors: List[str] = Field(default_factory=list, description="Lista de errores matemáticos o contables fatales")
    warnings: List[str] = Field(default_factory=list, description="Avisos sobre años faltantes o anomalías")
    historical_completeness: str = Field(description="Ej. '12_YEARS_COMPLETE' o 'PARTIAL_X_YEARS'")