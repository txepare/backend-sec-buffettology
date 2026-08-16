import logging
from typing import Dict, Any
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """
    Agente auditor para certificar la consistencia de los datos financieros.
    """

    def __init__(self):
        super().__init__(
            agent_name="ValidationAgent",
            prompt_file="validator_agent.xml",
            temperature=0.0
        )

    def validate_data(self, normalized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audita las series temporales y asegura que no haya vacíos insuperables.
        """
        years = normalized_data.get("years", [])
        aligned_series = normalized_data.get("aligned_series", {})

        warnings = []
        errors = []

        if len(years) < 10:
            warnings.append(f"Histórico parcial detectado: {len(years)} años disponibles (esperados 10-12).")

        # Verificar que las series tengan la misma longitud que los años
        for key, series in aligned_series.items():
            if len(series) != len(years):
                errors.append(f"Discrepancia en la longitud de la serie contable: {key}.")

        status = "PASSED" if not errors else "FAILED"

        logger.info(f"[{self.agent_name}] Resultado de la auditoría: {status}")

        return {
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "historical_completeness": f"{len(years)}_YEARS_AVAILABLE"
        }

    def evaluate(self, normalized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Punto de entrada llamado por el OrchestratorAgent.
        """
        audit_res = self.validate_data(normalized_data)
        status = audit_res.get("status", "FAILED")
        estado_auditoria = "APROBADO" if status == "PASSED" else "RECHAZADO"
        motivos = audit_res.get("errors", []) + audit_res.get("warnings", [])
        if not motivos and estado_auditoria == "APROBADO":
            motivos = ["Datos contables verificados y consistentes."]

        return {
            "estado_auditoria": estado_auditoria,
            "motivos": motivos,
            "partidas_a_corregir": audit_res.get("errors", [])
        }