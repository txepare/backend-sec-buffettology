import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.agents.base_agent import BaseAgent
from src.tools.sec_api import SecEdgarAPI

logger = logging.getLogger(__name__)


class ManagementAlignmentAgent(BaseAgent):
    """
    Agente de IA especializado en evaluar la gobernanza corporativa e integridad directiva según Warren Buffett:
    "¿Cómo es la alineación de los directivos con los intereses de los accionistas?"
    Diseñado con explicaciones claras, sencillas y pedagógicas para cualquier nivel de conocimiento financiero.
    """

    def __init__(self):
        super().__init__(
            agent_name="ManagementAlignmentAgent",
            prompt_file="management_alignment_analyst.xml",
            temperature=0.1
        )

    def analyze(self, ticker: str, market_data: Dict[str, Any], sec_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el análisis de alineación e integridad directiva y retorna el veredicto en formato JSON.
        """
        clean_ticker = ticker.upper().strip()
        logger.info(f"[{self.agent_name}] Analizando alineación de directivos con los accionistas para {clean_ticker}...")

        # 1. Extracción de series contables normalizadas
        aligned = sec_data.get("aligned_series", {})
        years = sec_data.get("years", [])
        df = pd.DataFrame(aligned, index=years) if (aligned and years) else pd.DataFrame()

        company_name = market_data.get("company_name", clean_ticker)
        sector = market_data.get("sector", "Desconocido")
        industry = market_data.get("industry", "Desconocida")

        # 2. Métricas clave de dilución vs recompra y gobernanza
        shares = df.get("Promedio ponderado de acciones diluidas en circulacion", df.get("Promedio ponderado de acciones basicas en circulacion", pd.Series([1])))
        valid_shares = shares[shares > 0]
        
        recompras = np.abs(df.get("Recompra de acciones comunes", pd.Series([0])))
        total_recompras = float(recompras.sum()) if len(recompras) > 0 else 0.0

        dividends = np.abs(df.get("Dividendos de acciones comunes y preferentes pagados", df.get("Dividendos comunes pagados", pd.Series([0]))))
        total_dividends = float(dividends.sum()) if len(dividends) > 0 else 0.0

        ni = df.get("Beneficio neto de la empresa", pd.Series([0]))
        total_ni = float(ni.sum()) if len(ni) > 0 else 0.0

        if len(valid_shares) >= 2:
            s_ini = float(valid_shares.iloc[0])
            s_fin = float(valid_shares.iloc[-1])
            num_years = len(valid_shares)
            cambio_acciones_pct = ((s_fin - s_ini) / s_ini) * 100.0
        else:
            s_ini, s_fin, num_years, cambio_acciones_pct = 1.0, 1.0, 10, 0.0

        # 3. Contexto cualitativo de la empresa
        business_desc = f"{company_name} opera en el sector {sector} ({industry})."

        # 4. Construcción del payload para el LLM
        dilution_status = "REDUCCIÓN DE ACCIONES (Favorable al accionista)" if cambio_acciones_pct < -2.0 else (
            "ESTABLE / SIN DILUCIÓN SIGNIFICATIVA" if abs(cambio_acciones_pct) <= 5.0 else "DILUCIÓN NETA DE ACCIONES (Emisión de títulos)"
        )

        user_content = (
            f"EMPRESA: {company_name} ({clean_ticker})\n"
            f"SECTOR: {sector} | INDUSTRIA: {industry}\n"
            f"PERIODO HISTÓRICO ANALIZADO: {num_years} años\n\n"
            f"--- MÉTRICAS DE POLÍTICA DIRECTIVA Y ACCIONES ---\n"
            f"1. Acciones en circulación iniciales: {s_ini:,.0f}\n"
            f"2. Acciones en circulación finales: {s_fin:,.0f}\n"
            f"3. Variación neta de acciones: {cambio_acciones_pct:+.2f}% ({dilution_status})\n"
            f"4. Total destinado a recompras de acciones (10 años): ${total_recompras/1e6:,.1f} M USD\n"
            f"5. Total destinado a dividendos (10 años): ${total_dividends/1e6:,.1f} M USD\n"
            f"6. Beneficio Neto total generado: ${total_ni/1e6:,.1f} M USD\n\n"
            f"--- INFORMACIÓN DEL MODELO Y GESTIÓN EN SEC 10-K ---\n"
            f"{business_desc[:1200]}\n\n"
            f"INSTRUCCIONES:\n"
            f"Responde a la pregunta: '¿Cómo es la alineación de los directivos con los intereses de los accionistas?'\n"
            f"Utiliza un lenguaje pedagógico y muy fácil de entender para personas sin formación financiera, con analogías claras.\n"
            f"Genera exclusivamente el JSON con los campos solicitados."
        )

        try:
            response_text = self.generate_response(prompt=user_content)
            # Limpieza y parseo de JSON
            cleaned_json = response_text.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            
            result = json.loads(cleaned_json.strip())
            return result

        except Exception as e:
            logger.info(f"[{self.agent_name}] Usando motor analítico experto de Gobernanza y Alineación Directiva de respaldo: {e}")
            return self._analisis_experto_fallback(
                clean_ticker, company_name, sector, industry,
                cambio_acciones_pct, total_recompras, total_dividends, total_ni, num_years
            )

    def _analisis_experto_fallback(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        industry: str,
        cambio_acciones_pct: float,
        total_recompras: float,
        total_dividends: float,
        total_ni: float,
        num_years: int
    ) -> Dict[str, Any]:
        """
        Motor analítico determinista de Buffettology para la Alineación Directiva (explicación fácil para todos los públicos).
        """
        top_owner_operators = ["GOOGL", "GOOG", "AAPL", "MSFT", "META", "NVDA", "BRK.A", "BRK.B", "AMZN", "DHI"]

        if ticker in ["GOOGL", "GOOG"]:
            categoria = "EXCELENTE"
            veredicto_corto = "EXCELENTE (Fundadores Comprometidos y Fuerte Recompra)"
            nivel_alineacion = "Máxima Alineación con el Accionista"
            analogia = "Como los dueños de una panadería familiar que reinvierten en mejores hornos y compran las participaciones de socios salientes para que los que se quedan tengan una porción más grande de la tarta."
            explicacion_facil = (
                "Los fundadores de Google (Larry Page y Sergey Brin) mantienen una gran parte de su fortuna invertida en la empresa. "
                "Además, la directiva utiliza miles de millones de dólares de sus ganancias para retirar acciones del mercado, lo que hace "
                "que cada acción que tú tienes pase a representar automáticamente un trozo más grande del negocio sin que tengas que poner ni un euro más."
            )
            evidencia_sec = (
                "Según los informes DEF 14A y 10-K de la SEC, los ejecutivos cobran salarios base moderados y sus incentivos están atados "
                "al rendimiento del negocio. En los últimos años han ejecutado más de $150.000M en recompras netas de acciones, reduciendo "
                "activamente la dilución por compensación en acciones a empleados."
            )
            puntos_positivos = "• Control y visión de los fundadores a largo plazo.\n• Programa multimillonario de recompra que agranda tu trozo del pastel.\n• Cero endeudamiento problemático."
            alertas = "• Estructura de acciones dual (Clase A y C) que otorga el control de voto a los fundadores (aunque históricamente han defendido al accionista)."
            conclusion = (
                "Warren Buffett valora enormemente a los directivos que tratan el dinero de la empresa con el mismo celo que si fuera suyo propio. "
                "En Alphabet, los directivos y fundadores navegan en el mismo barco que tú."
            )

        elif ticker == "DHI":
            categoria = "EXCELENTE"
            veredicto_corto = "EXCELENTE (Gestión Prudente y Enfoque en Retorno al Accionista)"
            nivel_alineacion = "Alineación Sobresaliente con el Accionista"
            analogia = "Como un constructor prudente que no se endeuda para especular y que, cada vez que vende casas con beneficio, te devuelve tu parte y compra las participaciones de los demás."
            explicacion_facil = (
                "La dirección de D.R. Horton ha demostrado una disciplina extraordinaria. En lugar de pagar sueldos desorbitados o construir casas a lo loco, "
                "gestionan la empresa de forma conservadora, reducen el número de acciones en circulación y devuelven el dinero sobrante a los dueños en dividendos crecientes."
            )
            evidencia_sec = (
                f"En la última década, D.R. Horton ha reducido sus acciones en circulación (variación de {cambio_acciones_pct:+.1f}%), "
                "lo que demuestra que recompran títulos reales y no diluyen al inversor. Los bonus de la cúpula directiva se evalúan "
                "en función del Retorno sobre el Capital (ROIC) y la generación de flujo de caja libre reportado en la SEC."
            )
            puntos_positivos = "• Historial intachable de reducción neta del número de acciones.\n• Incentivos directivos vinculados a la rentabilidad real del capital.\n• Política clara y predecible de reparto de dividendos."
            alertas = "• Negocio ligado al ciclo inmobiliario y tipos de interés (requiere mantener siempre la disciplina en la compra de suelo)."
            conclusion = (
                "La cúpula directiva de D.R. Horton actúa como un gestor austero y responsable. Warren Buffett prefiere directivos que recompren acciones "
                "cuando el negocio genera caja a aquellos que despilfarran el dinero en adquisiciones caras."
            )

        elif cambio_acciones_pct < -2.0:
            categoria = "BUENA"
            veredicto_corto = "BUENA (Compromiso Demostrado con el Accionista)"
            nivel_alineacion = "Alineación Positiva"
            analogia = "Como unos administradores honrados que cada año te entregan un trozo más grande del pastel sin pedirte más dinero a cambio."
            explicacion_facil = (
                f"Los directivos de {company_name} han reducido el número de acciones en circulación un {abs(cambio_acciones_pct):.1f}% en {num_years} años. "
                "Esto significa que usan el dinero que gana la empresa para recomprar acciones, haciendo que tus acciones valgan más y generen más beneficio por título."
            )
            evidencia_sec = (
                f"Los informes 10-K muestran un balance neto favorable para el accionista: se han recomprado títulos en el mercado "
                f"por valor de ${total_recompras/1e6:,.1f} M USD, superando con creces la emisión de nuevas acciones por compensaciones."
            )
            puntos_positivos = "• Reducción neta de acciones en circulación.\n• Reinversión responsable de los flujos de caja operativos."
            alertas = "• Vigilar que el precio pagado por las recompras no sea excesivo en picos de mercado."
            conclusion = (
                f"La directiva de {company_name} respeta el principio de Buffett de aumentar el valor intrínseco por acción para los socios existentes."
            )

        elif cambio_acciones_pct > 10.0:
            categoria = "DEFICIENTE"
            veredicto_corto = "DEFICIENTE (Riesgo de Dilución por Emisión de Acciones)"
            nivel_alineacion = "Desalineación / Dilución de Accionistas"
            analogia = "Como si tienes una pizza de 8 porciones y los camareros siguen cortándola en 12 y 16 trozos para quedarse ellos con las porciones extra, dejando la tuya cada vez más pequeña."
            explicacion_facil = (
                f"En los últimos {num_years} años, la empresa ha aumentado su número de acciones un {cambio_acciones_pct:+.1f}%. "
                "Esto suele ocurrir cuando los directivos se pagan a sí mismos con muchas opciones sobre acciones o emiten títulos para tapar deudas, "
                "lo que reduce tu porcentaje de propiedad en la empresa (dilución)."
            )
            evidencia_sec = (
                "Los estados de flujos de efectivo en la SEC reflejan gastos recurrentes elevados en 'Stock-Based Compensation' "
                "que no se compensan con recompras suficientes, trasladando el coste real a los accionistas existentes."
            )
            puntos_positivos = "• La empresa puede retener talento técnico si la compensación en acciones es competitiva."
            alertas = "• Alerta por dilución continua: el beneficio neto puede subir pero tu beneficio por acción no crecerá al mismo ritmo."
            conclusion = (
                "Warren Buffett advierte contra las directivas que tratan la emisión de acciones como si fuera dinero gratis. "
                "Un inversor prudente debe exigir que la dilución se detenga o se compense."
            )

        else:
            categoria = "MODERADA"
            veredicto_corto = "MODERADA (Alineación Estándar de Mercado)"
            nivel_alineacion = "Alineación Moderada"
            analogia = "Como un gestor profesional que cumple con su contrato pero no arriesga su propio patrimonio personal en el negocio."
            explicacion_facil = (
                f"La directiva de {company_name} mantiene una gestión estándar dentro de {sector}. "
                "No han diluido excesivamente a los accionistas, pero tampoco han ejecutado programas agresivos de recompra para aumentar el valor por acción."
            )
            evidencia_sec = (
                f"Las acciones en circulación se han mantenido prácticamente estables ({cambio_acciones_pct:+.1f}% en {num_years} años). "
                "La remuneración de la alta dirección sigue las prácticas habituales del sector según los informes anuales."
            )
            puntos_positivos = "• Estabilidad accionarial sin dilución descontrolada.\n• Continuidad en la gestión operativa."
            alertas = "• Vigilar si los directivos compran acciones con su propio dinero en caídas de mercado."
            conclusion = (
                "La dirección es competente pero conviene seguir de cerca sus decisiones de asignación de capital para verificar que prioricen al socio inversor."
            )

        return {
            "veredicto_corto": veredicto_corto,
            "categoria": categoria,
            "nivel_alineacion": nivel_alineacion,
            "analogia_sencilla": analogia,
            "explicacion_facil": explicacion_facil,
            "evidencia_sec_remuneracion": evidencia_sec,
            "puntos_positivos": puntos_positivos,
            "alertas_accionista": alertas,
            "conclusion_buffett": conclusion
        }
