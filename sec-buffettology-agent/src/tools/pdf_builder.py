import os
import gc
import textwrap
import logging
from io import BytesIO
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 0
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from pypdf import PdfWriter, PdfReader

logger = logging.getLogger(__name__)

# --- CONSTANTES DE ESTÉTICA ---
REPORT_THEME = {
    "navy": "#0B1F33", "teal": "#1F6F78", "gold": "#D9A441", "sky": "#6FA8DC",
    "ink": "#203040", "text": "#334E68", "muted": "#627D98", "line": "#C8D4E3",
    "panel": "#F7F4EE", "canvas": "#FFFDFC", "positive": "#2D6A4F", "negative": "#B85C38",
}

VALUE_INVESTING_NOTES = [
    ("Comparar la empresa con su competencia", "Hay que comparar la empresa con su competencia y si tiene menos dinero por cobrar que la competencia puede significar que entonces tiene una ventaja competitva ya que esta ofreciendo peores condiciones de pago y sigue tendiendo beneficios."),
    ("El Pasivo de una empresa (Propiedad, Planta y Equipo)", "El property/plant and equipo - Total Net debe ser menor cada año, porque al Bruto se le resta de amortización y una buena empresa no necesita reinvertir sus beneficios en maquinaria nueva."),
    ("Riesgo de Deuda a Corto Plazo", "Los bancos buscan dinero a corto plazo con un interes del 5%, luego lo prestan a largo plazo con un interes mayor y lo que buscan es refinanciar la deuda a corto plazo hasta que reciban el beneficio de la deuda a largo plazo ganando así la diferencia, pero esto es mala estrategia porque puede que los Préstamos de corto plazo suban o dejen de hacérnoslos y entonces tengamos que pagarlos y no podamos con la deuda a corto plazo, por lo que buscaremos empresas con poca deuda a corto plazo y alta a largo plazo."),
    ("Fondos Propios de la Empresa", "El total del activo menos el total del pasivo son los fondos propios de la empresa."),
    ("Relación Deuda/Fondos Propios en Bancos", "En cambio, en los bancos la relación entre la deuda y los fondos propios es mucho mayor porque estos se endeudan para luego obtener beneficios, suelen tener 10 dólares de deuda por 1 dólar de fondos propios; si es menor es mucho mejor."),
    ("Prima de Emisión", "Additional paid-in capital es la prima de emisión."),
    ("Reservas", "Las reservas es el beneficio que no se ha reinvertido o pagado a los accionistas."),
    ("Acciones Propias", "Cuando encuentre acciones propias en el balance ver el capítulo 46 del libro (referencia a un texto de análisis financiero)."),
    ("Valor Contable por Acción (P/B)", "Los fondos propios de la empresa entre el número total de acciones en circulación es el valor contable por acción. Este debe cumplir (valor contable por acción * 1.33 < Precio de la acción)."),
    ("Riesgo: Ganancia a Través de los Inversores", "Dentro del flujo de efectivo, si el saldo de efectivo obtenido con las actividades de explotación es permanentemente negativo, mientras que el saldo de efectivo obtenido con las actividades de financiación es permanentemente positivo, la empresa está ganando dinero a costa de sus inversores y no por su actividad económica."),
    ("Opciones sobre Acciones o Incentivos de los Directivos", "En el 10-K hay que buscar 'Opciones sobre acciones' y hay que ver cuántas se lanzan; son los incentivos para los directivos. Si hay muchas o si se lanzan a precios muy ventajosos no es bueno para los accionistas."),
    ("Contabilidades Engañosas (No Recurrentes)", "Hay que ver si las prácticas de contabilidad de la empresa están diseñadas para que sus resultados financieros sean transparentes, o si lo que persiguen es que sean opacos. Si las cargas <No recurrentes> no dejan de ocurrir, si las partidas extraordinarias aparecen con tanta frecuencia lo único que se puede decir es que son ordinarias. Si ponen el EBITDA antes que el Beneficio Neto, etc."),
    ("Endeudamiento a Largo Plazo", "El endeudamiento a largo plazo debe estar por debajo del 50 por ciento del capital total. En los estados financieros hay que comprobar si el endeudamiento a largo plazo está contratado a tipo fijo o tipo variable, pero que sea fijo mejor que variable."),
    ("Pregunta 1 de Warren", "¿Tiene la empresa un monopolio facilmente identificable?"),
    ("Pregunta 2 de Warren", "¿son solidos los beneficios de la empresa?"),
    ("Pregunta 3 de Warren", "¿Esta la empresa financiada prudentemente?"),
    ("Pregunta 4 de Warren", "Obtiene continuamente la empresa una tasa elevada de rentabilidad del capital de los accionistas"),
    ("Pregunta 5 de Warren", "¿Retiene la empresa sus beneficios?"),
    ("Pregunta 6 de Warren", "¿Cuanto tienen que invertir la empresa para mantener las operaciones actuales?"),
    ("Pregunta 7 de Warren", "¿Puede la empresa reinvertir los beneficios no distribuidos en nuevas oportunidades de negocio, en expandir sus operaciones o en recompra de acciones?"),
    ("Pregunta 8 de Warren", "¿Puede la empresa ajustar los precios según la inflación?"),
    ("Pregunta 9 de Warren", "¿Incrementará el valor añadido de los beneficios no distribuidos el valor de mercado de la empresa?"),
]

def formato_legible(num):
    if pd.isna(num) or num == 0: return "0"
    if abs(num) < 10 and num % 1 != 0: return f'{num:.2f}'
    magnitude = 0
    val = abs(num)
    while val >= 1000 and magnitude < 3:
        magnitude += 1
        val /= 1000.0
    suffix = ['', 'K', 'M', 'B'][magnitude]
    signo = "-" if num < 0 else ""
    formateado = f"{val:.2f}".rstrip('0').rstrip('.')
    return f'{signo}{formateado}{suffix}'

def formatear_grafica(barras, ax):
    etiquetas = [formato_legible(val) for val in barras.datavalues]
    ax.bar_label(barras, labels=etiquetas, padding=3, fontsize=9)
    ax.margins(y=0.15)

def estilizar_figura_profesional(fig):
    fig.patch.set_facecolor(REPORT_THEME["canvas"])
    for ax in fig.axes:
        ax.set_facecolor(REPORT_THEME["canvas"])
        ax.grid(axis='y', color=REPORT_THEME["line"], linestyle='--', linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
        for spine_name in ('top', 'right'): ax.spines[spine_name].set_visible(False)
        for spine_name in ('left', 'bottom'):
            ax.spines[spine_name].set_color(REPORT_THEME["line"])
            ax.spines[spine_name].set_linewidth(1)
        ax.tick_params(colors=REPORT_THEME["muted"], labelsize=9)
        ax.title.set_color(REPORT_THEME["navy"]); ax.title.set_fontsize(11); ax.title.set_fontweight('bold')
        ax.xaxis.label.set_color(REPORT_THEME["text"]); ax.yaxis.label.set_color(REPORT_THEME["text"])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: formato_legible(x)))
        for container in ax.containers:
            if not hasattr(container, "patches"): continue
            for barra in container.patches:
                barra.set_facecolor(REPORT_THEME["negative"] if barra.get_height() < 0 else REPORT_THEME["sky"])
                barra.set_edgecolor("white"); barra.set_linewidth(1); barra.set_alpha(0.96)
    try: fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.95))
    except: pass

class PDFBuilder:
    @staticmethod
    def _calcular_metricas_derivadas(df: pd.DataFrame) -> pd.DataFrame:
        def safe_div(a, b): return np.where(b != 0, a / b, 0)
        
        # Selección de acciones (Prioridad Diluidas -> Básicas -> Total)
        shares = df.get('Promedio ponderado de acciones diluidas en circulacion', df.get('Promedio ponderado de acciones basicas en circulacion', df.get('Total de acciones fuera. en la fecha de presentacion', 0)))
        shares = np.where(shares == 0, 1, shares)
        
        # 32 CÁLCULOS EXACTOS
        df['Beneficio neto de la empresa por accion'] = safe_div(df.get('Beneficio neto de la empresa', 0), shares)
        df['Ingresos totales por accion'] = safe_div(df.get('Ingresos totales', 0), shares)
        df['Rendimiento de los Fondos Propios'] = safe_div(df.get('Beneficio neto de la empresa', 0), np.where(df.get('Fondos propios totales', 0)==0, 1, df.get('Fondos propios totales', 1)))
        
        df['Gastos de venta generales y administrativos/Beneficio bruto'] = safe_div(df.get('Gastos de venta generales y administrativos', 0), np.where(df.get('Beneficio bruto', 0)==0, 1, df.get('Beneficio bruto', 1)))
        df['Beneficio bruto/Ingresos totales'] = safe_div(df.get('Beneficio bruto', 0), np.where(df.get('Ingresos totales', 0)==0, 1, df.get('Ingresos totales', 1)))
        
        # Beneficios del Propietario = (Net Income + D&A - CapEx) / Shares
        df['Beneficios del propietario por accion'] = safe_div(
            df.get('Beneficio neto de la empresa', 0) + df.get('Depreciacion y amortizacion', 0) - np.abs(df.get('Gastos de capital', 0)), shares)
        
        df['Depreciacion y amortizacion/Beneficio bruto'] = safe_div(df.get('Depreciacion y amortizacion', 0), np.where(df.get('Beneficio bruto', 0)==0, 1, df.get('Beneficio bruto', 1)))
        df['Gasto en I + D/Beneficio bruto'] = safe_div(df.get('Gastos de I + D', 0), np.where(df.get('Beneficio bruto', 0)==0, 1, df.get('Beneficio bruto', 1)))
        df['Gastos por intereses/Beneficio neto de la empresa'] = safe_div(np.abs(df.get('Gastos por intereses', 0)), np.where(df.get('Beneficio neto de la empresa', 0)==0, 1, df.get('Beneficio neto de la empresa', 1)))
        df['Gastos de impuestos/Beneficio neto de la empresa'] = safe_div(np.abs(df.get('Gastos de impuestos', 0)), np.where(df.get('Beneficio neto de la empresa', 0)==0, 1, df.get('Beneficio neto de la empresa', 1)))
        df['Beneficio neto de la empresa/Ingresos totales'] = safe_div(df.get('Beneficio neto de la empresa', 0), np.where(df.get('Ingresos totales', 0)==0, 1, df.get('Ingresos totales', 1)))
        
        df['Total de cuentas por cobrar/Ingresos totales'] = safe_div(df.get('Total de cuentas por cobrar', 0), np.where(df.get('Ingresos totales', 0)==0, 1, df.get('Ingresos totales', 1)))
        df['Total de activo corriente/Total pasivo corriente'] = safe_div(df.get('Total de activo corriente', 0), np.where(df.get('Total pasivo corriente', 0)==0, 1, df.get('Total pasivo corriente', 1)))
        
        # Prueba Ácida = (Activo Corriente - Inventario) / Pasivo Corriente
        df['Capital circulante'] = safe_div(df.get('Total de activo corriente', 0) - df.get('Inventario', 0), np.where(df.get('Total pasivo corriente', 0)==0, 1, df.get('Total pasivo corriente', 1)))
        
        df['Beneficio neto de la empresa/Activo total'] = safe_div(df.get('Beneficio neto de la empresa', 0), np.where(df.get('Activo total', 0)==0, 1, df.get('Activo total', 1)))
        df['Rotacion de existencias'] = safe_div(df.get('Ingresos totales', 0), np.where(df.get('Inventario', 0)==0, 1, df.get('Inventario', 1)))
        
        df['Prestamos de corto plazo/Deuda a largo plazo'] = safe_div(df.get('Prestamos de corto plazo', 0), np.where(df.get('Deuda a largo plazo', 0)==0, 1, df.get('Deuda a largo plazo', 1)))
        df['Deuda a largo plazo/Beneficio neto de la empresa'] = safe_div(df.get('Deuda a largo plazo', 0), np.where(df.get('Beneficio neto de la empresa', 0)<=0, 1, df.get('Beneficio neto de la empresa', 1)))
        df['Ratio entre deuda y fondos propios'] = safe_div(df.get('Pasivo Total', 0), np.where(df.get('Fondos propios totales', 0)==0, 1, df.get('Fondos propios totales', 1)))
        
        # Reservas = Beneficio Neto - Dividendos - Recompras (Acumuladas en el tiempo)
        divs = np.abs(df.get('Dividendos comunes pagados', df.get('Dividendos de acciones comunes y preferentes pagados', 0)))
        recompras = np.abs(df.get('Recompra de acciones comunes', 0))
        df['Reservas'] = (df.get('Beneficio neto de la empresa', 0) - divs - recompras).cumsum()
        
        df['Gastos de capital/Beneficios netos'] = safe_div(np.abs(df.get('Gastos de capital', 0)), np.where(df.get('Beneficio neto de la empresa', 0)<=0, 1, df.get('Beneficio neto de la empresa', 1)))
        
        df['Punto de equilibrio'] = safe_div(df.get('Gastos operativos totales', 0), (1 - safe_div(df.get('Coste de los bienes vendidos', 0), np.where(df.get('Ingresos totales', 0)==0, 1, df.get('Ingresos totales', 1)))))
        df['Margen de seguridad'] = 1 - safe_div(df.get('Gastos operativos totales', 0), (df.get('Ingresos totales', 0) + df.get('Coste de los bienes vendidos', 0)))
        
        df['Deuda_Total_Calculada'] = df.get('Prestamos de corto plazo', 0) + df.get('Porcion corriente de la deuda a largo plazo', 0) + df.get('Deuda a largo plazo', 0)
        df['Flujo de caja libre/Deuda total'] = safe_div(df.get('Efectivo de Operaciones', 0) - np.abs(df.get('Gastos de capital', 0)), np.where(df['Deuda_Total_Calculada']==0, 1, df['Deuda_Total_Calculada']))
        
        df['Valor contable / Accion'] = safe_div(df.get('Fondos propios totales', 0), shares)

        return df.fillna(0)

    @staticmethod
    def generate_pdf_report(ticker: str, current_price: float, df_financials: pd.DataFrame, output_pdf_path: str, sector_config: dict) -> str:
        logger.info(f"[PDF Builder] Generando los 32 GRÁFICOS MAESTROS para {ticker}...")
        
        df = PDFBuilder._calcular_metricas_derivadas(df_financials)
        content_buffer = BytesIO()
        
        with PdfPages(content_buffer) as pdf:
            # LAS 32 GRÁFICAS DE BUFFETTOLOGY
            graficas_a_generar = [
                # 1
                ('Beneficio neto de la empresa por accion', 'Beneficio Neto / Accion', 'Beneficio neto por accion', 'El Beneficio Neto por acción debe mostrar una tendencia creciente y ser regular a lo largo del tiempo.', None, None, None, None),
                # 2
                ('Ingresos totales por accion', 'Ingresos / Accion', 'Ingresos totales por accion', 'Los ingresos generados por acción deben mantener un crecimiento estable y sostenido.', None, None, None, None),
                # 3
                ('Rendimiento de los Fondos Propios', 'ROE', 'Rendimiento de los Fondos Propios (ROE)', 'Un ROE consistente y superior al 15% es excelente (siempre que la deuda sea controlada).', 0.15, 'green', None, None),
                # 4
                ('Valor contable / Accion', 'Valor Contable / Accion', 'Valor contable por accion (Book Value)', 'Crecimiento constante del Valor Contable indica creación de riqueza sostenida para el accionista.', None, None, None, None),
                # 5
                ('Gastos de venta generales y administrativos/Beneficio bruto', 'VGA / Beneficio Bruto', 'Gastos VGA / Beneficio bruto', 'Los gastos en VGA deben ser idealmente inferiores al 30% (o al menos consistentes).', 0.8, 'red', 0.3, 'orange'),
                # 6
                ('Beneficio bruto/Ingresos totales', 'Margen Bruto', 'Margen Bruto (Beneficio Bruto / Ingresos)', 'Un porcentaje > 40% constante suele indicar ventaja competitiva (Fijación de precios).', 0.4, 'green', None, None),
                # 7
                ('Beneficios del propietario por accion', 'Beneficios Propietario', 'Beneficios del propietario por accion', 'BPA Ajustado por D&A y CapEx. Crecimiento sostenido es un fuerte indicador de valor intrínseco.', None, None, None, None),
                # 8
                ('Depreciacion y amortizacion/Beneficio bruto', 'D&A / Beneficio Bruto', 'Depreciacion y amortizacion / Beneficio bruto', 'Una baja D&A/Beneficio Bruto (<10%) indica que la empresa no requiere constante reinversión.', 0.1, 'red', None, None),
                # 9
                ('Gasto en I + D/Beneficio bruto', 'I+D / Beneficio Bruto', 'Gasto en I+D / Beneficio bruto', 'El gasto en I+D no debe superar el 10% del Beneficio Bruto (menor obsolescencia tecnológica).', 0.1, 'red', None, None),
                # 10
                ('Gastos por intereses/Beneficio neto de la empresa', 'Intereses / Beneficio Neto', 'Gastos por intereses / Beneficio neto', 'Para compañías no financieras, los gastos por intereses deberían ser bajos (<15%).', 0.15, 'red', None, None),
                # 11
                ('Gastos de impuestos/Beneficio neto de la empresa', 'Impuestos / Beneficio Neto', 'Tasa Impositiva Efectiva (Impuestos/Beneficio Neto)', 'La tasa impositiva efectiva debe ser razonable (~35%). Valores bajísimos alertan contabilidad agresiva.', 0.35, 'red', None, None),
                # 12
                ('Beneficio neto de la empresa/Ingresos totales', 'Margen Neto', 'Margen Neto (Beneficio Neto / Ingresos totales)', 'Un Margen Neto estable y alto (ej. > 20%) es señal inequívoca de poder monopólico.', 0.2, 'green', None, None),
                # 13
                ('Inventario', 'Inventario', 'Inventario por Periodo Fiscal', 'El Inventario debería crecer a la par de los Beneficios Netos, indicando gestión de stock eficiente.', None, None, None, None),
                # 14
                ('Total de cuentas por cobrar/Ingresos totales', 'Cuentas por Cobrar / Ingresos', 'Cuentas por Cobrar Netas / Ingresos Totales', 'Ratio constante o decreciente significa que los clientes pagan a tiempo y no se "fuerzan" ventas.', None, None, None, None),
                # 15
                ('Total de activo corriente/Total pasivo corriente', 'Solvencia Corriente', 'Solvencia Corriente (Activo C. / Pasivo C.)', 'Generalmente > 1. Empresas con brutal poder de mercado a veces operan con < 1.', 1.0, 'green', None, None),
                # 16
                ('Capital circulante', 'Prueba Acida', 'Capital Circulante (Prueba Acida)', 'Mide la capacidad de cubrir pasivos a corto plazo sin depender de vender el inventario. > 1 ideal.', 1.0, 'green', None, None),
                # 17
                ('Beneficio neto de la empresa/Activo total', 'ROA (Beneficio / Activo)', 'Rentabilidad de los Activos (ROA)', 'Un ROA alto indica una gran eficiencia en el uso de los activos para generar beneficios.', 0.1, 'green', None, None),
                # 18
                ('Rotacion de existencias', 'Rotacion Existencias', 'Rotacion de Existencias (Ingresos / Inventario)', 'Un valor alto indica que el inventario se vende rápidamente. Buscar consistencia.', None, None, None, None),
                # 19
                ('Prestamos de corto plazo', 'Deuda a Corto Plazo', 'Prestamos de corto plazo (Absoluto)', 'La deuda a corto plazo debe ser baja o nula en industrias no financieras para evitar crisis de liquidez.', None, None, None, None),
                # 20
                ('Porcion corriente de la deuda a largo plazo', 'Porcion Corriente D/L-P', 'Porcion corriente de la deuda a largo plazo', 'Parte de la deuda a largo plazo que vence este mismo año (Debe estar holgadamente cubierta).', None, None, None, None),
                # 21
                ('Unearned Revenue Current', 'Ingresos No Devengados', 'Ingresos no devengados', 'Cobros anticipados por servicios. A mayor cantidad, más financiación "gratuita" de los clientes.', None, None, None, None),
                # 22
                ('Prestamos de corto plazo/Deuda a largo plazo', 'D/C-P / D/L-P', 'Prestamos de corto plazo / Deuda a largo plazo', 'La Deuda a Corto Plazo debe ser mucho menor que la Deuda a Largo Plazo.', 0.5, 'red', None, None),
                # 23
                ('Deuda a largo plazo/Beneficio neto de la empresa', 'Años para pagar D/L-P', 'Años de Beneficio Neto para pagar Deuda L-P', 'Idealmente, toda la Deuda L-P debería poder pagarse con los Beneficios Netos de 3 o 4 años.', 4.0, 'red', None, None),
                # 24
                ('Ratio entre deuda y fondos propios', 'D/E Ratio', 'Ratio entre deuda y fondos propios (D/E)', 'Deuda Total / Patrimonio. Idealmente, debe ser menor a 0.8 en empresas no financieras.', 0.8, 'red', None, None),
                # 25
                ('Reservas', 'Reservas Acumuladas', 'Reservas Acumuladas por Periodo Fiscal', 'Beneficios Netos - Dividendos - Recompras. Debe mostrar una tendencia netamente creciente.', None, None, None, None),
                # 26
                ('Gastos de capital/Beneficios netos', 'CapEx / Beneficios Netos', 'Gastos de capital / Beneficios netos', 'Un ratio consistentemente bajo (<0.25) indica fuerte ventaja competitiva y baja intensidad de capital.', 0.5, 'red', 0.25, 'green'),
                # 27
                ('Promedio ponderado de acciones diluidas en circulacion', 'Acciones en Circulacion', 'Acciones diluidas en circulacion', 'No debe diluir el capital emitiendo acciones; idealmente deben mantenerse o descender.', None, None, None, None),
                # 28
                ('Recompra de acciones comunes', 'Recompra Acciones', 'Recompra de acciones comunes', 'Una recompra constante es positivo porque concentra y aumenta el BPA de los accionistas restantes.', None, None, None, None),
                # 29
                ('Punto de equilibrio', 'Punto de Equilibrio', 'Punto de equilibrio', 'Nivel de ventas necesario para no perder dinero. Mantenerlo estable o decreciente es vital.', None, None, None, None),
                # 30
                ('Margen de seguridad', 'Margen de Seguridad', 'Margen de seguridad (Ratio)', 'Mide cuánto pueden caer las ventas sin entrar en pérdidas. Debe ser lo más alto posible.', None, None, None, None),
                # 31
                ('Efectivo de Operaciones', 'Efectivo Operaciones', 'Efectivo de Operaciones por Periodo', 'El Cash Flow de Operaciones (Sangre de la empresa) debe ser consistentemente creciente.', None, None, None, None),
                # 32
                ('Flujo de caja libre/Deuda total', 'FCF / Deuda Total', 'Flujo de caja libre / Deuda total', 'Un ratio > 1 indica que la empresa genera suficiente flujo libre para saldar toda su deuda en 1 año.', 1.0, 'green', None, None)
            ]

            for col, y_label, title, desc, ref1, col1, ref2, col2 in graficas_a_generar:
                # Comprobación de seguridad para que no falle si la columna está entera vacía por error
                if col in df.columns and not df[col].isna().all() and df[col].sum() != 0:
                    try:
                        fig = plt.figure(figsize=(9, 7))
                        ax = fig.add_subplot(111)
                        barras = ax.bar(df['Periodo Fiscal'], df[col])
                        formatear_grafica(barras, ax)
                        ax.set_xlabel('Periodo Fiscal')
                        ax.set_ylabel(y_label)
                        ax.set_title(f'{title} ({ticker})')
                        plt.text(0.95, 0.95, textwrap.fill(desc, 80), transform=plt.gcf().transFigure, ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
                        if ref1 is not None: ax.axhline(y=ref1, color=col1, linestyle='--', label=f'Ref: {ref1}')
                        if ref2 is not None: ax.axhline(y=ref2, color=col2, linestyle='--', label=f'Ref: {ref2}')
                        if ref1 or ref2: ax.legend(loc='upper left', fontsize=8)
                        estilizar_figura_profesional(fig)
                        pdf.savefig(fig)
                    except Exception as e:
                        logger.warning(f"Error graficando {col}: {e}")
                    finally:
                        plt.close('all')

        writer = PdfWriter()
        reader = PdfReader(BytesIO(content_buffer.getvalue()))
        for page in reader.pages:
            writer.add_page(page)
            
        final_path = output_pdf_path
        try:
            with open(final_path, "wb") as output_file:
                writer.write(output_file)
        except PermissionError:
            timestamp = datetime.now().strftime("%H%M%S")
            base, ext = os.path.splitext(output_pdf_path)
            final_path = f"{base}_{timestamp}{ext}"
            with open(final_path, "wb") as output_file:
                writer.write(output_file)

        return final_path
