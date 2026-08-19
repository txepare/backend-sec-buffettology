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
import matplotlib.patches as patches

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
    while val >= 1000 and magnitude < 4:
        magnitude += 1
        val /= 1000.0
    suffix = ['', 'K', 'M', 'B', 'T'][magnitude]
    signo = "-" if num < 0 else ""
    formateado = f"{val:.2f}".rstrip('0').rstrip('.')
    return f'{signo}{formateado}{suffix}'

def formatear_grafica(barras, ax):
    etiquetas = [formato_legible(val) for val in barras.datavalues]
    ax.bar_label(barras, labels=etiquetas, padding=4, fontsize=9.5, fontweight='bold')
    ax.margins(y=0.22)

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
        ax.tick_params(colors=REPORT_THEME["muted"], labelsize=9.5)
        ax.xaxis.label.set_color(REPORT_THEME["text"])
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_color(REPORT_THEME["text"])
        ax.yaxis.label.set_fontsize(10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: formato_legible(x)))
        for container in ax.containers:
            if not hasattr(container, "patches"): continue
            for barra in container.patches:
                barra.set_facecolor(REPORT_THEME["negative"] if barra.get_height() < 0 else REPORT_THEME["sky"])
                barra.set_edgecolor("white")
                barra.set_linewidth(1)
                barra.set_alpha(0.96)
    try:
        fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.91))
    except Exception:
        pass

def estilizar_tabla(table, header_fontsize: float = 9.5, cell_fontsize: float = 8.5, pad: float = 0.05):
    """
    Aplica una estética limpia y profesional a las tablas de matplotlib acorde al tema corporativo,
    asegurando márgenes internos para que el texto nunca se corte ni se solape.
    """
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(REPORT_THEME["line"])
        cell.set_linewidth(0.9)
        cell.PAD = pad
        if row == 0:
            cell.set_facecolor(REPORT_THEME["navy"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(header_fontsize)
        else:
            cell.set_facecolor(REPORT_THEME["panel"] if row % 2 == 0 else REPORT_THEME["canvas"])
            cell.get_text().set_color(REPORT_THEME["text"])
            cell.get_text().set_fontsize(cell_fontsize)
            if col == 0:
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_ha("left")


def _calcular_cagrs_adaptativos(s: pd.Series):
    """
    Calcula el CAGR a largo plazo (10 años o máximo disponible) y a medio plazo (5 años o mitad disponible).
    Retorna: (cagr_10, cagr_5, años_10, años_5)
    """
    n_points = len(s)
    if n_points < 2:
        return float('nan'), float('nan'), 10, 5

    # 1. Periodo largo: 10 años si hay al menos 11 puntos, o todos los años disponibles (n_points - 1)
    if n_points >= 11:
        n_10 = 10
        start_idx_10 = -11
    else:
        n_10 = n_points - 1
        start_idx_10 = 0

    cagr_10 = float('nan')
    if s.iloc[start_idx_10] > 0 and s.iloc[-1] > 0:
        cagr_10 = ((s.iloc[-1] / s.iloc[start_idx_10]) ** (1.0 / n_10) - 1) * 100

    # 2. Periodo medio: 5 años si hay al menos 6 puntos, o la mitad de los años disponibles
    if n_points >= 6:
        n_5 = 5
        start_idx_5 = -6
    else:
        n_5 = max(1, (n_points - 1) // 2)
        start_idx_5 = -1 - n_5

    cagr_5 = float('nan')
    if s.iloc[start_idx_5] > 0 and s.iloc[-1] > 0:
        cagr_5 = ((s.iloc[-1] / s.iloc[start_idx_5]) ** (1.0 / n_5) - 1) * 100

    return cagr_10, cagr_5, n_10, n_5


def show_percentage_difference(pdf, df, ticker: str = "", current_price: float = None):
    """
    Calcula y renderiza la tabla de panorama de crecimiento y valoración por ROE,
    adaptándose dinámicamente si la empresa dispone de menos de 10 años históricos.
    """
    # ---------- Ordenar por fecha (CRÍTICO) ----------
    if 'fecha' in df.columns:
        df = df.sort_values('fecha')
    elif 'Periodo Fiscal' in df.columns:
        df = df.sort_values('Periodo Fiscal')

    # ---------- Crecimiento % del beneficio neto ----------
    if 'Beneficio neto de la empresa' in df.columns:
        df['Beneficio neto de la empresa Diff (%)'] = (
            df['Beneficio neto de la empresa'].pct_change() * 100
        )
        bn = df['Beneficio neto de la empresa'].dropna().reset_index(drop=True)
    else:
        bn = pd.Series([], dtype=float)

    crecimientoBN10, crecimientoBN5, n_bn_10, n_bn_5 = _calcular_cagrs_adaptativos(bn)

    # ---------- BPA ----------
    shares = df.get('Promedio ponderado de acciones basicas en circulacion',
             df.get('Promedio ponderado de acciones básicas en circulación',
             df.get('Promedio ponderado de acciones diluidas en circulacion',
             df.get('Promedio ponderado de acciones diluidas en circulación',
             df.get('Total de acciones fuera. en la fecha de presentacion',
             pd.Series([1]*len(df)))))))
    shares = shares.replace(0, np.nan)

    if 'Beneficio neto de la empresa' in df.columns:
        df['Beneficio por acción'] = df['Beneficio neto de la empresa'] / shares
        bpa = df['Beneficio por acción'].dropna().reset_index(drop=True)
    else:
        bpa = pd.Series([], dtype=float)

    # ---------- CAGR BPA Adaptativo ----------
    crecimientoBPA10, crecimientoBPA5, n_bpa_10, n_bpa_5 = _calcular_cagrs_adaptativos(bpa)

    # ---------- Resolución de Precio Actual (Self-healing si falló el primer intento) ----------
    if (current_price is None or current_price <= 0) and ticker:
        try:
            from src.tools.market_api import MarketDataAPI
            mq = MarketDataAPI.get_market_quote(ticker)
            current_price = mq.get("current_price", 0.0)
        except Exception as e:
            logger.debug(f"No se pudo resolver cotización en show_percentage_difference: {e}")

    # --------------- PER ------------------------------
    ultimo_bpa = bpa.iloc[-1] if len(bpa) > 0 else 0
    if ultimo_bpa > 0 and current_price is not None and current_price > 0:
        per_actual = current_price / ultimo_bpa
    else:
        per_actual = float('nan')

    # ---------- Valoración por (ROE) ------------------
    cagr_estimado = None
    tasa_retencion = None
    try:
        patrimonio_total = df['Fondos propios totales'].iloc[-1] if 'Fondos propios totales' in df.columns else 0
        acciones_circulacion = shares.iloc[-1] if hasattr(shares, 'iloc') else shares[-1]
        beneficio_neto_total = df['Beneficio neto de la empresa'].iloc[-1] if 'Beneficio neto de la empresa' in df.columns else 0
        
        serie_dividendos = df.get('Dividendos de acciones comunes y preferentes pagados',
                           df.get('Dividendos comunes pagados',
                           df.get('Dividendos preferenciales pagados', pd.Series([0]*len(df)))))
        serie_recompras = df.get('Recompra de acciones comunes', pd.Series([0]*len(df)))

        div_pagados = abs(serie_dividendos.iloc[-1]) if len(serie_dividendos) > 0 else 0
        if pd.isna(div_pagados): div_pagados = 0

        recompras = abs(serie_recompras.iloc[-1]) if len(serie_recompras) > 0 else 0
        if pd.isna(recompras): recompras = 0

        # Cálculos de Valor Actual
        fp_por_accion = patrimonio_total / acciones_circulacion if acciones_circulacion and acciones_circulacion != 0 else 0
        ultimo_bpa_calc = beneficio_neto_total / acciones_circulacion if acciones_circulacion and acciones_circulacion != 0 else 0
        roe_actual = ultimo_bpa_calc / fp_por_accion if fp_por_accion != 0 else (beneficio_neto_total / patrimonio_total if patrimonio_total and patrimonio_total != 0 else 0)

        # Cálculo de Tasa de Retención (Retention Ratio)
        beneficio_retenido_total = beneficio_neto_total - div_pagados - recompras
        if beneficio_retenido_total <= 0:
            recompras_mean = abs(df.get('Recompra de acciones comunes', pd.Series([0])).mean())
            beneficio_retenido_total = beneficio_neto_total - (div_pagados + recompras_mean)
        
        tasa_retencion = beneficio_retenido_total / beneficio_neto_total if beneficio_neto_total != 0 else 0

        # Proyecciones a 10 años (Fórmula de Crecimiento Sostenible: g = ROE * Retención)
        tasa_crecimiento_g = roe_actual * tasa_retencion
        
        fp_proyectado_10_años = fp_por_accion * ((1 + tasa_crecimiento_g) ** 10)
        bpa_proyectado_10_años = fp_proyectado_10_años * roe_actual
        
        # Estimación de Precio Futuro
        precio_objetivo_10_años = bpa_proyectado_10_años * per_actual if not pd.isna(per_actual) else 0

        # Rentabilidad Anualizada (CAGR)
        Beneficio_actual = ultimo_bpa_calc if (ultimo_bpa_calc > 0) else (beneficio_neto_total / acciones_circulacion if acciones_circulacion else 1.0)
        Total_dividendos_acumulados = 0
        años = 10
        for año in range(1, años + 1):
            Beneficio_actual *= (1 + tasa_crecimiento_g)
            Dividendo_del_año = Beneficio_actual * roe_actual
            Total_dividendos_acumulados += Dividendo_del_año

        if current_price and current_price > 0 and precio_objetivo_10_años > 0:
            Beneficio_10años_despues_impuestos = (precio_objetivo_10_años + Total_dividendos_acumulados) 
            if Beneficio_10años_despues_impuestos > 0:
                cagr_estimado = ((Beneficio_10años_despues_impuestos / current_price) ** (1 / 10) - 1) * 100
            else:
                cagr_estimado = float('nan')
        else:
            cagr_estimado = float('nan')

    except ZeroDivisionError:
        cagr_estimado = None
    except Exception as e:
        logger.warning(f"Error en proyección por ROE: {e}")
        cagr_estimado = None

    # ---------- Preparar datos con etiquetas adaptativas ----------
    lbl_bn_10 = 'Crecimiento medio Beneficio Neto 10 AÑOS (%)' if n_bn_10 == 10 else f'Crecimiento medio Beneficio Neto {n_bn_10} AÑOS (%)'
    lbl_bn_5 = 'Crecimiento medio Beneficio Neto 5 AÑOS (%)' if n_bn_5 == 5 else f'Crecimiento medio Beneficio Neto {n_bn_5} AÑOS (%)'
    lbl_bpa_10 = 'CAGR BPA últimos 10 años (%)' if n_bpa_10 == 10 else f'CAGR BPA últimos {n_bpa_10} años (%)'
    lbl_bpa_5 = 'CAGR BPA últimos 5 años (%)' if n_bpa_5 == 5 else f'CAGR BPA últimos {n_bpa_5} años (%)'

    metricas_raw = [
        lbl_bn_10,
        lbl_bn_5,
        lbl_bpa_10,
        lbl_bpa_5,
        'CAGR ESTIMADO POR ROE (%)',
        'TASA DE RETENCIÓN (%)',
        'PER ACTUAL'
    ]
    valores_raw = [
        f'{crecimientoBN10:.2f}%' if not pd.isna(crecimientoBN10) else 'N/A',
        f'{crecimientoBN5:.2f}%' if not pd.isna(crecimientoBN5) else 'N/A',
        f'{crecimientoBPA10:.2f}%' if not pd.isna(crecimientoBPA10) else 'N/A',
        f'{crecimientoBPA5:.2f}%' if not pd.isna(crecimientoBPA5) else 'N/A',
        f'{cagr_estimado:.2f}%' if (cagr_estimado is not None and not pd.isna(cagr_estimado)) else 'N/A',
        f'{(tasa_retencion * 100):.2f}%' if (tasa_retencion is not None and not pd.isna(tasa_retencion)) else 'N/A',
        f'{per_actual:.2f}' if not pd.isna(per_actual) else 'N/A'
    ]

    cell_data = []
    for m, v in zip(metricas_raw, valores_raw):
        cell_data.append([textwrap.fill(m, width=42), str(v)])

    # ---------- Renderizar figura dentro de márgenes estrictos ----------
    fig = plt.figure(figsize=(9, 7), dpi=150)
    fig.patch.set_facecolor(REPORT_THEME["canvas"])
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
    ax.set_facecolor(REPORT_THEME["canvas"])
    ax.axis('off')
    
    ticker_str = ticker or ""
    ax.text(
        0.0, 0.985,
        f'Panorama de crecimiento y valoracion ({ticker_str})',
        fontsize=13.5,
        fontweight='bold',
        color=REPORT_THEME["navy"],
        ha='left',
        va='top'
    )

    sub_text = textwrap.fill(
        "Comparar el CAGR del Beneficio Neto con el CAGR del BPA para comprobar si la empresa crece por su propio motor o por las recompras de acciones.",
        width=85
    )
    ax.text(
        0.0, 0.935,
        sub_text,
        fontsize=9.0,
        ha='left',
        va='top',
        color=REPORT_THEME["muted"]
    )

    table = ax.table(
        cellText=cell_data,
        colLabels=['Métrica', 'Valor'],
        colWidths=[0.68, 0.32],
        bbox=[0.0, 0.05, 1.0, 0.82],
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    estilizar_tabla(table, header_fontsize=10.0, cell_fontsize=9.0, pad=0.06)

    pdf.savefig(fig)
    plt.close(fig)


def show_capex_vs_netincome(pdf, df, ticker: str = ""):
    """
    Calcula y renderiza el análisis de CapEx vs Beneficio Neto con diagnóstico visual,
    garantizando que los textos no se solapen ni salgan de los márgenes de la página.
    """
    total_net_income = df['Beneficio neto de la empresa'].sum() if 'Beneficio neto de la empresa' in df.columns else 0
    total_capital_expenditures = df['Gastos de capital'].sum() if 'Gastos de capital' in df.columns else 0

    if total_net_income != 0:
        cociente_total = round((abs(total_capital_expenditures) / total_net_income) * 100, 3)
        cociente_str = f'{cociente_total:.3f}%'
        color_mensaje = REPORT_THEME["positive"] if cociente_total < 50 else REPORT_THEME["negative"]
    else:
        cociente_total = None
        cociente_str = "N/A (Beneficio Neto Total = 0)"
        color_mensaje = REPORT_THEME["negative"]

    metric_label = textwrap.fill("Gasto de Capital Total / Beneficio Neto Total (%)", width=46)
    cell_data = [[metric_label, cociente_str]]

    fig = plt.figure(figsize=(9, 7), dpi=150)
    fig.patch.set_facecolor(REPORT_THEME["canvas"])
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
    ax.set_facecolor(REPORT_THEME["canvas"])
    ax.axis('off')

    ticker_str = ticker or ""
    ax.text(
        0.0, 0.985,
        f'Análisis de Inversión ({ticker_str}): Gastos de Capital vs. Beneficio Neto',
        fontsize=13.0,
        fontweight='bold',
        color=REPORT_THEME["navy"],
        ha='left',
        va='top'
    )

    mensaje_regla = textwrap.fill(
        "Regla de la Ventaja Competitiva Duradera (Buffett/Graham): Una empresa con una ventaja competitiva fuerte no debería necesitar reinvertir más del 50% de sus Beneficios Netos en Gastos de Capital para mantener sus operaciones actuales.",
        width=82
    )
    ax.text(
        0.5, 0.88,
        mensaje_regla,
        fontsize=9.2,
        ha='center',
        va='top',
        color=REPORT_THEME["text"],
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor=REPORT_THEME["panel"],
            edgecolor=REPORT_THEME["line"],
            linewidth=1.0
        )
    )

    table = ax.table(
        cellText=cell_data,
        colLabels=['Métrica', 'Valor Calculado'],
        colWidths=[0.68, 0.32],
        bbox=[0.05, 0.46, 0.90, 0.18],
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    estilizar_tabla(table, header_fontsize=10.0, cell_fontsize=9.6, pad=0.06)

    if cociente_total is not None and cociente_total < 50:
        diag_text = f"CUMPLE LA REGLA: {cociente_str} (< 50%)\nLa empresa retiene la mayor parte de sus beneficios sin requerir un CapEx intensivo."
    elif cociente_total is not None:
        diag_text = f"NO CUMPLE LA REGLA: {cociente_str} (>= 50%)\nLa empresa requiere una alta reinversión de capital en maquinaria o infraestructura."
    else:
        diag_text = f"Resultado del Análisis: {cociente_str}"

    color_mensaje = REPORT_THEME['positive'] if (cociente_total is not None and cociente_total < 50) else (REPORT_THEME['negative'] if cociente_total is not None else REPORT_THEME['navy'])
    ax.text(
        0.5, 0.24,
        diag_text,
        fontsize=10.5,
        fontweight='bold',
        ha='center',
        va='center',
        color=color_mensaje,
        bbox=dict(
            boxstyle='round,pad=0.7',
            facecolor=REPORT_THEME["panel"],
            edgecolor=color_mensaje,
            linewidth=1.4
        )
    )

    pdf.savefig(fig)
    plt.close(fig)


def _render_dynamic_executive_cards_page(
    pdf,
    main_title: str,
    subtitle: str,
    verdict_text: str,
    badge_color: str,
    cards: list,
    ticker: str = ""
):
    """
    Motor de maquetación dinámico para páginas ejecutivas con tarjetas de texto.
    Aprovecha el 100% del lienzo vertical de la página distribuyendo las tarjetas
    de forma proporcional, con tipografías legibles, fondos elegantes y paneles pulidos.
    """
    fig = plt.figure(figsize=(9.5, 7.2), dpi=150)
    fig.patch.set_facecolor(REPORT_THEME['canvas'])
    ax = fig.add_axes([0.045, 0.035, 0.91, 0.93])
    ax.set_facecolor(REPORT_THEME['canvas'])
    ax.axis('off')

    def _esc(t): return str(t or "").replace('$', r'\$')

    # 1. Encabezado institucional
    ax.text(0.0, 0.985, _esc(main_title), fontsize=13.0, fontweight='bold', color=REPORT_THEME['navy'], va='top')
    ax.text(0.0, 0.948, _esc(subtitle), fontsize=8.8, style='italic', color=REPORT_THEME['muted'], va='top')

    # 2. Veredicto destacado
    ax.text(0.5, 0.885, _esc(verdict_text), fontsize=9.2, fontweight='bold', ha='center', va='center', color=badge_color,
            bbox=dict(boxstyle='round,pad=0.42', facecolor=REPORT_THEME['panel'], edgecolor=badge_color, linewidth=1.3))

    # 3. Medición y formateo dinámico de tarjetas
    wrap_w = 104
    wrapped_cards = []
    
    for title, body, is_hl in cards:
        lines = []
        for p in str(body or "").split('\n\n'):
            if p.strip():
                lines.extend(textwrap.fill(_esc(p.strip()), width=wrap_w).split('\n'))
        if not lines:
            lines = ["Sin información adicional reportada."]
        wrapped_cards.append((title, '\n'.join(lines), len(lines), is_hl))

    n = len(wrapped_cards)
    if n == 0:
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Área vertical disponible para las tarjetas: desde y_top hasta y_bottom
    y_top = 0.815
    y_bottom = 0.035
    avail_h = y_top - y_bottom
    gap = 0.022
    total_gaps = gap * (n - 1)
    usable_h = avail_h - total_gaps

    weights = [max(c[2], 1) for c in wrapped_cards]
    total_w = sum(weights)
    card_heights = [(w / total_w) * usable_h for w in weights]
    
    # Asegurar altura mínima para cada tarjeta
    min_h = 0.115
    for i in range(n):
        card_heights[i] = max(card_heights[i], min_h)
    
    # Re-normalizar para ocupar exactamente el 100% del espacio útil
    scale_factor = usable_h / sum(card_heights)
    card_heights = [h * scale_factor for h in card_heights]

    total_lines = sum(c[2] for c in wrapped_cards)
    if total_lines >= 26:
        font_s = 7.4
    elif total_lines >= 18:
        font_s = 7.8
    elif total_lines >= 12:
        font_s = 8.2
    else:
        font_s = 8.6

    curr_y = y_top
    for idx, (c_title, c_body, n_lines, is_hl) in enumerate(wrapped_cards):
        card_h = card_heights[idx]
        box_y = curr_y - card_h

        # Fondo y borde de tarjeta completa
        bg_col = '#F0FDFA' if is_hl else REPORT_THEME['panel']
        edge_col = REPORT_THEME['teal'] if is_hl else REPORT_THEME['line']
        lw = 1.3 if is_hl else 0.8
        
        rect = patches.FancyBboxPatch(
            (0.0, box_y), 1.0, card_h,
            boxstyle='round,pad=0.0,rounding_size=0.015',
            facecolor=bg_col, edgecolor=edge_col, linewidth=lw,
            transform=ax.transAxes, zorder=2
        )
        ax.add_patch(rect)

        # Título de tarjeta dentro del panel
        title_color = '#0F766E' if is_hl else REPORT_THEME['navy']
        ax.text(0.018, curr_y - 0.022, _esc(c_title), fontsize=font_s + 0.8, fontweight='bold',
                color=title_color, va='top', transform=ax.transAxes, zorder=3)

        # Cuerpo de texto dentro del panel
        body_color = REPORT_THEME['navy'] if is_hl else REPORT_THEME['text']
        body_weight = 'bold' if is_hl else 'normal'
        ax.text(0.018, curr_y - 0.052, c_body, fontsize=font_s, fontweight=body_weight,
                color=body_color, va='top', transform=ax.transAxes, zorder=3)

        curr_y -= (card_h + gap)

    pdf.savefig(fig)
    plt.close(fig)


def show_company_overview(pdf, overview_result: dict, ticker: str = "", market_data: dict = None):
    """
    Renderiza la primera página del informe PDF aprovechando el 100% del lienzo,
    con tarjetas ejecutivas bien proporcionadas, explicando qué hace la empresa, sus divisiones de negocio,
    ubicación, dinámicas sectoriales y evaluación en el Círculo de Competencia de Warren Buffett a partir del 10-K.
    """
    if not overview_result:
        return

    market_data = market_data or {}
    company_name = market_data.get("company_name", ticker)
    sector = market_data.get("sector", "Desconocido")
    industry = market_data.get("industry", "Desconocida")
    current_price = float(market_data.get("current_price", 0.0) or 0.0)
    market_cap = float(market_data.get("market_cap", 0) or 0)

    if market_cap <= 0 and current_price > 0:
        shares = float(market_data.get("shares_outstanding", 0) or 0)
        if shares > 0:
            market_cap = current_price * shares
            market_data["market_cap"] = int(market_cap)

    fig = plt.figure(figsize=(9.5, 7.2), dpi=150)
    fig.patch.set_facecolor(REPORT_THEME['canvas'])
    ax = fig.add_axes([0.045, 0.035, 0.91, 0.93])
    ax.set_facecolor(REPORT_THEME['canvas'])
    ax.axis('off')

    def _esc(t):
        if not t: return ""
        return str(t).replace('$', r'\$')

    ticker_str = ticker or ""
    # 1. Encabezado institucional de la Empresa
    ax.text(0.0, 0.985, f'{company_name} ({ticker_str}) - Perfil & Modelo de Negocio (SEC Form 10-K)', fontsize=12.8, fontweight='bold', color=REPORT_THEME['navy'], va='top')
    
    # Subtítulo con datos clave de mercado
    if market_cap > 0:
        mcap_formatted = formato_legible(market_cap)
        mcap_str = f"${mcap_formatted}" if not mcap_formatted.startswith("$") else mcap_formatted
    else:
        mcap_str = "N/A"

    price_str = f"${current_price:.2f}" if current_price > 0 else "N/A"
    sub_info = f"Sector: {sector}  |  Industria: {industry}  |  Cotización: {price_str}  |  Market Cap: {mcap_str}"
    ax.text(0.0, 0.950, _esc(sub_info), fontsize=8.6, color=REPORT_THEME['muted'], va='top')

    # 2. Veredicto destacado: Círculo de Competencia y Predictibilidad
    categoria = overview_result.get('categoria_comprensibilidad', 'ALTO')
    badge_color = REPORT_THEME['positive'] if categoria == 'ALTO' else (REPORT_THEME['teal'] if categoria == 'MODERADO' else REPORT_THEME['negative'])
    verdicto = overview_result.get('veredicto_comprensibilidad', 'Evaluación de Comprensibilidad')
    desc_corta = overview_result.get('descripcion_corta', '')
    verdict_text = f"CÍRCULO DE COMPETENCIA DE BUFFETT: {_esc(verdicto)}\n{_esc(desc_corta)}"
    
    ax.text(0.5, 0.885, verdict_text, fontsize=9.0, fontweight='bold', ha='center', va='center', color=badge_color,
            bbox=dict(boxstyle='round,pad=0.42', facecolor=REPORT_THEME['panel'], edgecolor=badge_color, linewidth=1.3))

    # Formateo estructurado de líneas de negocio
    lineas_raw = overview_result.get('lineas_de_negocio', [])
    lines_formatted_list = []
    if isinstance(lineas_raw, list) and lineas_raw:
        for item in lineas_raw:
            if isinstance(item, dict):
                nom = item.get('nombre', '').strip()
                desc = item.get('descripcion', '').strip()
                if nom and desc:
                    lines_formatted_list.append(f"• {nom}: {desc}")
                elif nom or desc:
                    lines_formatted_list.append(f"• {nom or desc}")
            elif isinstance(item, str) and item.strip():
                lines_formatted_list.append(f"• {item.strip()}")
    
    lineas_texto = "\n\n".join(lines_formatted_list) if lines_formatted_list else overview_result.get('modelo_ingresos', 'Desglose detallado disponible en informe 10-K.')

    # Ubicación y actividad combinada
    resumen_act = overview_result.get('resumen_actividad', 'Sin datos descriptivos.')
    ubicacion_texto = overview_result.get('ubicacion_y_mercados', overview_result.get('mercado_y_clientes', ''))
    if ubicacion_texto:
        card1_body = f"{resumen_act}\n\n• Sede y Alcance Geográfico: {ubicacion_texto}"
    else:
        card1_body = resumen_act

    # Vientos de cola y perspectivas
    tailwinds_texto = overview_result.get('vientos_de_cola_y_sector', 'Impulsores de demanda según dinámica sectorial.')
    crecimiento_texto = overview_result.get('perspectivas_crecimiento', 'Análisis de evolución operativa y expansión de cuota.')
    card3_body = f"• Vientos de Cola del Sector (Tailwinds): {tailwinds_texto}\n\n• Perspectivas de Crecimiento / Riesgos: {crecimiento_texto}"

    # Propuesta de valor y dictamen de Buffett
    propuesta_val = overview_result.get('propuesta_valor', '')
    circulo_val = overview_result.get('circulo_competencia', '')
    card4_body = f"• Propuesta de Valor Diferencial: {propuesta_val}\n\n• Dictamen de Warren Buffett: {circulo_val}"

    # 4 Tarjetas que cubren el 100% del lienzo vertical
    cards = [
        (
            '1. Actividad de la Empresa y a qué se dedica en la economía real (SEC Form 10-K):',
            card1_body,
            False
        ),
        (
            '2. Líneas de Negocio y Segmentos Operativos Detallados:',
            lineas_texto,
            False
        ),
        (
            '3. Vientos de Cola del Sector y Perspectivas de Crecimiento / Decrecimiento:',
            card3_body,
            False
        ),
        (
            '4. Propuesta de Valor Diferencial y Dictamen de Warren Buffett:',
            card4_body,
            True
        )
    ]

    wrap_w = 104
    wrapped_cards = []
    for title, body, is_hl in cards:
        lines = []
        for p in str(body or "").split('\n\n'):
            if p.strip():
                lines.extend(textwrap.fill(_esc(p.strip()), width=wrap_w).split('\n'))
        if not lines:
            lines = ["Sin información adicional reportada."]
        wrapped_cards.append((title, '\n'.join(lines), len(lines), is_hl))

    n = len(wrapped_cards)
    y_top = 0.815
    y_bottom = 0.035
    avail_h = y_top - y_bottom
    gap = 0.022
    usable_h = avail_h - gap * (n - 1)

    weights = [max(c[2], 1) for c in wrapped_cards]
    total_w = sum(weights)
    card_heights = [(w / total_w) * usable_h for w in weights]
    for i in range(n):
        card_heights[i] = max(card_heights[i], 0.115)
    scale_factor = usable_h / sum(card_heights)
    card_heights = [h * scale_factor for h in card_heights]

    total_lines = sum(c[2] for c in wrapped_cards)
    font_s = 7.5 if total_lines >= 26 else (7.9 if total_lines >= 18 else 8.3)

    curr_y = y_top
    for idx, (title, body, n_lines, is_hl) in enumerate(wrapped_cards):
        h = card_heights[idx]
        box_y = curr_y - h

        bg_col = '#F0FDFA' if is_hl else REPORT_THEME['panel']
        edge_col = REPORT_THEME['teal'] if is_hl else REPORT_THEME['line']
        rect = patches.FancyBboxPatch(
            (0.0, box_y), 1.0, h,
            boxstyle='round,pad=0.0,rounding_size=0.015',
            facecolor=bg_col, edgecolor=edge_col, linewidth=1.3 if is_hl else 0.8,
            transform=ax.transAxes, zorder=2
        )
        ax.add_patch(rect)

        title_col = '#0F766E' if is_hl else REPORT_THEME['navy']
        ax.text(0.018, curr_y - 0.022, _esc(title), fontsize=font_s + 0.8, fontweight='bold', color=title_col, va='top', transform=ax.transAxes, zorder=3)

        body_col = REPORT_THEME['navy'] if is_hl else REPORT_THEME['text']
        ax.text(0.018, curr_y - 0.052, body, fontsize=font_s, fontweight='bold' if is_hl else 'normal', color=body_col, va='top', transform=ax.transAxes, zorder=3)

        curr_y -= (h + gap)

    pdf.savefig(fig)
    plt.close(fig)


def show_revenue_segments_table(pdf, segments_result: dict, ticker: str = "", market_data: dict = None):
    """
    Renderiza la Página 2 del documento PDF: Tabla histórica de fuentes de ingresos por años (hasta 5 años),
    pesos porcentuales, crecimiento interanual, descripción de líneas de negocio y análisis de diversificación.
    Aprovecha el 100% del lienzo con paneles amplios y sin huecos en blanco.
    """
    if not segments_result:
        return

    market_data = market_data or {}
    company_name = market_data.get("company_name", ticker)

    fig = plt.figure(figsize=(9.5, 7.2), dpi=150)
    fig.patch.set_facecolor(REPORT_THEME['canvas'])
    ax = fig.add_axes([0.045, 0.035, 0.91, 0.93])
    ax.set_facecolor(REPORT_THEME['canvas'])
    ax.axis('off')

    def _esc(t): return str(t or "").replace('$', r'\$')

    ticker_str = ticker or ""
    # 1. Encabezado institucional
    ax.text(0.0, 0.985, f'{company_name} ({ticker_str}) - Desglose de Fuentes de Ingresos', fontsize=13.0, fontweight='bold', color=REPORT_THEME['navy'], va='top')
    unit_str = segments_result.get("unidad_monetaria", "Billion USD")
    ax.text(0.0, 0.950, f'Evolución histórica por líneas de negocio ({unit_str}) y grado de diversificación (SEC Form 10-K)', fontsize=8.6, style='italic', color=REPORT_THEME['muted'], va='top')

    # 2. Construir datos de la tabla
    years = segments_result.get("years", segments_result.get("años", []))
    historico = segments_result.get("historico_segmentos", {})
    seg_meta = {s.get("nombre"): s for s in segments_result.get("segmentos", [])}

    col_labels = ["Línea de Negocio / Segmento"] + [f"FY {y}" for y in years] + ["% Total", "Crec. YoY"]
    
    table_data = []
    totales_por_ano = [0.0] * len(years)

    for seg_name, vals in historico.items():
        row = [textwrap.fill(seg_name, width=32)]
        for idx in range(len(years)):
            v = vals[idx] if (isinstance(vals, list) and idx < len(vals)) else None
            if v is None or v == "-" or v == "N/A":
                row.append("-")
            elif isinstance(v, (int, float)):
                if v == 0.0 and idx < len(years) - 2:
                    row.append("-")
                else:
                    row.append(f"{v:.1f}")
                    if idx < len(totales_por_ano):
                        totales_por_ano[idx] += float(v)
            else:
                row.append(str(v))
        
        meta = seg_meta.get(seg_name, {})
        pct_val = meta.get("porcentaje_ultimo_ano")
        pct_str = f"{pct_val:.1f}%" if pct_val is not None else "N/A"
        row.append(pct_str)

        yoy_val = meta.get("crecimiento_yoy_pct")
        if yoy_val is not None and isinstance(yoy_val, (int, float)):
            yoy_str = f"{yoy_val:+.1f}%"
        else:
            yoy_str = str(yoy_val or "N/A")
        row.append(yoy_str)
        table_data.append(row)

    if any(totales_por_ano):
        total_row = ["TOTAL CONSOLIDADO"] + [f"{t:.1f}" if t > 0 else "-" for t in totales_por_ano] + ["100.0%", ""]
        table_data.append(total_row)

    n_cols = len(col_labels)
    w_first = 0.36 if len(years) >= 5 else 0.40
    w_other = (1.0 - w_first) / max(n_cols - 1, 1)
    col_widths = [w_first] + [w_other] * (n_cols - 1)

    num_rows = len(table_data) + 1
    row_h = 0.038
    table_h = min(0.32, num_rows * row_h)
    table_y = 0.920 - table_h

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        bbox=[0.0, table_y, 1.0, table_h],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    estilizar_tabla(table, header_fontsize=8.6 if len(years) >= 5 else 9.0, cell_fontsize=8.0 if len(years) >= 5 else 8.4, pad=0.05)

    # 3. Formatear y medir Cards inferiores ocupando el 100% del espacio restante
    desc_lines = []
    for s in segments_result.get("segmentos", []):
        s_name = s.get("nombre", "")
        s_desc = s.get("descripcion", "")
        formatted_bullet = textwrap.fill(f"• {s_name}: {s_desc}", width=104, subsequent_indent="  ")
        desc_lines.append(formatted_bullet)
    
    desc_full_text = "\n".join(desc_lines)
    analisis_div = segments_result.get("analisis_diversificacion", "La empresa cuenta con un modelo diversificado de ingresos.")
    div_wrapped = textwrap.fill(_esc(analisis_div), width=104)

    avail_y_top = table_y - 0.030
    y_bottom = 0.035
    usable_h = avail_y_top - y_bottom
    gap = 0.022
    cards_usable_h = usable_h - gap

    # Proporción: 58% Descripción, 42% Análisis de Diversificación
    card1_h = cards_usable_h * 0.58
    card2_h = cards_usable_h * 0.42

    # Panel Card 1: Descripción
    box1_y = avail_y_top - card1_h
    rect1 = patches.FancyBboxPatch(
        (0.0, box1_y), 1.0, card1_h,
        boxstyle='round,pad=0.0,rounding_size=0.015',
        facecolor=REPORT_THEME['panel'], edgecolor=REPORT_THEME['line'], linewidth=0.8,
        transform=ax.transAxes, zorder=2
    )
    ax.add_patch(rect1)
    ax.text(0.018, avail_y_top - 0.022, '1. Descripción de las Líneas de Negocio y Segmentos:',
            fontsize=8.8, fontweight='bold', color=REPORT_THEME['navy'], va='top', transform=ax.transAxes, zorder=3)
    ax.text(0.018, avail_y_top - 0.052, _esc(desc_full_text),
            fontsize=7.9, color=REPORT_THEME['text'], va='top', transform=ax.transAxes, zorder=3)

    # Panel Card 2: Diversificación
    curr_y_2 = box1_y - gap
    box2_y = curr_y_2 - card2_h
    rect2 = patches.FancyBboxPatch(
        (0.0, box2_y), 1.0, card2_h,
        boxstyle='round,pad=0.0,rounding_size=0.015',
        facecolor='#F0FDFA', edgecolor=REPORT_THEME['teal'], linewidth=1.3,
        transform=ax.transAxes, zorder=2
    )
    ax.add_patch(rect2)
    ax.text(0.018, curr_y_2 - 0.022, '2. Análisis de Diversificación y Riesgo de Concentración de Ingresos:',
            fontsize=8.8, fontweight='bold', color='#0F766E', va='top', transform=ax.transAxes, zorder=3)
    ax.text(0.018, curr_y_2 - 0.052, div_wrapped,
            fontsize=8.2, fontweight='bold', color=REPORT_THEME['navy'], va='top', transform=ax.transAxes, zorder=3)

    pdf.savefig(fig)
    plt.close(fig)


def show_income_statement_flow(pdf, flow_data: dict, ticker: str = "", market_data: dict = None):
    """
    Renderiza la Página 3 del documento PDF: Diagrama de flujo Sankey del estado de resultados del último año fiscal.
    """
    if not flow_data:
        return

    try:
        from src.tools.sankey_builder import SankeyFlowBuilder
        company_name = (market_data or {}).get("company_name", ticker)
        year = flow_data.get("year", "2024")
        segments = flow_data.get("segments_data", [])
        financial_flow = flow_data.get("financial_flow", {})
        
        fig = SankeyFlowBuilder.generate_sankey_figure(
            ticker=ticker,
            company_name=company_name,
            year=year,
            segments_data=segments,
            financial_flow=financial_flow
        )
        pdf.savefig(fig)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error generando diagrama Sankey de estado de resultados: {e}", exc_info=True)


def show_monopoly_analysis(pdf, analysis_result: dict, ticker: str = ""):
    """
    Renderiza la página ejecutiva final de evaluación de monopolio de consumo según Warren Buffett (Pregunta 1).
    """
    if not analysis_result:
        return

    ticker_str = ticker or ""
    categoria = analysis_result.get('categoria', 'MODERADO')
    badge_color = REPORT_THEME['positive'] if categoria == 'FUERTE' else (REPORT_THEME['negative'] if categoria == 'COMMODITY' else REPORT_THEME['gold'])
    verdicto_corto = analysis_result.get('veredicto_corto', 'Evaluación de Monopolio')
    tipo_foso = analysis_result.get('tipo_foso', 'Ventaja Competitiva')
    verdict_text = f"VEREDICTO: {verdicto_corto}\nTipo de Foso: {tipo_foso}"

    sec_desc = analysis_result.get('analisis_sec', 'Sin datos descriptivos.')
    pricing_power = analysis_result.get('poder_fijacion_precios', '')
    pilares = analysis_result.get('pilares_cuantitativos', '')
    quant_text = f"• Fijación de Precios: {pricing_power}\n• Métricas Contables: {pilares}"
    threats = analysis_result.get('amenazas_foso', 'Sin amenazas críticas reportadas.')
    conclusion = analysis_result.get('conclusion_buffett', '')

    cards = [
        ('1. Análisis del Modelo de Negocio e Informes 10-K (SEC):', sec_desc, False),
        ('2. Poder de Fijación de Precios & Evidencia Cuantitativa (10 Años):', quant_text, False),
        ('3. Amenazas al Foso Defensivo & Riesgos Regulatorios:', threats, False),
        ('4. Conclusión Final de Warren Buffett:', conclusion, True)
    ]

    _render_dynamic_executive_cards_page(
        pdf=pdf,
        main_title=f'Evaluación de Monopolio de Buffettology ({ticker_str})',
        subtitle='Pregunta 1 de Warren Buffett: ¿Tiene la empresa un monopolio fácilmente identificable?',
        verdict_text=verdict_text,
        badge_color=badge_color,
        cards=cards,
        ticker=ticker
    )


def show_retained_earnings_analysis(pdf, analysis_result: dict, ticker: str = ""):
    """
    Renderiza la página ejecutiva de evaluación de beneficios no distribuidos según Warren Buffett (Test del $1 y Asignación de Capital).
    """
    if not analysis_result:
        return

    ticker_str = ticker or ""
    categoria = analysis_result.get('categoria', 'MODERADO')
    badge_color = REPORT_THEME['positive'] if categoria == 'EXCELENTE' else (REPORT_THEME['negative'] if categoria == 'DEFICIENTE' else REPORT_THEME['gold'])
    verdicto_corto = analysis_result.get('veredicto_corto', 'Evaluación de Asignación de Capital')
    eficiencia = analysis_result.get('eficiencia_capital', 'Asignación de Capital')
    verdict_text = f"VEREDICTO: {verdicto_corto}\nEficiencia: {eficiencia}"

    sec_desc = analysis_result.get('analisis_sec_reinversion', 'Sin datos descriptivos.')
    metrica = analysis_result.get('metrica_dolar_retenido', '')
    politica = analysis_result.get('politica_retorno_accionista', 'Sin política específica reportada.')
    conclusion = analysis_result.get('conclusion_buffett', '')

    cards = [
        ('1. Estrategia de Reinversión según Informes 10-K (SEC):', sec_desc, False),
        ('2. Test del Dólar Retenido & Evolución del BPA (10 Años):', metrica, False),
        ('3. Política de Retorno al Accionista (Recompras & Dividendos):', politica, False),
        ('4. Dictamen Final de Warren Buffett sobre Asignación de Capital:', conclusion, True)
    ]

    _render_dynamic_executive_cards_page(
        pdf=pdf,
        main_title=f'Evaluación de Beneficios No Distribuidos ({ticker_str})',
        subtitle='Pregunta de Warren Buffett: ¿Incrementará el valor añadido de los beneficios no distribuidos el valor de mercado?',
        verdict_text=verdict_text,
        badge_color=badge_color,
        cards=cards,
        ticker=ticker
    )


def show_management_alignment_analysis(pdf, analysis_result: dict, ticker: str = ""):
    """
    Renderiza la página ejecutiva final de evaluación de alineación de directivos con los accionistas según Warren Buffett.
    """
    if not analysis_result:
        return

    ticker_str = ticker or ""
    categoria = analysis_result.get('categoria', 'MODERADA')
    badge_color = REPORT_THEME['positive'] if categoria == 'EXCELENTE' else (REPORT_THEME['teal'] if categoria == 'BUENA' else (REPORT_THEME['negative'] if categoria == 'DEFICIENTE' else REPORT_THEME['gold']))
    verdicto_corto = analysis_result.get('veredicto_corto', 'Evaluación de Alineación')
    nivel_alineacion = analysis_result.get('nivel_alineacion', 'Alineación de Intereses')
    verdict_text = f"VEREDICTO: {verdicto_corto}\nNivel: {nivel_alineacion}"

    analogia = analysis_result.get('analogia_sencilla', '')
    explicacion = analysis_result.get('explicacion_facil', '')
    card1_text = f"• Analogía: {analogia}\n\n• ¿Cómo cuidan tu dinero?: {explicacion}"
    sec_evidencia = analysis_result.get('evidencia_sec_remuneracion', '')
    positivos = analysis_result.get('puntos_positivos', '')
    alertas = analysis_result.get('alertas_accionista', '')
    puntos_text = f"Puntos a favor:\n{positivos}\n\nAlertas a vigilar:\n{alertas}"
    conclusion = analysis_result.get('conclusion_buffett', '')

    cards = [
        ('1. En palabras sencillas (Explicación para todos los públicos):', card1_text, False),
        ('2. Evidencia en los Informes de la SEC (Recompras y Piel en el Juego):', sec_evidencia, False),
        ('3. Puntos Clave a Favor y Alertas para el Inversor:', puntos_text, False),
        ('4. Dictamen Final de Warren Buffett sobre el Equipo Directivo:', conclusion, True)
    ]

    _render_dynamic_executive_cards_page(
        pdf=pdf,
        main_title=f'Evaluación de Alineación Directiva con los Accionistas ({ticker_str})',
        subtitle='Pregunta de Warren Buffett: ¿Cómo es la alineación de los directivos con los intereses de los accionistas?',
        verdict_text=verdict_text,
        badge_color=badge_color,
        cards=cards,
        ticker=ticker
    )


def show_accounting_forensic_analysis(pdf, analysis_result: dict, ticker: str = ""):
    """
    Renderiza la página ejecutiva de auditoría forense y calidad contable según Warren Buffett.
    """
    if not analysis_result:
        return

    ticker_str = ticker or ""
    categoria = analysis_result.get('categoria', 'LIMPIA')
    badge_color = REPORT_THEME['positive'] if categoria == 'LIMPIA' else (REPORT_THEME['negative'] if categoria == 'CRITICA' else REPORT_THEME['gold'])
    verdicto_corto = analysis_result.get('veredicto_corto', 'Auditoría Forense')
    calidad = analysis_result.get('calidad_beneficios', 'Calidad Contable')
    verdict_text = f"VEREDICTO: {verdicto_corto}\nCalidad de Beneficios: {calidad}"

    caja_desc = analysis_result.get('coherencia_caja_vs_beneficio', '')
    cobros_desc = analysis_result.get('analisis_cobros_inventarios', '')
    ajustes_desc = analysis_result.get('analisis_ajustes_y_notas', '')
    señales = analysis_result.get('señales_alerta', '')
    card3_text = f"{ajustes_desc}\n\nDiagnóstico de Alertas:\n{señales}"
    conclusion = analysis_result.get('conclusion_buffett', '')

    cards = [
        ('1. Calidad de los Beneficios (Caja Real vs Beneficio Contable):', caja_desc, False),
        ('2. Análisis de Cuentas por Cobrar, Inventarios y Capital Circulante:', cobros_desc, False),
        ('3. Ajustes No-GAAP, Notas a los Informes 10-K y Señales de Alerta:', card3_text, False),
        ('4. Dictamen Final de Warren Buffett sobre Integridad Contable:', conclusion, True)
    ]

    _render_dynamic_executive_cards_page(
        pdf=pdf,
        main_title=f'Auditoría Forense y Calidad Contable ({ticker_str})',
        subtitle='Pregunta de Warren Buffett: ¿Hay indicios de contabilidad engañosa o datos que no cuadren?',
        verdict_text=verdict_text,
        badge_color=badge_color,
        cards=cards,
        ticker=ticker
    )


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
        divs = np.abs(df.get('Dividendos de acciones comunes y preferentes pagados', df.get('Dividendos comunes pagados', 0)))
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
    def generate_pdf_report(
        ticker: str,
        current_price: float,
        df_financials: pd.DataFrame,
        output_pdf_path: str,
        sector_config: dict,
        company_overview: dict = None,
        segments_data: dict = None,
        income_flow_data: dict = None,
        monopoly_analysis: dict = None,
        retained_earnings_analysis: dict = None,
        management_analysis: dict = None,
        forensic_analysis: dict = None,
        market_data: dict = None
    ) -> str:
        logger.info(f"[PDF Builder] Generando informe completo de Buffettology para {ticker}...")
        
        df = PDFBuilder._calcular_metricas_derivadas(df_financials)
        
        final_path = output_pdf_path
        try:
            with open(final_path, "ab"): pass
        except PermissionError:
            timestamp = datetime.now().strftime("%H%M%S")
            base, ext = os.path.splitext(output_pdf_path)
            final_path = f"{base}_{timestamp}{ext}"

        market_data = market_data or {}
        if market_data:
            mcap = float(market_data.get("market_cap", 0) or 0)
            if mcap <= 0 and current_price > 0:
                if 'Promedio ponderado de acciones diluidas en circulacion' in df.columns:
                    valid_sh = df['Promedio ponderado de acciones diluidas en circulacion'].dropna()
                    if not valid_sh.empty and float(valid_sh.iloc[-1]) > 0:
                        market_data["market_cap"] = int(current_price * float(valid_sh.iloc[-1]))

        with PdfPages(final_path) as pdf:
            # PÁGINA 1: DESCRIPCIÓN DE LA EMPRESA, ACTIVIDAD Y MODELO DE NEGOCIO (AGENTE DE OVERVIEW)
            if company_overview:
                try:
                    show_company_overview(pdf, company_overview, ticker=ticker, market_data=market_data)
                except Exception as e:
                    logger.error(f"Error generando página de descripción de la empresa: {e}", exc_info=True)

            # PÁGINA 2: DESGLOSE Y FUENTES DE INGRESOS POR AÑOS (REVENUE SEGMENTS AGENT)
            if segments_data:
                try:
                    show_revenue_segments_table(pdf, segments_data, ticker=ticker, market_data=market_data)
                except Exception as e:
                    logger.error(f"Error generando página de fuentes de ingresos: {e}", exc_info=True)

            # PÁGINA 3: DIAGRAMA SANKEY DEL ESTADO DE RESULTADOS (INCOME STATEMENT FLOW AGENT)
            if income_flow_data:
                try:
                    show_income_statement_flow(pdf, income_flow_data, ticker=ticker, market_data=market_data)
                except Exception as e:
                    logger.error(f"Error generando diagrama Sankey de estado de resultados: {e}", exc_info=True)

            # PÁGINA 4: PANORAMA DE CRECIMIENTO Y VALORACIÓN (CAGR BN, BPA, PER, ROE)
            try:
                show_percentage_difference(pdf, df, ticker=ticker, current_price=current_price)
            except Exception as e:
                logger.error(f"Error generando tabla show_percentage_difference: {e}", exc_info=True)

            # PÁGINA 5: ANÁLISIS DE CAPEX VS BENEFICIO NETO
            try:
                show_capex_vs_netincome(pdf, df, ticker=ticker)
            except Exception as e:
                logger.error(f"Error generando tabla show_capex_vs_netincome: {e}", exc_info=True)

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
                if col in df.columns and not df[col].isna().all() and df[col].sum() != 0:
                    try:
                        fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
                        barras = ax.bar(df['Periodo Fiscal'], df[col])
                        formatear_grafica(barras, ax)
                        ax.set_xlabel('Periodo Fiscal')
                        ax.set_ylabel(y_label)
                        
                        # Título principal en la parte superior izquierda de la figura (sin colisión)
                        fig.suptitle(f'{title} ({ticker})', x=0.06, y=0.96, ha='left', fontsize=12.5, fontweight='bold', color=REPORT_THEME["navy"])
                        
                        # Subtítulo explicativo en la parte superior del gráfico
                        wrapped_desc = textwrap.fill(desc, width=90)
                        ax.set_title(wrapped_desc, loc='left', fontsize=8.8, color=REPORT_THEME["muted"], style='italic', pad=10)

                        if ref1 is not None: ax.axhline(y=ref1, color=col1, linestyle='--', label=f'Ref: {ref1}')
                        if ref2 is not None: ax.axhline(y=ref2, color=col2, linestyle='--', label=f'Ref: {ref2}')
                        if ref1 or ref2: ax.legend(loc='upper right', fontsize=8.5)
                        
                        estilizar_figura_profesional(fig)
                        pdf.savefig(fig)
                    except Exception as e:
                        logger.warning(f"Error graficando {col}: {e}")
                    finally:
                        plt.close('all')

            # PÁGINA: EVALUACIÓN DE MONOPOLIO DE BUFFETTOLOGY
            if monopoly_analysis:
                try:
                    show_monopoly_analysis(pdf, monopoly_analysis, ticker=ticker)
                except Exception as e:
                    logger.error(f"Error generando página de análisis de monopolio: {e}", exc_info=True)

            # PÁGINA: EVALUACIÓN DE BENEFICIOS NO DISTRIBUIDOS (REGLA DEL DÓLAR DE BUFFETT)
            if retained_earnings_analysis:
                try:
                    show_retained_earnings_analysis(pdf, retained_earnings_analysis, ticker=ticker)
                except Exception as e:
                    logger.error(f"Error generando página de análisis de beneficios retenidos: {e}", exc_info=True)

            # PÁGINA: EVALUACIÓN DE ALINEACIÓN DIRECTIVA CON LOS ACCIONISTAS
            if management_analysis:
                try:
                    show_management_alignment_analysis(pdf, management_analysis, ticker=ticker)
                except Exception as e:
                    logger.error(f"Error generando página de análisis de alineación directiva: {e}", exc_info=True)

            # PÁGINA FINAL: AUDITORÍA FORENSE Y DETECCIÓN DE CONTABILIDAD ENGAÑOSA
            if forensic_analysis:
                try:
                    show_accounting_forensic_analysis(pdf, forensic_analysis, ticker=ticker)
                except Exception as e:
                    logger.error(f"Error generando página de auditoría forense contable: {e}", exc_info=True)

        logger.info(f"[PDF Builder] Informe generado exitosamente en: {final_path}")
        return final_path
