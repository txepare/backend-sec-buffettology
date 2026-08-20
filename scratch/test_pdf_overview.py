import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap

# Load cached 10-K narrative or facts for MMM
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(base_dir, "data", "output_reports", "preview_page1_test.png")

REPORT_THEME = {
    "navy": "#0B1F33", "teal": "#1F6F78", "gold": "#D9A441", "sky": "#6FA8DC",
    "ink": "#203040", "text": "#334E68", "muted": "#627D98", "line": "#C8D4E3",
    "panel": "#F7F4EE", "canvas": "#FFFDFC", "positive": "#2D6A4F", "negative": "#B85C38",
}

def _esc(t):
    if not t: return ""
    return str(t).replace('$', r'\$')

# Example data for MMM from the run
overview_result = {
    "veredicto_comprensibilidad": "MODERADO (Modelo Comprensible con Particularidades de Industria)",
    "categoria_comprensibilidad": "MODERADO",
    "descripcion_corta": "Conglomerado tecnológico diversificado líder global en ciencia de materiales, adhesivos, abrasivos y equipos de protección personal.",
    "resumen_actividad": "3M Company es una empresa de tecnología diversificada que investiga, desarrolla, fabrica y comercializa una vasta gama de productos industriales, de seguridad, electrónicos y de consumo. Su actividad se basa en aplicar tecnologías propietarias en ciencia de materiales para crear soluciones innovadoras, operando a través de tres segmentos principales: Safety and Industrial, Transportation and Electronics, y Consumer. Tras la escisión de su negocio de salud (Solventum) en 2024 y su salida de la manufactura de PFAS a finales de 2025, la compañía se enfoca en la excelencia comercial y la optimización de su cadena de valor global.",
    "ubicacion_y_mercados": "La sede central corporativa y los laboratorios principales de investigación se ubican en St. Paul, Minnesota (EE. UU.). 3M opera 48 instalaciones de manufactura en 26 estados de EE. UU. y 60 instalaciones de fabricación y conversión en 25 países a nivel internacional. Sus ventas están geográficamente diversificadas: un 54.5% en América, 28.4% en Asia-Pacífico y 17.1% en EMEA.",
    "lineas_de_negocio": [
        {"nombre": "Safety and Industrial", "descripcion": "Abarca abrasivos industriales, soluciones para talleres automotrices, materiales eléctricos, adhesivos y cintas industriales, divisiones de especialidades industriales y equipos de protección personal (respiradores, protección auditiva y ocular, sistemas anticaídas). Sus marcas insignia incluyen Cubitron™, Scotch-Brite™, VHB™ Tapes, DBI-Sala™ y Scott™."},
        {"nombre": "Transportation and Electronics", "descripcion": "Comprende soluciones de materiales avanzados, componentes para automoción y aeroespacial, marcas comerciales y transporte (gráficos y señalización vial reflectante), materiales para pantallas y soluciones para la industria electrónica (interconexión de chips, materiales para semiconductores y centros de datos). Marcas destacadas incluyen Scotchlite™, Thinsulate™, Controltac™ y Diamond Grade™."},
        {"nombre": "Consumer", "descripcion": "Diseñado para el mercado minorista y de consumo masivo, incluye productos de seguridad y bienestar en el hogar, cuidado del automóvil, mejoras para el hogar y suministros de oficina. Sus productos y marcas más icónicos son Post-it®, Scotch®, Command™, Filtrete™, Nexcare™ y Meguiar's™."}
    ],
    "vientos_de_cola_y_sector": "La compañía se beneficia de tendencias seculares como la automatización y la robótica, la electrificación automotriz, la digitalización y el crecimiento de la infraestructura de centros de datos, la modernización de redes eléctricas, la demanda de tecnologías de semiconductores y la creciente conciencia global sobre la seguridad personal e industrial.",
    "perspectivas_crecimiento": "Perspectivas de crecimiento moderado impulsadas por la excelencia comercial, el lanzamiento de nuevos productos y la eficiencia operativa. No obstante, la dirección advierte sobre riesgos como presiones inflacionarias persistentes, aranceles, incertidumbre geopolítica, volatilidad en los mercados automotrices y de la construcción, así como potenciales costos y pasivos asociados a obligaciones de remediación ambiental y litigios históricos.",
    "propuesta_valor": "La propuesta de valor de 3M descansa en su profunda experiencia en I+D y ciencia de materiales, permitiéndole ofrecer productos de alto rendimiento y durabilidad bajo marcas de reconocido prestigio global. Esto genera una alta confianza y fidelidad en clientes industriales y consumidores finales que buscan fiabilidad y calidad superior frente a alternativas genéricas.",
    "circulo_competencia": "3M presenta un modelo de negocio tangible y comprensible en sus líneas tradicionales, pero su diversificación masiva en miles de referencias, la alta intensidad tecnológica y los riesgos legales históricos exigen un seguimiento continuo para inversores de valor bajo la filosofía de Buffett."
}

def render_page1_preview():
    fig = plt.figure(figsize=(9.5, 7.2), dpi=150)
    fig.patch.set_facecolor(REPORT_THEME['canvas'])
    ax = fig.add_axes([0.045, 0.035, 0.91, 0.93])
    ax.set_facecolor(REPORT_THEME['canvas'])
    ax.axis('off')

    company_name = "3M Company"
    ticker_str = "MMM"
    sector = "Industrial"
    industry = "General"
    current_price = 180.66
    market_cap = 97.79e9

    # 1. Header
    ax.text(0.0, 0.985, f'{company_name} ({ticker_str}) - Perfil & Modelo de Negocio (SEC Form 10-K)',
            fontsize=12.5, fontweight='bold', color=REPORT_THEME['navy'], va='top')
    
    sub_info = f"Sector: {sector}  |  Industria: {industry}  |  Cotización: ${current_price:.2f}  |  Market Cap: $97.79B"
    ax.text(0.0, 0.950, _esc(sub_info), fontsize=8.6, color=REPORT_THEME['muted'], va='top')

    # 2. Verdict Banner
    categoria = overview_result.get('categoria_comprensibilidad', 'MODERADO')
    badge_color = REPORT_THEME['positive'] if categoria == 'ALTO' else (REPORT_THEME['teal'] if categoria == 'MODERADO' else REPORT_THEME['negative'])
    verdicto = overview_result.get('veredicto_comprensibilidad', '')
    desc_corta = overview_result.get('descripcion_corta', '')
    verdict_text = f"CÍRCULO DE COMPETENCIA DE BUFFETT: {_esc(verdicto)}\n{_esc(desc_corta)}"
    
    ax.text(0.5, 0.885, verdict_text, fontsize=8.6, fontweight='bold', ha='center', va='center', color=badge_color,
            bbox=dict(boxstyle='round,pad=0.38', facecolor=REPORT_THEME['panel'], edgecolor=badge_color, linewidth=1.2))

    # Prepare 4 blocks of content
    resumen_act = overview_result.get('resumen_actividad', '')
    ubicacion = overview_result.get('ubicacion_y_mercados', '')
    card1_body = f"{resumen_act}\n\n• Sede y Alcance: {ubicacion}"

    lineas_list = []
    for item in overview_result.get('lineas_de_negocio', []):
        nom = item.get('nombre', '')
        desc = item.get('descripcion', '')
        lineas_list.append(f"• {nom}: {desc}")
    card2_body = "\n\n".join(lineas_list)

    tailwinds = overview_result.get('vientos_de_cola_y_sector', '')
    growth = overview_result.get('perspectivas_crecimiento', '')
    card3_body = f"• Vientos de Cola (Tailwinds): {tailwinds}\n\n• Perspectivas & Riesgos: {growth}"

    propuesta = overview_result.get('propuesta_valor', '')
    circulo = overview_result.get('circulo_competencia', '')
    card4_body = f"• Propuesta de Valor: {propuesta}\n\n• Dictamen de Warren Buffett: {circulo}"

    # 2-Column Grid Layout
    cards_col1 = [
        ("1. Actividad de la Empresa y Sede Corporativa (10-K):", card1_body, False),
        ("2. Líneas de Negocio y Segmentos Operativos:", card2_body, False)
    ]
    cards_col2 = [
        ("3. Vientos de Cola y Perspectivas de Crecimiento:", card3_body, False),
        ("4. Propuesta de Valor Diferencial & Veredicto Buffett:", card4_body, True)
    ]

    y_top = 0.825
    y_bottom = 0.035
    usable_h = y_top - y_bottom
    gap_y = 0.022
    usable_col_h = usable_h - gap_y
    col_w = 0.485
    col_x_offsets = [0.0, 0.515]
    wrap_w = 56

    def layout_column(cards, col_x):
        # Wrap lines and measure
        wrapped = []
        for title, body, is_hl in cards:
            lines = []
            for p in body.split('\n\n'):
                if p.strip():
                    wrapped_p = textwrap.fill(_esc(p.strip()), width=wrap_w)
                    lines.extend(wrapped_p.split('\n'))
            wrapped.append((title, lines, is_hl))

        lines_0 = max(len(wrapped[0][1]), 3)
        lines_1 = max(len(wrapped[1][1]), 3)
        total_l = lines_0 + lines_1
        ratio_0 = max(0.38, min(0.62, lines_0 / total_l))
        h0 = usable_col_h * ratio_0
        h1 = usable_col_h * (1.0 - ratio_0)

        card_layouts = [(wrapped[0], y_top, h0), (wrapped[1], y_top - h0 - gap_y, h1)]

        for (title, lines, is_hl), top_y, h in card_layouts:
            box_y = top_y - h
            bg_col = '#F0FDFA' if is_hl else REPORT_THEME['panel']
            edge_col = REPORT_THEME['teal'] if is_hl else REPORT_THEME['line']
            lw = 1.3 if is_hl else 0.8

            rect = patches.FancyBboxPatch(
                (col_x, box_y), col_w, h,
                boxstyle='round,pad=0.0,rounding_size=0.015',
                facecolor=bg_col, edgecolor=edge_col, linewidth=lw,
                transform=ax.transAxes, zorder=2
            )
            ax.add_patch(rect)

            title_col = '#0F766E' if is_hl else REPORT_THEME['navy']
            ax.text(col_x + 0.015, top_y - 0.020, _esc(title),
                    fontsize=8.3, fontweight='bold', color=title_col, va='top', transform=ax.transAxes, zorder=3)

            # Calculate how many lines fit inside this card height
            # Height in axes units: h. Height for text: h - 0.045
            # At 7.2pt font, each line is approx 0.0185 in axes units (482 points total height)
            axis_pt = 482.0
            h_pt = h * axis_pt
            text_avail_pt = h_pt - 26.0  # 26pt for title and margins
            line_height_pt = 8.6  # for 7.0pt font
            max_fit_lines = int(text_avail_pt / line_height_pt)

            font_sz = 7.2
            if len(lines) > max_fit_lines:
                # Truncate gracefully
                display_lines = lines[:max_fit_lines]
                if display_lines:
                    display_lines[-1] = display_lines[-1].rstrip('.') + '...'
            else:
                display_lines = lines
                if len(lines) <= max_fit_lines - 4:
                    font_sz = 7.6

            body_str = '\n'.join(display_lines)
            body_col = REPORT_THEME['navy'] if is_hl else REPORT_THEME['text']
            ax.text(col_x + 0.015, top_y - 0.046, body_str,
                    fontsize=font_sz, fontweight='bold' if is_hl else 'normal',
                    color=body_col, va='top', transform=ax.transAxes, zorder=3)

    layout_column(cards_col1, col_x_offsets[0])
    layout_column(cards_col2, col_x_offsets[1])

    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
    print(f"Generated preview at: {pdf_path}")

if __name__ == "__main__":
    render_page1_preview()
