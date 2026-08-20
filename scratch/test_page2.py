import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap

REPORT_THEME = {
    "navy": "#0B1F33", "teal": "#1F6F78", "gold": "#D9A441", "sky": "#6FA8DC",
    "ink": "#203040", "text": "#334E68", "muted": "#627D98", "line": "#C8D4E3",
    "panel": "#F7F4EE", "canvas": "#FFFDFC", "positive": "#2D6A4F", "negative": "#B85C38",
}

def _esc(t):
    if not t: return ""
    return str(t).replace('$', r'\$')

def estilizar_tabla(table, header_fontsize=9.0, cell_fontsize=8.4, pad=0.05):
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

def render_page2_preview(output_png: str):
    fig = plt.figure(figsize=(9.5, 7.2), dpi=150)
    fig.patch.set_facecolor(REPORT_THEME['canvas'])
    ax = fig.add_axes([0.045, 0.035, 0.91, 0.93])
    ax.set_facecolor(REPORT_THEME['canvas'])
    ax.axis('off')

    company_name = "3M Company"
    ticker_str = "MMM"
    unit_str = "Billion USD"
    years = [2020, 2021, 2022, 2023, 2024]

    # 1. Header
    ax.text(0.0, 0.985, f'{company_name} ({ticker_str}) - Desglose de Fuentes de Ingresos',
            fontsize=12.5, fontweight='bold', color=REPORT_THEME['navy'], va='top')
    ax.text(0.0, 0.950, f'Evolución histórica por líneas de negocio ({unit_str}) y grado de diversificación (SEC Form 10-K)',
            fontsize=8.6, style='italic', color=REPORT_THEME['muted'], va='top')

    # 2. Table Data
    col_labels = ["Línea de Negocio / Segmento"] + [f"FY {y}" for y in years] + ["% Total", "Crec. YoY"]
    table_data = [
        ["Safety and Industrial", "11.8", "12.9", "11.6", "11.0", "10.8", "44.8%", "-1.8%"],
        ["Transportation & Electronics", "8.8", "9.8", "8.9", "8.5", "8.3", "34.4%", "-2.4%"],
        ["Consumer", "5.3", "5.9", "5.3", "5.0", "5.0", "20.8%", "0.0%"],
        ["TOTAL CONSOLIDADO", "25.9", "28.6", "25.8", "24.5", "24.1", "100.0%", ""]
    ]

    n_cols = len(col_labels)
    w_first = 0.36
    w_other = (1.0 - w_first) / max(n_cols - 1, 1)
    col_widths = [w_first] + [w_other] * (n_cols - 1)

    num_rows = len(table_data) + 1
    row_h = 0.040
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
    estilizar_tabla(table, header_fontsize=8.6, cell_fontsize=8.2, pad=0.05)

    # 3. Cards Below Table: Side-by-Side (2 Columns)
    avail_y_top = table_y - 0.025
    y_bottom = 0.035
    cards_h = avail_y_top - y_bottom
    gap_x = 0.030
    col_w = (1.0 - gap_x) / 2.0
    col_x_offsets = [0.0, col_w + gap_x]
    wrap_w = 56

    # Card 1: Description
    desc_p = [
        "• Safety and Industrial: Abrasivos, cintas, adhesivos y EPIs de seguridad laboral con marcas como Scotch-Brite™ y Cubitron™.",
        "• Transportation & Electronics: Materiales avanzados para automoción, semiconductores, centros de datos y señalización reflectante Scotchlite™.",
        "• Consumer: Productos para el hogar, oficina y bienestar con marcas icónicas como Post-it®, Scotch® y Command™."
    ]
    card1_lines = []
    for p in desc_p:
        wrapped_p = textwrap.fill(_esc(p), width=wrap_w, subsequent_indent="  ")
        card1_lines.extend(wrapped_p.split('\n'))

    # Card 2: Diversification
    div_text = (
        "La empresa presenta una diversificación de ingresos balanceada entre tres pilares operativos con una distribución del 45% en Seguridad e Industria, 34% en Transporte y Electrónica, y 21% en Consumo.\n\n"
        "No existe riesgo de concentración monopsonista en un único cliente ni dependencia extrema de un solo producto. La diversificación geográfica (América 55%, Asia-Pacífico 28%, EMEA 17%) otorga resiliencia frente a recesiones regionales."
    )
    card2_lines = []
    for p in div_text.split('\n\n'):
        if p.strip():
            wrapped_p = textwrap.fill(_esc(p.strip()), width=wrap_w)
            card2_lines.extend(wrapped_p.split('\n'))

    # Render Card 1 (Left)
    rect1 = patches.FancyBboxPatch(
        (col_x_offsets[0], y_bottom), col_w, cards_h,
        boxstyle='round,pad=0.0,rounding_size=0.015',
        facecolor=REPORT_THEME['panel'], edgecolor=REPORT_THEME['line'], linewidth=0.8,
        transform=ax.transAxes, zorder=2
    )
    ax.add_patch(rect1)
    ax.text(col_x_offsets[0] + 0.015, avail_y_top - 0.018, '1. Descripción de las Líneas de Negocio:',
            fontsize=8.3, fontweight='bold', color=REPORT_THEME['navy'], va='top', transform=ax.transAxes, zorder=3)
    ax.text(col_x_offsets[0] + 0.015, avail_y_top - 0.044, '\n'.join(card1_lines),
            fontsize=7.4, color=REPORT_THEME['text'], va='top', transform=ax.transAxes, zorder=3)

    # Render Card 2 (Right - Highlighted)
    rect2 = patches.FancyBboxPatch(
        (col_x_offsets[1], y_bottom), col_w, cards_h,
        boxstyle='round,pad=0.0,rounding_size=0.015',
        facecolor='#F0FDFA', edgecolor=REPORT_THEME['teal'], linewidth=1.3,
        transform=ax.transAxes, zorder=2
    )
    ax.add_patch(rect2)
    ax.text(col_x_offsets[1] + 0.015, avail_y_top - 0.018, '2. Análisis de Diversificación y Riesgos:',
            fontsize=8.3, fontweight='bold', color='#0F766E', va='top', transform=ax.transAxes, zorder=3)
    ax.text(col_x_offsets[1] + 0.015, avail_y_top - 0.044, '\n'.join(card2_lines),
            fontsize=7.4, fontweight='bold', color=REPORT_THEME['navy'], va='top', transform=ax.transAxes, zorder=3)

    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Saved {output_png}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "data", "output_reports")
    render_page2_preview(os.path.join(out_dir, "preview_page2_test.png"))
