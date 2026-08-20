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

def render_2x2_executive_page(
    main_title: str,
    subtitle: str,
    verdict_text: str,
    badge_color: str,
    cards: list,
    output_png: str
):
    fig = plt.figure(figsize=(9.5, 7.2), dpi=150)
    fig.patch.set_facecolor(REPORT_THEME['canvas'])
    ax = fig.add_axes([0.045, 0.035, 0.91, 0.93])
    ax.set_facecolor(REPORT_THEME['canvas'])
    ax.axis('off')

    # 1. Header
    ax.text(0.0, 0.985, _esc(main_title), fontsize=12.5, fontweight='bold', color=REPORT_THEME['navy'], va='top')
    ax.text(0.0, 0.950, _esc(subtitle), fontsize=8.6, style='italic', color=REPORT_THEME['muted'], va='top')

    # 2. Verdict Banner
    ax.text(0.5, 0.885, _esc(verdict_text), fontsize=8.6, fontweight='bold', ha='center', va='center', color=badge_color,
            bbox=dict(boxstyle='round,pad=0.38', facecolor=REPORT_THEME['panel'], edgecolor=badge_color, linewidth=1.2))

    y_top = 0.825
    y_bottom = 0.035
    usable_h = y_top - y_bottom
    gap_y = 0.022
    gap_x = 0.030
    usable_col_h = usable_h - gap_y
    col_w = (1.0 - gap_x) / 2.0
    col_x_offsets = [0.0, col_w + gap_x]
    wrap_w = 56
    title_wrap_w = 40

    cols_data = [
        [cards[0], cards[1]],
        [cards[2], cards[3]]
    ]

    for col_idx, col_cards in enumerate(cols_data):
        col_x = col_x_offsets[col_idx]
        wrapped = []
        for title, body, is_hl in col_cards:
            title_lines = textwrap.fill(_esc(title), width=title_wrap_w).split('\n')
            lines = []
            for p in str(body or "").split('\n\n'):
                if p.strip():
                    wrapped_p = textwrap.fill(_esc(p.strip()), width=wrap_w, subsequent_indent="  " if p.strip().startswith("•") else "")
                    lines.extend(wrapped_p.split('\n'))
            if not lines:
                lines = ["Sin información adicional reportada."]
            wrapped.append((title_lines, lines, is_hl))

        lines_0 = max(len(wrapped[0][1]), 3) + len(wrapped[0][0])
        lines_1 = max(len(wrapped[1][1]), 3) + len(wrapped[1][0])
        total_l = lines_0 + lines_1
        ratio_0 = max(0.36, min(0.64, lines_0 / total_l))
        h0 = usable_col_h * ratio_0
        h1 = usable_col_h * (1.0 - ratio_0)

        card_layouts = [(wrapped[0], y_top, h0), (wrapped[1], y_top - h0 - gap_y, h1)]

        for (t_lines, lines, is_hl), top_y, h in card_layouts:
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
            title_text = '\n'.join(t_lines)
            ax.text(col_x + 0.015, top_y - 0.018, title_text,
                    fontsize=8.1, fontweight='bold', color=title_col, va='top', transform=ax.transAxes, zorder=3)

            # Available height in points
            axis_pt = 482.0
            h_pt = h * axis_pt
            title_space_pt = len(t_lines) * 10.5 + 14.0
            text_avail_pt = h_pt - title_space_pt
            line_height_pt = 8.6
            max_fit_lines = max(int(text_avail_pt / line_height_pt), 1)

            font_sz = 7.2
            if len(lines) > max_fit_lines:
                display_lines = lines[:max_fit_lines]
                if display_lines:
                    display_lines[-1] = display_lines[-1].rstrip('.') + '...'
            else:
                display_lines = lines
                if len(lines) <= max_fit_lines - 4:
                    font_sz = 7.6

            body_str = '\n'.join(display_lines)
            body_col = REPORT_THEME['navy'] if is_hl else REPORT_THEME['text']
            body_top_offset = (title_space_pt - 4.0) / axis_pt
            ax.text(col_x + 0.015, top_y - body_top_offset, body_str,
                    fontsize=font_sz, fontweight='bold' if is_hl else 'normal',
                    color=body_col, va='top', transform=ax.transAxes, zorder=3)

    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Saved {output_png}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "data", "output_reports")

    # Monopoly test
    cards_monopoly = [
        ('1. Análisis del Modelo de Negocio e Informes 10-K (SEC):',
         '3M Company opera con un foso económico basado en patentes industriales, marcas registradas de consumo e integración en cadenas de suministro globales. Sus segmentos de Safety & Industrial y Transportation & Electronics gozan de elevados costes de cambio para clientes corporativos debido a homologaciones y especificaciones técnicas.', False),
        ('2. Poder de Fijación de Precios & Evidencia Cuantitativa (10 Años):',
         '• Fijación de Precios: Capacidad de traspasar incrementos de materias primas en productos de alta especialidad.\n• Métricas Contables: Márgenes brutos medios superiores al 42% y ROE medio del 24% a lo largo de la última década.', False),
        ('3. Amenazas al Foso Defensivo & Riesgos Regulatorios:',
         'Litigios ambientales por PFAS y tapones auditivos militares (Dual-Ended Combat Arms). Transición hacia modelos sin sustancias perfluoroalquiladas y mayor escrutinio regulatorio en EE. UU. y Europa.', False),
        ('4. Conclusión Final de Warren Buffett:',
         '3M representa una franquicia industrial sólida con ventajas competitivas duraderas en adhesivos y abrasivos, aunque los pasivos legales pasados requieren exigir un mayor margen de seguridad en el precio de adquisición.', True)
    ]
    render_2x2_executive_page(
        'Evaluación de Monopolio de Buffettology (MMM)',
        'Pregunta 1 de Warren Buffett: ¿Tiene la empresa un monopolio fácilmente identificable?',
        'VEREDICTO: FOSO ECONÓMICO AMPLIO\nTipo de Foso: Marcas Registradas & Costes de Cambio',
        REPORT_THEME['positive'],
        cards_monopoly,
        os.path.join(out_dir, "preview_monopoly_test.png")
    )
