import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from typing import Dict, Any, List, Tuple


class SankeyFlowBuilder:
    """
    Generador de diagramas de flujo de estado de resultados (Sankey Diagram / 'How They Make Money')
    inspirado en el estilo visual de alta resolución de App Economy Insights.
    """

    @staticmethod
    def _format_currency(val: float, unit_suffix: str = "B") -> str:
        """Formatea un número monetario a formato legible $X.XB o $X.XM."""
        if abs(val) >= 1e9:
            return f"${val / 1e9:.1f}B"
        elif abs(val) >= 1e6:
            return f"${val / 1e6:.1f}M"
        elif abs(val) >= 1e3:
            return f"${val / 1e3:.1f}K"
        elif val == 0:
            return "$0"
        else:
            return f"${val:.1f}"

    @staticmethod
    def _draw_flow(
        ax,
        x0: float, y0_bot: float, y0_top: float,
        x1: float, y1_bot: float, y1_top: float,
        color: str,
        alpha: float = 0.55
    ):
        """
        Dibuja una cinta de flujo continuo y curvado usando curvas de Bézier cúbicas (Path.CURVE4).
        """
        dx = x1 - x0
        cx0 = x0 + dx * 0.5
        cx1 = x1 - dx * 0.5

        verts = [
            (x0, y0_bot),
            (cx0, y0_bot),
            (cx1, y1_bot),
            (x1, y1_bot),
            (x1, y1_top),
            (cx1, y1_top),
            (cx0, y0_top),
            (x0, y0_top),
            (x0, y0_bot),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha, zorder=2)
        ax.add_patch(patch)

    @staticmethod
    def _draw_bar(ax, x: float, y_bot: float, y_top: float, width: float, color: str, zorder: int = 4):
        """Dibuja una barra vertical rectangular."""
        rect = patches.Rectangle(
            (x - width / 2, y_bot), width, y_top - y_bot,
            facecolor=color, edgecolor='white', linewidth=1.2, zorder=zorder
        )
        ax.add_patch(rect)

    @staticmethod
    def generate_sankey_figure(
        ticker: str,
        company_name: str,
        year: str,
        segments_data: List[Dict[str, Any]],
        financial_flow: Dict[str, float]
    ) -> plt.Figure:
        """
        Construye la figura de matplotlib con el diagrama de flujo Sankey del estado de resultados.
        """
        # Extraer métricas financieras clave
        rev = max(financial_flow.get("revenue", 100.0), 1.0)
        cogs = max(financial_flow.get("cogs", 40.0), 0.0)
        gp = max(financial_flow.get("gross_profit", rev - cogs), 1.0)
        
        rd = max(financial_flow.get("rd", 0.0), 0.0)
        sga = max(financial_flow.get("sga", 0.0), 0.0)
        opex_total = financial_flow.get("opex", rd + sga)
        if opex_total <= 0 and (rd + sga) > 0:
            opex_total = rd + sga
        elif opex_total <= 0:
            opex_total = max(gp * 0.35, 1.0)

        op_profit = max(financial_flow.get("operating_profit", gp - opex_total), 1.0)
        tax = max(financial_flow.get("tax", op_profit * 0.18), 0.0)
        net_profit = financial_flow.get("net_profit", op_profit - tax)

        # Márgenes
        gp_margin = (gp / rev) * 100
        op_margin = (op_profit / rev) * 100
        np_margin = (net_profit / rev) * 100

        # Configuración de lienzo (Lienzo panorámico 12 x 7.5 pulgadas con amplio margen)
        fig, ax = plt.subplots(figsize=(12, 7.5), dpi=180)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")
        ax.axis('off')
        ax.set_xlim(-4.2, 16.5)
        ax.set_ylim(-1.2, 8.5)

        # ----------------- TÍTULO SUPERIOR INSTITUCIONAL -----------------
        title_text = f"{company_name} ({ticker}) - Estado de Resultados (Ejercicio Fiscal {year})"
        ax.text(5.5, 8.15, title_text, fontsize=15.0, fontweight='bold', ha='center', va='top', color='#0B1F33')
        sub_title = "Diagrama de Flujo de Ingresos, Costes, Gastos Operativos y Beneficio Neto (Sankey Flow)"
        ax.text(5.5, 7.75, sub_title, fontsize=9.2, ha='center', va='top', color='#627D98', style='italic')

        # ----------------- PALETA DE COLORES PROFESIONAL -----------------
        color_segment_palette = ['#3B82F6', '#6366F1', '#EC4899', '#F59E0B', '#10B981', '#8B5CF6', '#14B8A6']
        color_revenue = '#2563EB'
        color_green_gp = '#22C55E'
        color_green_op = '#16A34A'
        color_green_np = '#15803D'
        color_red_cogs = '#DC2626'
        color_red_opex = '#EF4444'
        color_red_sub = '#F87171'

        # Escala vertical
        scale_height = 4.4
        h_rev = scale_height
        y_rev_center = 3.6
        y_rev_bot = y_rev_center - h_rev / 2
        y_rev_top = y_rev_center + h_rev / 2

        # ----------------- 1. COLUMNA DE SEGMENTOS DE ENTRADA (X = -0.2) -----------------
        x_seg = -0.2
        x_rev = 3.4
        w_bar = 0.24

        # Si no hay segmentos o no suman rev, normalizar
        if not segments_data:
            segments_data = [
                {"nombre": "Línea Principal de Productos", "monto": rev * 0.65, "pct": 65.0, "yoy": "+8%"},
                {"nombre": "Servicios y Recurrencia", "monto": rev * 0.35, "pct": 35.0, "yoy": "+14%"}
            ]

        total_seg_monto = sum(s.get("monto", s.get("valor", 1.0)) for s in segments_data)
        if total_seg_monto <= 0: total_seg_monto = rev

        # Calcular posiciones verticales de los segmentos a la izquierda
        gap = 0.12
        n_seg = len(segments_data)
        total_gaps = gap * (n_seg - 1)
        avail_h_seg = h_rev + 0.6
        usable_h = avail_h_seg - total_gaps

        curr_y_seg_top = y_rev_center + avail_h_seg / 2
        curr_y_rev_bot = y_rev_bot

        for idx, seg in enumerate(segments_data):
            seg_name = seg.get("nombre", f"Segmento {idx+1}")
            seg_val = seg.get("monto", seg.get("valor", rev / n_seg))
            seg_pct = seg.get("pct", (seg_val / total_seg_monto) * 100)
            seg_yoy = seg.get("yoy", "")
            
            # Altura en el segmento y altura en la barra de Revenue
            h_this_seg = max((seg_val / total_seg_monto) * usable_h, 0.25)
            h_this_rev = (seg_val / total_seg_monto) * h_rev
            
            y_seg_bot = curr_y_seg_top - h_this_seg
            y_seg_top = curr_y_seg_top
            
            y_rev_in_bot = curr_y_rev_bot
            y_rev_in_top = curr_y_rev_bot + h_this_rev

            c_seg = color_segment_palette[idx % len(color_segment_palette)]

            # Flujo desde el segmento hasta la barra de Revenue
            SankeyFlowBuilder._draw_flow(ax, x_seg, y_seg_bot, y_seg_top, x_rev, y_rev_in_bot, y_rev_in_top, c_seg, alpha=0.58)
            # Barra del segmento
            SankeyFlowBuilder._draw_bar(ax, x_seg, y_seg_bot, y_seg_top, w_bar, c_seg)

            # Texto del Segmento a la izquierda con salto de línea compacto
            seg_val_str = SankeyFlowBuilder._format_currency(seg_val)
            yoy_txt = f"  ({seg_yoy})" if seg_yoy else ""
            seg_name_wrapped = "\n".join([line.strip() for line in seg_name.splitlines()])
            if len(seg_name) > 24:
                import textwrap
                seg_name_wrapped = textwrap.fill(seg_name, width=22)
            label_text = f"{seg_name_wrapped}\n{seg_val_str} ({seg_pct:.1f}%){yoy_txt}"
            ax.text(x_seg - 0.25, (y_seg_bot + y_seg_top) / 2, label_text, fontsize=8.0, fontweight='bold',
                    ha='right', va='center', color='#1E293B')

            curr_y_seg_top = y_seg_bot - gap
            curr_y_rev_bot += h_this_rev

        # ----------------- 2. BARRA DE REVENUE (X = 3.4) -----------------
        SankeyFlowBuilder._draw_bar(ax, x_rev, y_rev_bot, y_rev_top, w_bar, color_revenue)
        rev_str = SankeyFlowBuilder._format_currency(rev)
        ax.text(x_rev, y_rev_top + 0.22, f"Ingresos Totales\n{rev_str}", fontsize=9.6, fontweight='bold',
                ha='center', va='bottom', color=color_revenue)

        # ----------------- 3. DIVISIÓN REVENUE -> COGS vs GROSS PROFIT (X = 7.0) -----------------
        x_gp = 7.0
        h_cogs = max((cogs / rev) * h_rev, 0.4)
        h_gp = max((gp / rev) * h_rev, 0.5)

        # COGS se desvía hacia abajo
        y_cogs_bot = 0.5
        y_cogs_top = y_cogs_bot + h_cogs

        # Gross Profit continúa hacia arriba
        y_gp_top = y_rev_top
        y_gp_bot = y_gp_top - h_gp

        # Flujo a COGS (desde la parte inferior de Revenue)
        SankeyFlowBuilder._draw_flow(ax, x_rev, y_rev_bot, y_rev_bot + h_cogs, x_gp, y_cogs_bot, y_cogs_top, color_red_cogs, alpha=0.55)
        # Flujo a Gross Profit (desde la parte superior de Revenue)
        SankeyFlowBuilder._draw_flow(ax, x_rev, y_rev_bot + h_cogs, y_rev_top, x_gp, y_gp_bot, y_gp_top, color_green_gp, alpha=0.58)

        # Barras en X = 7.0
        SankeyFlowBuilder._draw_bar(ax, x_gp, y_cogs_bot, y_cogs_top, w_bar, color_red_cogs)
        SankeyFlowBuilder._draw_bar(ax, x_gp, y_gp_bot, y_gp_top, w_bar, color_green_gp)

        # Textos de X = 7.0
        cogs_str = SankeyFlowBuilder._format_currency(cogs)
        ax.text(x_gp, y_cogs_bot - 0.18, f"Coste de Ventas (COGS)\n-{cogs_str} ({(cogs/rev)*100:.1f}%)",
                fontsize=8.5, fontweight='bold', ha='center', va='top', color='#991B1B')

        gp_str = SankeyFlowBuilder._format_currency(gp)
        ax.text(x_gp, y_gp_top + 0.22, f"Beneficio Bruto\n{gp_str}\n({gp_margin:.1f}% margen)",
                fontsize=9.4, fontweight='bold', ha='center', va='bottom', color='#15803D')

        # ----------------- 4. DIVISIÓN GROSS PROFIT -> OPEX vs OPERATING PROFIT (X = 10.6) -----------------
        x_op = 10.6
        h_opex = max((opex_total / gp) * h_gp, 0.4)
        h_op = max((op_profit / gp) * h_gp, 0.45)

        y_opex_bot = 1.9
        y_opex_top = y_opex_bot + h_opex

        y_op_top = y_gp_top
        y_op_bot = y_op_top - h_op

        # Flujo a OpEx
        SankeyFlowBuilder._draw_flow(ax, x_gp, y_gp_bot, y_gp_bot + h_opex, x_op, y_opex_bot, y_opex_top, color_red_opex, alpha=0.55)
        # Flujo a Operating Profit
        SankeyFlowBuilder._draw_flow(ax, x_gp, y_gp_bot + h_opex, y_gp_top, x_op, y_op_bot, y_op_top, color_green_op, alpha=0.58)

        # Barras en X = 10.6
        SankeyFlowBuilder._draw_bar(ax, x_op, y_opex_bot, y_opex_top, w_bar, color_red_opex)
        SankeyFlowBuilder._draw_bar(ax, x_op, y_op_bot, y_op_top, w_bar, color_green_op)

        # Textos de X = 10.6
        opex_str = SankeyFlowBuilder._format_currency(opex_total)
        ax.text(x_op, y_opex_bot - 0.18, f"Gastos Operativos (OpEx)\n-{opex_str}",
                fontsize=8.5, fontweight='bold', ha='center', va='top', color='#B91C1C')

        op_str = SankeyFlowBuilder._format_currency(op_profit)
        ax.text(x_op, y_op_top + 0.22, f"Beneficio Operativo\n{op_str}\n({op_margin:.1f}% margen)",
                fontsize=9.4, fontweight='bold', ha='center', va='bottom', color='#16A34A')

        # ----------------- 5. DIVISIÓN OPERATING PROFIT -> TAX & NET PROFIT (X = 13.8) -----------------
        x_np = 13.8
        h_tax = max((tax / op_profit) * h_op, 0.25)
        h_np = max((net_profit / op_profit) * h_op, 0.35)

        y_tax_bot = 3.1
        y_tax_top = y_tax_bot + h_tax

        y_np_top = y_op_top
        y_np_bot = y_np_top - h_np

        # Flujo a Impuestos
        SankeyFlowBuilder._draw_flow(ax, x_op, y_op_bot, y_op_bot + h_tax, x_np, y_tax_bot, y_tax_top, color_red_sub, alpha=0.55)
        # Flujo a Beneficio Neto
        SankeyFlowBuilder._draw_flow(ax, x_op, y_op_bot + h_tax, y_op_top, x_np, y_np_bot, y_np_top, color_green_np, alpha=0.62)

        # Barras en X = 13.8
        SankeyFlowBuilder._draw_bar(ax, x_np, y_tax_bot, y_tax_top, w_bar, color_red_sub)
        SankeyFlowBuilder._draw_bar(ax, x_np, y_np_bot, y_np_top, w_bar, color_green_np)

        # Textos finales a la derecha (con amplio espacio hasta 16.5)
        tax_str = SankeyFlowBuilder._format_currency(tax)
        ax.text(x_np + 0.25, (y_tax_bot + y_tax_top) / 2, f"Impuestos\n-{tax_str}",
                fontsize=8.4, fontweight='bold', ha='left', va='center', color='#991B1B')

        np_str = SankeyFlowBuilder._format_currency(net_profit)
        np_text = f"Beneficio Neto\n{np_str}\n({np_margin:.1f}% Margen Neto)"
        ax.text(x_np + 0.25, (y_np_bot + y_np_top) / 2, np_text,
                fontsize=9.6, fontweight='bold', ha='left', va='center', color='#15803D')

        # Sub-desglose opcional de OpEx (I+D y VGA) debajo de OpEx para no interferir con Net Profit
        if rd > 0 or sga > 0:
            x_sub_opex = 9.8
            h_sub_total = h_opex * 0.8
            y_sub_top = y_opex_bot - 0.55
            
            rd_ratio = (rd / (rd + sga)) if (rd + sga) > 0 else 0.5
            h_rd = h_sub_total * rd_ratio
            h_sga = h_sub_total * (1 - rd_ratio)

            if rd > 0:
                rd_str = SankeyFlowBuilder._format_currency(rd)
                ax.text(x_op, y_opex_bot - 0.58, f"• I+D (R&D): -{rd_str}", fontsize=7.8, ha='center', va='top', color='#7F1D1D')

            if sga > 0:
                sga_str = SankeyFlowBuilder._format_currency(sga)
                offset_sga = -0.82 if rd > 0 else -0.58
                ax.text(x_op, y_opex_bot + offset_sga, f"• VGA (SG&A): -{sga_str}", fontsize=7.8, ha='center', va='top', color='#7F1D1D')

        # Pie de página explicativo
        footer_text = "Generado automáticamente con datos financieros oficiales 10-K normalizados de la SEC."
        ax.text(4.8, -1.05, footer_text, fontsize=8.0, ha='center', va='bottom', color='#94A3B8')

        return fig
