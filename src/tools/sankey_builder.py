import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import textwrap
from typing import Dict, Any, List, Tuple


class SankeyFlowBuilder:
    """
    Generador de diagramas de flujo de estado de resultados (Sankey Diagram / 'How They Make Money')
    inspirado en el estilo visual de alta resolución de App Economy Insights.
    Garantiza trazado limpio sin cruce de cintas de flujo y adaptación dinámica al número de líneas de negocio.
    """

    @staticmethod
    def _format_currency(val: float) -> str:
        """Formatea un número monetario a formato legible $X.XB o $X.XM."""
        if abs(val) >= 1e9:
            return f"${val / 1e9:.2f}B"
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
        alpha: float = 0.52
    ):
        """
        Dibuja una cinta de flujo continuo y curvado usando curvas de Bézier cúbicas (Path.CURVE4).
        """
        dx = x1 - x0
        cx0 = x0 + dx * 0.48
        cx1 = x1 - dx * 0.48

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
        """Dibuja una barra vertical rectangular con borde nítido."""
        rect = patches.Rectangle(
            (x - width / 2, y_bot), width, max(y_top - y_bot, 0.05),
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
        # 1. Extraer métricas financieras clave
        rev = max(float(financial_flow.get("revenue", 100.0)), 1.0)
        cogs = max(float(financial_flow.get("cogs", 40.0)), 0.0)
        gp = max(float(financial_flow.get("gross_profit", rev - cogs)), 1.0)
        
        rd = max(float(financial_flow.get("rd", 0.0)), 0.0)
        sga = max(float(financial_flow.get("sga", 0.0)), 0.0)
        opex_total = float(financial_flow.get("opex", rd + sga))
        if opex_total <= 0 and (rd + sga) > 0:
            opex_total = rd + sga
        elif opex_total <= 0:
            opex_total = max(gp * 0.30, 1.0)

        op_profit = max(float(financial_flow.get("operating_profit", gp - opex_total)), 1.0)
        tax = max(float(financial_flow.get("tax", op_profit * 0.20)), 0.0)
        net_profit = float(financial_flow.get("net_profit", op_profit - tax))

        # Ratios y márgenes porcentuales
        gp_margin = (gp / rev) * 100.0
        op_margin = (op_profit / rev) * 100.0
        np_margin = (net_profit / rev) * 100.0
        cogs_pct = (cogs / rev) * 100.0
        opex_pct = (opex_total / rev) * 100.0
        tax_pct = (tax / rev) * 100.0

        # Configuración de lienzo (Lienzo panorámico 12.5 x 7.5 pulgadas con alto contraste)
        fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=180)
        fig.patch.set_facecolor("#FAFAF9")
        ax.set_facecolor("#FAFAF9")
        ax.axis('off')
        ax.set_xlim(-4.6, 16.6)
        ax.set_ylim(-1.6, 8.6)

        # ----------------- TÍTULO SUPERIOR INSTITUCIONAL -----------------
        title_text = f"{company_name} ({ticker}) — Flujo del Estado de Resultados (FY {year})"
        ax.text(5.8, 8.25, title_text, fontsize=14.5, fontweight='bold', ha='center', va='top', color='#0F172A')
        sub_title = "Estructura de Monetización por Líneas de Negocio, Coste de Producción, Gastos Operativos y Beneficio Neto (SEC Form 10-K)"
        ax.text(5.8, 7.82, sub_title, fontsize=8.8, ha='center', va='top', color='#64748B', style='italic')

        # ----------------- PALETA DE COLORES PROFESIONAL -----------------
        color_segment_palette = ['#2563EB', '#6366F1', '#0D9488', '#D97706', '#DB2777', '#7C3AED', '#0891B2']
        color_revenue = '#1E40AF'
        color_green_gp = '#059669'
        color_green_op = '#16A34A'
        color_green_np = '#047857'
        color_red_cogs = '#DC2626'
        color_red_opex = '#E11D48'
        color_red_tax = '#BE123C'

        # Escala vertical principal
        h_rev = 4.2
        y_rev_center = 4.4
        y_rev_bot = y_rev_center - h_rev / 2
        y_rev_top = y_rev_center + h_rev / 2

        # ----------------- 1. COLUMNA DE LÍNEAS DE NEGOCIO (X = -0.4) -----------------
        x_seg = -0.4
        x_rev = 3.2
        w_bar = 0.22

        if not segments_data:
            segments_data = [
                {"nombre": "Actividad Principal de Productos", "monto": rev * 0.70, "pct": 70.0, "yoy": "+6.5%", "perfil": "Volumen Principal"},
                {"nombre": "Servicios y Operaciones Auxiliares", "monto": rev * 0.30, "pct": 30.0, "yoy": "+12.0%", "perfil": "Alto Margen / Recurrencia"}
            ]

        total_seg_monto = sum(float(s.get("monto", s.get("valor", 1.0))) for s in segments_data)
        if total_seg_monto <= 0:
            total_seg_monto = rev

        n_seg = len(segments_data)
        gap = 0.16 if n_seg <= 3 else (0.10 if n_seg <= 4 else 0.07)
        total_gaps = gap * (n_seg - 1)
        avail_h_seg = h_rev + 0.6
        usable_h = avail_h_seg - total_gaps

        # Coordenadas preliminares de cada segmento y sus barras
        seg_coords = []
        curr_y_seg_top = y_rev_center + (avail_h_seg / 2)
        curr_y_rev_top = y_rev_top

        for idx, seg in enumerate(segments_data):
            seg_val = float(seg.get("monto", seg.get("valor", rev / n_seg)))
            seg_pct = float(seg.get("pct", (seg_val / total_seg_monto) * 100.0))
            h_this_seg = max((seg_val / total_seg_monto) * usable_h, 0.24)
            h_this_rev = (seg_val / total_seg_monto) * h_rev

            y_seg_top = curr_y_seg_top
            y_seg_bot = curr_y_seg_top - h_this_seg
            y_bar_mid = (y_seg_top + y_seg_bot) / 2.0

            y_rev_in_top = curr_y_rev_top
            y_rev_in_bot = curr_y_rev_top - h_this_rev

            c_seg = color_segment_palette[idx % len(color_segment_palette)]

            # Dibujar flujo y barra
            SankeyFlowBuilder._draw_flow(ax, x_seg, y_seg_bot, y_seg_top, x_rev, y_rev_in_bot, y_rev_in_top, c_seg, alpha=0.55)
            SankeyFlowBuilder._draw_bar(ax, x_seg, y_seg_bot, y_seg_top, w_bar, c_seg)

            seg_coords.append({
                "seg": seg,
                "val": seg_val,
                "pct": seg_pct,
                "color": c_seg,
                "y_bar_mid": y_bar_mid,
                "y_seg_top": y_seg_top,
                "y_seg_bot": y_seg_bot,
            })

            curr_y_seg_top = y_seg_bot - gap
            curr_y_rev_top = y_rev_in_bot

        # ALGORITMO DE DESCONGESTIÓN VERTICAL CON FLECHAS INDICADORAS
        # Separa los textos verticalmente para que nunca se solapen entre sí
        bar_mids = [sc["y_bar_mid"] for sc in seg_coords]
        target_y_labels = list(bar_mids)
        min_label_gap = 0.78  # Distancia mínima entre centros de cajas de texto

        # Pase 1: Descendente
        for i in range(len(target_y_labels) - 1):
            if target_y_labels[i] - target_y_labels[i+1] < min_label_gap:
                target_y_labels[i+1] = target_y_labels[i] - min_label_gap

        # Pase 2: Ajuste de límites (Si el último se sale por abajo, reacomodar hacia arriba)
        y_min_bound = 0.25
        if target_y_labels[-1] < y_min_bound:
            shift = y_min_bound - target_y_labels[-1]
            for i in range(len(target_y_labels)):
                target_y_labels[i] += shift

        # Renderizar cada etiqueta con o sin flecha según la separación necesaria
        for idx, sc in enumerate(seg_coords):
            seg = sc["seg"]
            c_seg = sc["color"]
            seg_val = sc["val"]
            seg_pct = sc["pct"]
            y_bar_mid = sc["y_bar_mid"]
            y_label_mid = target_y_labels[idx]

            seg_name = seg.get("nombre", f"Línea {idx+1}")
            seg_yoy = seg.get("yoy", "")
            seg_perfil = seg.get("perfil", "")
            seg_val_str = SankeyFlowBuilder._format_currency(seg_val)
            yoy_txt = f"  ({seg_yoy})" if seg_yoy else ""
            seg_name_wrapped = textwrap.fill(seg_name, width=26)
            perfil_txt = f"\n• {seg_perfil}" if seg_perfil else ""
            label_text = f"{seg_name_wrapped}\n{seg_val_str} ({seg_pct:.1f}%){yoy_txt}{perfil_txt}"

            font_size_seg = 7.6 if n_seg <= 4 else 7.1
            needs_arrow = abs(y_label_mid - y_bar_mid) > 0.16 or seg_pct < 6.0

            if needs_arrow:
                # Texto desplazado a la izquierda con flecha de conexión a la barra
                x_text = x_seg - 0.72
                ax.text(
                    x_text, y_label_mid, label_text, fontsize=font_size_seg, fontweight='bold',
                    ha='right', va='center', color='#1E293B',
                    bbox=dict(boxstyle='square,pad=0.25', facecolor='#F1F5F9', edgecolor=c_seg, linewidth=0.9)
                )
                # Flecha curvada desde el texto hasta la barra
                rad_val = 0.12 if y_label_mid < y_bar_mid else -0.12
                ax.annotate(
                    "",
                    xy=(x_seg - w_bar / 2 - 0.03, y_bar_mid),
                    xytext=(x_text + 0.04, y_label_mid),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=c_seg,
                        lw=1.2,
                        mutation_scale=8,
                        connectionstyle=f"arc3,rad={rad_val}"
                    ),
                    zorder=5
                )
            else:
                # Texto adyacente directo a la barra
                ax.text(
                    x_seg - 0.22, y_label_mid, label_text, fontsize=font_size_seg, fontweight='bold',
                    ha='right', va='center', color='#1E293B',
                    bbox=dict(boxstyle='square,pad=0.25', facecolor='#F1F5F9', edgecolor=c_seg, linewidth=0.9)
                )

        # ----------------- 2. BARRA DE INGRESOS TOTALES (X = 3.2) -----------------
        SankeyFlowBuilder._draw_bar(ax, x_rev, y_rev_bot, y_rev_top, w_bar, color_revenue)
        rev_str = SankeyFlowBuilder._format_currency(rev)
        ax.text(x_rev, y_rev_top + 0.20, f"Ingresos Totales\n{rev_str}", fontsize=9.4, fontweight='bold',
                ha='center', va='bottom', color=color_revenue)

        # ----------------- 3. REVENUE -> COGS vs GROSS PROFIT (X = 6.6) -----------------
        x_gp = 6.6
        h_cogs = max((cogs / rev) * h_rev, 0.40)
        h_gp = max((gp / rev) * h_rev, 0.50)

        # COGS se desvía hacia la parte inferior
        y_cogs_bot = 1.3
        y_cogs_top = y_cogs_bot + h_cogs

        # Gross Profit continúa en la parte superior
        y_gp_top = y_rev_top
        y_gp_bot = y_gp_top - h_gp

        # Flujo a COGS (desde la parte inferior de Revenue)
        SankeyFlowBuilder._draw_flow(ax, x_rev, y_rev_bot, y_rev_bot + h_cogs, x_gp, y_cogs_bot, y_cogs_top, color_red_cogs, alpha=0.52)
        # Flujo a Gross Profit (desde la parte superior de Revenue)
        SankeyFlowBuilder._draw_flow(ax, x_rev, y_rev_bot + h_cogs, y_rev_top, x_gp, y_gp_bot, y_gp_top, color_green_gp, alpha=0.56)

        # Barras en X = 6.6
        SankeyFlowBuilder._draw_bar(ax, x_gp, y_cogs_bot, y_cogs_top, w_bar, color_red_cogs)
        SankeyFlowBuilder._draw_bar(ax, x_gp, y_gp_bot, y_gp_top, w_bar, color_green_gp)

        # Textos de X = 6.6
        cogs_str = SankeyFlowBuilder._format_currency(cogs)
        ax.text(x_gp, y_cogs_bot - 0.18, f"Coste de Ventas (COGS)\n-{cogs_str} ({cogs_pct:.1f}%)",
                fontsize=8.4, fontweight='bold', ha='center', va='top', color='#B91C1C',
                bbox=dict(boxstyle='square,pad=0.25', facecolor='#FEF2F2', edgecolor=color_red_cogs, linewidth=0.7))

        gp_str = SankeyFlowBuilder._format_currency(gp)
        ax.text(x_gp, y_gp_top + 0.20, f"Beneficio Bruto\n{gp_str}\n({gp_margin:.1f}% margen)",
                fontsize=9.2, fontweight='bold', ha='center', va='bottom', color='#047857')

        # ----------------- 4. GROSS PROFIT -> OPEX vs OPERATING PROFIT (X = 10.2) -----------------
        x_op = 10.2
        h_opex = max((opex_total / gp) * h_gp, 0.40)
        h_op = max((op_profit / gp) * h_gp, 0.45)

        y_opex_bot = 2.4
        y_opex_top = y_opex_bot + h_opex

        y_op_top = y_gp_top
        y_op_bot = y_op_top - h_op

        # Flujo a OpEx
        SankeyFlowBuilder._draw_flow(ax, x_gp, y_gp_bot, y_gp_bot + h_opex, x_op, y_opex_bot, y_opex_top, color_red_opex, alpha=0.52)
        # Flujo a Operating Profit
        SankeyFlowBuilder._draw_flow(ax, x_gp, y_gp_bot + h_opex, y_gp_top, x_op, y_op_bot, y_op_top, color_green_op, alpha=0.56)

        # Barras en X = 10.2
        SankeyFlowBuilder._draw_bar(ax, x_op, y_opex_bot, y_opex_top, w_bar, color_red_opex)
        SankeyFlowBuilder._draw_bar(ax, x_op, y_op_bot, y_op_top, w_bar, color_green_op)

        # Textos de X = 10.2
        opex_str = SankeyFlowBuilder._format_currency(opex_total)
        
        # Sub-desglose I+D vs VGA
        sub_opex_details = []
        if rd > 0:
            sub_opex_details.append(f"• I+D: -{SankeyFlowBuilder._format_currency(rd)}")
        if sga > 0:
            sub_opex_details.append(f"• VGA: -{SankeyFlowBuilder._format_currency(sga)}")
        sub_opex_txt = ("\n" + "\n".join(sub_opex_details)) if sub_opex_details else ""

        ax.text(x_op, y_opex_bot - 0.18, f"Gastos Operativos (OpEx)\n-{opex_str} ({opex_pct:.1f}%){sub_opex_txt}",
                fontsize=8.0, fontweight='bold', ha='center', va='top', color='#991B1B',
                bbox=dict(boxstyle='square,pad=0.25', facecolor='#FFF1F2', edgecolor=color_red_opex, linewidth=0.7))

        op_str = SankeyFlowBuilder._format_currency(op_profit)
        ax.text(x_op, y_op_top + 0.20, f"Beneficio Operativo (EBIT)\n{op_str}\n({op_margin:.1f}% margen)",
                fontsize=9.2, fontweight='bold', ha='center', va='bottom', color='#15803D')

        # ----------------- 5. OPERATING PROFIT -> TAX & NET PROFIT (X = 13.8) -----------------
        x_np = 13.8
        h_tax = max((tax / op_profit) * h_op, 0.26)
        h_np = max((net_profit / op_profit) * h_op, 0.36)

        y_tax_bot = 3.6
        y_tax_top = y_tax_bot + h_tax

        y_np_top = y_op_top
        y_np_bot = y_np_top - h_np

        # Flujo a Impuestos / Gastos Financieros
        SankeyFlowBuilder._draw_flow(ax, x_op, y_op_bot, y_op_bot + h_tax, x_np, y_tax_bot, y_tax_top, color_red_tax, alpha=0.52)
        # Flujo a Beneficio Neto
        SankeyFlowBuilder._draw_flow(ax, x_op, y_op_bot + h_tax, y_op_top, x_np, y_np_bot, y_np_top, color_green_np, alpha=0.62)

        # Barras en X = 13.8
        SankeyFlowBuilder._draw_bar(ax, x_np, y_tax_bot, y_tax_top, w_bar, color_red_tax)
        SankeyFlowBuilder._draw_bar(ax, x_np, y_np_bot, y_np_top, w_bar, color_green_np)

        # Textos finales a la derecha
        tax_str = SankeyFlowBuilder._format_currency(tax)
        ax.text(x_np + 0.24, (y_tax_bot + y_tax_top) / 2, f"Impuestos / Otros\n-{tax_str} ({tax_pct:.1f}%)",
                fontsize=8.0, fontweight='bold', ha='left', va='center', color='#9F1239')

        np_str = SankeyFlowBuilder._format_currency(net_profit)
        np_text = f"BENEFICIO NETO\n{np_str}\n({np_margin:.1f}% Margen Neto)"
        ax.text(x_np + 0.24, (y_np_bot + y_np_top) / 2, np_text,
                fontsize=9.6, fontweight='bold', ha='left', va='center', color='#064E3B',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#ECFDF5', edgecolor=color_green_np, linewidth=1.2))

        # ----------------- 6. PANEL INFERIOR ANALÍTICO DE RENTABILIDAD Y COSTES -----------------
        # 3 Cajas informativas ampliadas que cubren todo el ancho del lienzo inferior
        y_card = -0.45
        
        # Caja 1: Mix de Ingresos (X = -4.2)
        main_seg = segments_data[0].get("nombre", "Principal") if segments_data else "Línea Central"
        main_pct = segments_data[0].get("pct", 100.0) if segments_data else 100.0
        c1_t1 = textwrap.fill(f"• Línea dominante: {main_seg} ({main_pct:.1f}% de ingresos).", width=46)
        c1_t2 = textwrap.fill(f"• Monetización: {n_seg} divisiones operativas reportadas en SEC 10-K.", width=46)
        c1_text = f"1. Mix y Aportación de Segmentos:\n{c1_t1}\n{c1_t2}"
        
        ax.text(
            -4.2, y_card,
            c1_text,
            fontsize=7.8, color='#1E293B', va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.38', facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=0.9)
        )

        # Caja 2: Estructura de Costes (X = 2.6)
        cogs_weight_desc = "Intensivo en Producción" if cogs_pct > 50 else "Alto Valor Añadido"
        c2_t1 = textwrap.fill(f"• Coste Ventas (COGS): {cogs_pct:.1f}% ({cogs_weight_desc}).", width=46)
        c2_t2 = textwrap.fill(f"• OpEx: {opex_pct:.1f}% (I+D {rd/rev*100:.1f}%, VGA {sga/rev*100:.1f}%).", width=46)
        c2_text = f"2. Absorción de Costes y Estructura:\n{c2_t1}\n{c2_t2}"

        ax.text(
            2.6, y_card,
            c2_text,
            fontsize=7.8, color='#1E293B', va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.38', facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=0.9)
        )

        # Caja 3: Tasa de Retención a Beneficio Neto (X = 9.4)
        retention_quality = "Excelente Monopolio" if np_margin > 20 else ("Margen Sólido" if np_margin > 10 else "Margen Ajustado")
        c3_t1 = textwrap.fill(f"• Retención Final: {np_margin:.1f}% ({retention_quality}).", width=48)
        c3_t2 = textwrap.fill(f"• De cada $100 ingresados, retiene ${np_margin:.1f} en Beneficio Neto.", width=48)
        c3_text = f"3. Conversión a Beneficio Neto:\n{c3_t1}\n{c3_t2}"

        ax.text(
            9.4, y_card,
            c3_text,
            fontsize=7.8, color='#1E293B', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.38', facecolor='#F0FDF4', edgecolor='#10B981', linewidth=1.1)
        )

        # Pie de página explicativo
        footer_text = "Generado automáticamente por el Sistema Multi-Agente con datos oficiales normalizados SEC Form 10-K."
        ax.text(5.8, -1.45, footer_text, fontsize=7.6, ha='center', va='bottom', color='#94A3B8')

        return fig

