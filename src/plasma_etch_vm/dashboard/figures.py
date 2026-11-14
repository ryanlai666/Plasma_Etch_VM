"""Plotly figure builders shared by the Streamlit dashboard and ``scripts/render_figures.py``.

Colour roles follow the validated reference palette of the dataviz guidance: categorical slots
1–2 (blue = SF6/etch step or the single series, orange = C4F8/passivation step), an ordinal blue ramp
for slot-in-lot (1→10, starting at step 250 so the lightest mark still clears 2:1 on the surface),
a one-hue sequential blue ramp for etch depth and a blue–grey–red diverging scale for residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE = "#2a78d6", "#eb6834"
MUTED = "#898781"
SURFACE = "#fcfcfb"
ORDINAL_BLUE = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
SLOT_SCALE = [[i / 9, c] for i, c in enumerate(ORDINAL_BLUE)]
SEQ_BLUE = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]
DIVERGING = [[0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#ee9493"], [1, "#c23535"]]
UNITS = {
    "si_etch_mean": "µm",
    "si_nu_halfrange_pct": "%",
    "si_nu_std_pct": "%",
    "si_nu_p90_pct": "%",
    "si_edge_center": "µm",
    "ox_etch_mean": "µm",
    "selectivity": "",
    "si_etch_point": "µm",
}
LABELS = {
    "si_etch_mean": "Si etch depth (wafer mean)",
    "si_nu_halfrange_pct": "Si non-uniformity (half-range)",
    "si_nu_p90_pct": "Si non-uniformity (robust P5–P95)",
    "si_edge_center": "Si edge − centre depth",
    "ox_etch_mean": "SiO₂ mask loss",
    "selectivity": "Si:SiO₂ selectivity",
}


def style(
    fig: go.Figure, title: str = "", x: str = "", y: str = "", height: int = 420, static: bool = False
) -> go.Figure:
    """Common layout. ``static=True`` pins light-mode chrome for exported images; in the dashboard the
    Streamlit theme supplies surface / ink colours (so dark mode follows the viewer's theme)."""
    n_named = sum(1 for t in fig.data if t.showlegend is not False and t.name)
    top = (34 if title else 8) + (30 if n_named >= 2 else 0)
    fig.update_layout(
        title=dict(
            text=title, x=0, xanchor="left", y=1, yanchor="top", yref="container", pad=dict(t=8), font=dict(size=15)
        ),
        height=height,
        margin=dict(l=10, r=10, t=top, b=10),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13),
        hovermode="closest",
        showlegend=n_named >= 2,
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, xanchor="left"),
    )
    fig.update_xaxes(title=x, showgrid=True, zeroline=False)
    fig.update_yaxes(title=y, showgrid=True, zeroline=False)
    if static:
        fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font_color="#0b0b0b", title_font_color="#0b0b0b")
        fig.update_xaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", tickfont_color="#52514e", title_font_color="#52514e")
        fig.update_yaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", tickfont_color="#52514e", title_font_color="#52514e")
    return fig


def pred_vs_measured(d: pd.DataFrame, target: str, static: bool = False) -> go.Figure:
    """Out-of-fold predictions with conformal intervals; colour = slot in lot (ordinal blue)."""
    d = d.dropna(subset=["pred"]).copy()
    d["slot"] = d["wafer_id"].str.slice(11).astype(int)
    u = UNITS.get(target, "")
    lo_all, hi_all = float(min(d["y"].min(), d["pred"].min())), float(max(d["y"].max(), d["pred"].max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lo_all, hi_all],
            y=[lo_all, hi_all],
            mode="lines",
            name="ideal (y = x)",
            line=dict(color=MUTED, width=1, dash="dot"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["y"],
            y=d["pred"],
            mode="markers",
            name="wafer (colour = slot in lot)",
            error_y=dict(
                type="data",
                symmetric=False,
                array=d["hi"] - d["pred"],
                arrayminus=d["pred"] - d["lo"],
                color="rgba(137,135,129,0.35)",
                thickness=1,
                width=0,
            ),
            marker=dict(
                size=9,
                color=d["slot"],
                colorscale=SLOT_SCALE,
                cmin=1,
                cmax=10,
                line=dict(width=1, color=SURFACE),
                colorbar=dict(title="slot", tickvals=[1, 5, 10], thickness=10),
            ),
            customdata=np.c_[d["wafer_id"], d["lot"], d["slot"], d["lo"], d["hi"]],
            hovertemplate="%{customdata[0]}<br>lot %{customdata[1]}, slot %{customdata[2]}<br>measured %{x:.4g}"
            "<br>predicted %{y:.4g}<br>interval [%{customdata[3]:.4g}, %{customdata[4]:.4g}]<extra></extra>",
        )
    )
    n_folds = d["fold"].nunique() if "fold" in d else 0
    return style(
        fig,
        f"{LABELS.get(target, target)} — out-of-fold, {n_folds} folds",
        f"measured [{u}]",
        f"predicted [{u}]",
        480,
        static,
    )


def importance_bar(d: pd.DataFrame, title: str, xlabel: str, static: bool = False) -> go.Figure:
    d = d.iloc[::-1]
    err = None
    if "importance_std" in d and d["importance_std"].notna().any():
        err = dict(type="data", array=d["importance_std"], color=MUTED, thickness=1)
    fig = go.Figure(
        go.Bar(
            x=d["importance_mean"],
            y=d["feature"],
            orientation="h",
            marker_color=BLUE,
            error_x=err,
            hovertemplate="%{y}<br>%{x:.4g}<extra></extra>",
        )
    )
    return style(fig, title, xlabel, "", 520, static)


def feature_scatter(
    x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, static: bool = False, color_by_slot: bool = True
) -> go.Figure:
    """Scatter of one feature vs a target; colour = slot in lot unless the x axis already is the slot."""
    slot = pd.Series(x.index.str.slice(11), index=x.index).astype(int)
    if color_by_slot:
        marker = dict(
            size=9,
            color=slot,
            cmin=1,
            cmax=10,
            colorscale=SLOT_SCALE,
            colorbar=dict(title="slot", thickness=10),
            line=dict(width=1, color=SURFACE),
        )
    else:
        marker = dict(size=9, color=BLUE, line=dict(width=1, color=SURFACE))
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            text=x.index,
            marker=marker,
            hovertemplate="%{text}<br>%{x:.4g} → %{y:.4g}<extra></extra>",
        )
    )
    return style(fig, "", xlabel, ylabel, 380, static)


def spectrum(
    x: np.ndarray,
    spectra: dict[str, np.ndarray],
    marks: list[tuple[str, float]],
    wl_range: tuple[float, float],
    static: bool = False,
) -> go.Figure:
    m = (x >= wl_range[0]) & (x <= wl_range[1])
    fig = go.Figure()
    names = {"etch": ("SF₆ step (etch)", BLUE), "dep": ("C₄F₈ step (passivation)", ORANGE)}
    for ph, s in spectra.items():
        name, col = names.get(ph, (ph, MUTED))
        fig.add_trace(
            go.Scatter(
                x=x[m],
                y=s[m],
                mode="lines",
                name=name,
                line=dict(color=col, width=1.5),
                hovertemplate="%{x:.2f} nm<br>%{y:.0f} counts<extra></extra>",
            )
        )
    for name, wl in marks:
        if wl_range[0] <= wl <= wl_range[1]:
            fig.add_vline(x=wl, line=dict(color=MUTED, width=1, dash="dot"))
            # label inside the plot area, beside its rule, so it never collides with title / legend
            fig.add_annotation(
                x=wl,
                y=1,
                xref="x",
                yref="y domain",
                text=name,
                showarrow=False,
                textangle=-90,
                xanchor="left",
                yanchor="top",
                xshift=2,
                font=dict(size=10, color=MUTED),
            )
    return style(
        fig,
        "Wafer-mean spectrum (calibrated axis, Δλ ≈ +1.22 nm applied)",
        "wavelength [nm]",
        "intensity [counts]",
        460,
        static,
    )


def line_trace(t: np.ndarray, f: np.ndarray, c2: np.ndarray, static: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=f, name="F I 685.6 nm", mode="lines", line=dict(color=BLUE, width=1.5)))
    fig.add_trace(go.Scatter(x=t, y=c2, name="C₂ Swan 516.5 nm", mode="lines", line=dict(color=ORANGE, width=1.5)))
    fig.update_layout(hovermode="x unified")
    return style(
        fig,
        "Line intensities at ~24 Hz — BOSCH switching (4.5 s SF₆ / 1.5 s C₄F₈)",
        "time since recording start [s]",
        "background-subtracted counts",
        360,
        static,
    )


def cycle_drift(g: pd.DataFrame, line: str, static: bool = False) -> go.Figure:
    fig = go.Figure()
    for ph, col in (("etch", BLUE), ("dep", ORANGE)):
        gp = g[g["phase"] == ph]
        fig.add_trace(
            go.Scatter(
                x=gp["cycle"],
                y=gp[f"L_{line}"] / gp["total"] * 1e3,
                mode="lines",
                name=f"{ph} step",
                line=dict(color=col, width=2),
            )
        )
    return style(
        fig, f"{line}: share of total emission per cycle", "BOSCH cycle", "‰ of baseline-corrected total", 320, static
    )


def wafer_map(
    p: pd.DataFrame, values: pd.Series, title: str, scale, crange: tuple[float, float], static: bool = False
) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=p["x_mm"],
            y=p["y_mm"],
            mode="markers",
            marker=dict(
                symbol="square",
                size=17,
                color=values,
                colorscale=scale,
                cmin=crange[0],
                cmax=crange[1],
                colorbar=dict(thickness=8),
            ),
            hovertemplate="x %{x} mm, y %{y} mm<br>%{marker.color:.3f} µm<extra></extra>",
        )
    )
    fig.add_shape(type="circle", x0=-100, y0=-100, x1=100, y1=100, line=dict(color=MUTED, width=1))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return style(fig, title, "x [mm]", "y [mm]", 380, static)
