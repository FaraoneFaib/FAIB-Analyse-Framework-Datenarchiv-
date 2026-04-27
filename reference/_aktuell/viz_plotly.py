"""
FAIB NEXUS — viz_plotly.py
==========================
Interaktiver Plotly-Chart fuer FAIB NEXUS.
Alle wichtigen Parameter sind ganz oben konfigurierbar.
"""

# ============================================================
# PARAMETER — hier anpassen
# ============================================================

# Maximale Datenpunkte im Plot
# None = alle Punkte (empfohlen fuer kleine Datensaetze)
# 2000 = fuer grosse Datensaetze (MNQ 12 Mio Ticks)
# 5000 = fuer mittlere Datensaetze (ERA5)
VP_MAX_POINTS = 5000

# Optionaler Zeitraumfilter (ISO-String), z.B. "2020-01-01"
# None = keine zeitliche Begrenzung
VP_TIME_START = None
VP_TIME_END = None

# Vibrations-Intensitaet: rollierende Fenstergroesse
VP_VIB_WINDOW = 200     # Ticks

# Browser automatisch oeffnen nach Generierung
VP_OPEN_BROWSER = True

# Plot-Hoehe je Panel in Pixeln
VP_ROW_HEIGHT = 120     # MACD-Panels
VP_TOP_HEIGHT = 300     # cum_height Panel (oben)
VP_VIB_HEIGHT = 150     # Vibrations-Panel (unten)

# Farben
VP_COLOR_EXPANSION   = "#00aaff"   # blau = Expansion
VP_COLOR_COMPRESSION = "#ffcc00"   # gelb = Kompression
VP_COLOR_LONG        = "#00ff88"   # gruen = Long
VP_COLOR_SHORT       = "#ff4444"   # rot = Short
VP_COLOR_CUM_HEIGHT  = "#ffffff"   # weiss = cum_height

# ============================================================

from pathlib import Path
import numpy as np
import json
from utils.viz_utils import apply_time_filter, safe_sample, select_active_fractal_levels

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("Plotly nicht installiert. Bitte: pip install plotly")

def plot_plotly(name, ur, fractals, nexus_df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Zeitbereich optional begrenzen (falls Zeitachse vorhanden).
    ur_filtered = apply_time_filter(ur, start=VP_TIME_START, end=VP_TIME_END) if ur is not None else None
    fractals_filtered = {
        F: apply_time_filter(dfF, start=VP_TIME_START, end=VP_TIME_END)
        for F, dfF in fractals.items()
    }
    # Leere Fraktale nach Filterung verwerfen.
    fractals_filtered = {F: dfF for F, dfF in fractals_filtered.items() if not dfF.empty}

    # Aktive Fraktale anhand zentralisierter Hilfslogik bestimmen.
    panel_levels = select_active_fractal_levels(fractals_filtered, log_prefix="[viz_plotly]")

    if not panel_levels:
        print("[viz_plotly] Keine aktiven Fraktale!")
        return

    print(f"[viz_plotly] Aktive Fraktale: {panel_levels}")

    # ── Sampling ────────────────────────────────────────────
    sampled = {}
    index_maps = {}
    min_len = None

    for F in panel_levels:
        df_full = fractals_filtered[F].copy().reset_index(drop=True)
        df_s = safe_sample(df_full, max_points=VP_MAX_POINTS)
        sampled[F] = df_s
        index_maps[F] = df_s.index.to_numpy()
        min_len = len(df_s) if min_len is None else min(min_len, len(df_s))

    if not min_len:
        return

    for F in panel_levels:
        sampled[F] = sampled[F].iloc[:min_len].copy().reset_index(drop=True)
        index_maps[F] = index_maps[F][:min_len]

    n_levels = len(panel_levels)
    x = list(range(min_len))

    # Vibrations-Intensitaet: rollierendes Fenster pro Tick
    VIB_WINDOW = VP_VIB_WINDOW  # letzte Ticks aus zentralem Viz-Parameter

    def calc_vib_array(F):
        # close_F aus sampled[F] — bereits auf min_len gesampelt
        df_s = sampled[F]
        if "close_F" not in df_s.columns:
            return [0.0] * min_len

        # cum_height aus ur — auf min_len samplen
        if ur_filtered is not None and "cum_height" in ur_filtered.columns:
            ur_r = ur_filtered.reset_index(drop=True)
            if len(ur_r) > min_len:
                step = max(1, len(ur_r) // min_len)
                ur_s = ur_r.iloc[::step].iloc[:min_len].reset_index(drop=True)
            else:
                ur_s = ur_r.iloc[:min_len].reset_index(drop=True)
            cum_h = ur_s["cum_height"].fillna(0.0).to_numpy()
        else:
            # Fallback: dot-basiert
            if "dot" not in df_s.columns:
                return [0.0] * min_len
            dot   = df_s["dot"].fillna(0.0).to_numpy()
            cl_f  = df_s["close_F"].fillna(0.0).to_numpy()
            state = (dot >= float(F) / 2.0).astype(int)
            wechsel = np.abs(np.diff(state, prepend=state[0])).astype(float)
            kernel  = np.ones(VIB_WINDOW) / VIB_WINDOW
            vib_arr = np.convolve(wechsel, kernel, mode="same")
            return [round(float(v), 3) for v in vib_arr]

        cl_f  = df_s["close_F"].fillna(0.0).to_numpy()
        n     = min(len(cum_h), len(cl_f))
        mitte = cl_f[:n] + float(F) / 2.0
        state = (cum_h[:n] > mitte).astype(int)
        wechsel = np.abs(np.diff(state, prepend=state[0])).astype(float)
        kernel  = np.ones(VIB_WINDOW) / VIB_WINDOW
        vib_arr = np.convolve(wechsel, kernel, mode="same")
        return [round(float(v), 3) for v in vib_arr]


    # ── Daten vorbereiten ───────────────────────────────────
    data = {}
    for F in panel_levels:
        df = sampled[F]
        c_macd   = df["c_macd"].fillna(0.0).to_numpy()    if "c_macd"    in df.columns else np.zeros(min_len)
        c_signal = df["c_signal"].fillna(0.0).to_numpy()  if "c_signal"  in df.columns else np.zeros(min_len)
        c_hist   = df["c_hist"].fillna(0.0).to_numpy()    if "c_hist"    in df.columns else np.zeros(min_len)
        cum_h    = df["cum_height"].fillna(0.0).to_numpy() if "cum_height" in df.columns else np.zeros(min_len)
        dom = np.where(c_macd >= c_signal, 1, 0)
        exp = np.where(c_macd >= 0, 1, 0)
        data[F] = dict(
            macd=c_macd.tolist(), signal=c_signal.tolist(),
            hist=c_hist.tolist(), cum_h=cum_h.tolist(),
            dom=dom.tolist(), exp=exp.tolist()
        )

    # ── Farben ──────────────────────────────────────────────
    color_cycle = ["#2ecc71","#e67e22","#9b59b6","#e74c3c",
                   "#795548","#f06292","#00bcd4","#8bc34a"]
    level_colors = {F: color_cycle[i % len(color_cycle)] for i, F in enumerate(panel_levels)}

    # ── Subplot-Struktur ────────────────────────────────────
    n_rows = 1 + n_levels + 3
    row_heights = [1.8] + [1.0] * n_levels + [0.8, 0.8, 1.2]

    subplot_titles = (
        [f"cum_height Referenzebenen ({name})"] +
        [f"Struktur-MACD F={F}" for F in panel_levels] +
        ["Binaere Dominanzmatrix (gruen=Long, rot=Short)"] +
        ["Expansion (orange) / Kompression (blau)"] +
        ["Vibrations-Intensitaet je Fraktal (rollierend 200 Ticks)"]
    )

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── ROW 1: cum_height (aus ur) + close_F Treppen je Fraktal
    # cum_height — schwarze Basislinie aus ur DataFrame
    if ur_filtered is not None and "cum_height" in ur_filtered.columns:
        # Timestamp aus Index retten (drop=False behaelt ihn als Spalte)
        ur_reset = ur_filtered.reset_index(drop=False)
        ts_col = ur_reset.columns[0]  # erster Spaltenname = Timestamp
        if len(ur_reset) > min_len:
            step = max(1, len(ur_reset) // min_len)
            ur_sampled = ur_reset.iloc[::step].iloc[:min_len]
        else:
            ur_sampled = ur_reset.iloc[:min_len]
        cum_h_vals = ur_sampled["cum_height"].fillna(0.0).tolist()
        # Datum fuer Status-Panel merken (aber x numerisch lassen!)
        try:
            ts_vals = ur_sampled[ts_col].tolist()
            if hasattr(ts_vals[0], "strftime"):
                x_dates_str = [str(t) for t in ts_vals]
            else:
                x_dates_str = []
        except Exception:
            x_dates_str = []
        x_ur = list(range(len(cum_h_vals)))
    else:
        cum_h_vals = [0.0] * min_len
        x_ur = x

    cum_h_ticks = [round(v * 4, 2) for v in cum_h_vals]

    fig.add_trace(go.Scatter(
        x=x_ur, y=cum_h_vals,
        customdata=cum_h_ticks,
        name="cum_height",
        line=dict(color="#ffffff", width=1.5),
        opacity=0.9,
        showlegend=True,
        hovertemplate="<b>cum_height: %{y:.2f} (%{customdata:.2f} Ticks)</b><extra></extra>",
    ), row=1, col=1)

    # close_F Treppen je Fraktal — wie in viz_panels
    for F in panel_levels:
        df = sampled[F]
        col = level_colors[F]
        if "close_F" in df.columns:
            y_vals = df["close_F"].fillna(0.0).tolist()
        else:
            y_vals = [0.0] * min_len

        fig.add_trace(go.Scatter(
            x=x, y=y_vals,
            name=f"close_F F={F}",
            line=dict(color=col, width=1.0, shape="hv"),
            opacity=0.75,
            legendgroup=f"F{F}",
            showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)

    # ── ROWS 2..n+1: MACD je Fraktal ────────────────────────
    # Scatter statt Bar — kein komischer Block-Effekt
    for row_i, F in enumerate(panel_levels, start=2):
        d = data[F]
        hist = np.array(d["hist"])
        hist_pos = np.where(hist >= 0, hist, 0.0)
        hist_neg = np.where(hist <  0, hist, 0.0)

        # Positives Histogramm als gefuellte Flaeche
        fig.add_trace(go.Scatter(
            x=x, y=hist_pos.tolist(),
            fill="tozeroy",
            fillcolor="rgba(31,122,31,0.35)",
            line=dict(width=0),
            showlegend=False, name="c_hist+",
        ), row=row_i, col=1)

        # Negatives Histogramm als gefuellte Flaeche
        fig.add_trace(go.Scatter(
            x=x, y=hist_neg.tolist(),
            fill="tozeroy",
            fillcolor="rgba(178,34,34,0.35)",
            line=dict(width=0),
            showlegend=False, name="c_hist-",
        ), row=row_i, col=1)

        # MACD Linie
        fig.add_trace(go.Scatter(
            x=x, y=d["macd"],
            name="c_macd",
            line=dict(color="#4488ff", width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ), row=row_i, col=1)

        # Signal Linie
        fig.add_trace(go.Scatter(
            x=x, y=d["signal"],
            name="c_signal",
            line=dict(color="#ffffff", width=0.9),
            opacity=0.8,
            showlegend=False,
            hoverinfo="skip",
        ), row=row_i, col=1)

        fig.add_hline(y=0, line_color="#444444", line_width=0.7, row=row_i, col=1)

    # ── ROW n+2: Dominanz-Heatmap — F=4 oben, F=256 unten ───
    dom_row = n_levels + 2
    # Umkehren: panel_levels reversed damit F=4 oben erscheint
    levels_rev = list(reversed(panel_levels))
    bin_matrix = np.array([data[F]["dom"] for F in levels_rev])

    fig.add_trace(go.Heatmap(
        z=bin_matrix.tolist(),
        x=x,
        y=[f"F={F}" for F in levels_rev],
        colorscale=[[0, "#b22222"], [1, "#1f7a1f"]],
        showscale=False, zmin=0, zmax=1,
        name="Dominanz",
        hovertemplate="Index: %{x}<br>Fraktal: %{y}<extra></extra>",
        xgap=0, ygap=1,
    ), row=dom_row, col=1)

    # ── ROW n+3: Expansion/Kompression — F=4 oben, F=256 unten
    exp_row = n_levels + 3
    comp_matrix = np.array([data[F]["exp"] for F in levels_rev])

    fig.add_trace(go.Heatmap(
        z=comp_matrix.tolist(),
        x=x,
        y=[f"F={F}" for F in levels_rev],
        colorscale=[[0, "#ffd700"], [1, "#1a6faf"]],
        showscale=False, zmin=0, zmax=1,
        name="Expansion",
        hovertemplate="Index: %{x}<br>Fraktal: %{y}<extra></extra>",
        xgap=0, ygap=1,
    ), row=exp_row, col=1)


    # ── VIBRATIONS-PANEL ────────────────────────────────────
    vib_row = n_levels + 4

    vib_colors = ["#ff4444","#ff8800","#ffcc00","#88ff00",
                  "#00ffcc","#00aaff","#aa44ff","#ff44aa"]

    for fi, F in enumerate(panel_levels):
        vib_arr = calc_vib_array(F)
        x_vib   = list(range(len(vib_arr)))
        col_vib = vib_colors[fi % len(vib_colors)]
        fig.add_trace(go.Scatter(
            x=x_vib, y=vib_arr,
            name=f"Vib F={F}",
            line=dict(color=col_vib, width=1.2),
            opacity=0.85,
            hovertemplate=f"F={int(F)}: %{{y:.3f}}<extra></extra>",
            legendgroup=f"vib_F{F}",
            showlegend=True,
        ), row=vib_row, col=1)

    fig.add_hline(y=0, line_color="#444444", line_width=0.5,
                  row=vib_row, col=1)

    # Fix total_height
    # ── Layout ───────────────────────────────────────────────
    total_height = 280 + n_levels * 180 + 280 + 220

    fig.update_layout(
        height=total_height,
        title=dict(
            text=f"FAIB NEXUS — {name} | Zoom & Scroll synchron | Maus = Crosshair",
            font=dict(size=13, color="#cccccc"), x=0.5,
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#111111",
        font=dict(color="#aaaaaa", size=10),
        legend=dict(
            bgcolor="#1a1a2e", bordercolor="#333333", borderwidth=1,
            font=dict(size=9), orientation="h", y=1.01, x=0,
        ),
        hovermode="x unified",
        barmode="overlay",
        margin=dict(l=60, r=320, t=80, b=40),
    )

    # Spike-Linien (Crosshair) auf allen x-Achsen
    spike_style = dict(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#ffff00", spikethickness=1.5, spikedash="dash",
    )

    for i in range(1, n_rows + 1):
        axis_key = f"xaxis{i if i > 1 else ''}"
        fig.update_layout(**{axis_key: dict(
            showgrid=True, gridcolor="#1e1e1e", gridwidth=0.5,
            zeroline=False, rangeslider=dict(visible=False),
            type="linear", **spike_style,
        )})

    for i in range(1, n_rows + 1):
        axis_key = f"yaxis{i if i > 1 else ''}"
        fig.update_layout(**{axis_key: dict(
            showgrid=True, gridcolor="#1e1e1e", gridwidth=0.5,
            zeroline=False,
        )})

    for annotation in fig.layout.annotations:
        annotation.font.color = "#666666"
        annotation.font.size = 9

    # Kein Rangeslider — sauberere Darstellung
    fig.update_layout(xaxis=dict(
        rangeslider=dict(visible=False)
    ))

    # ── JavaScript Status-Panel ──────────────────────────────
    js_data = json.dumps({
        str(F): {
            "dom":     data[F]["dom"],
            "exp":     data[F]["exp"],
            "macd":    [round(v, 2) for v in data[F]["macd"]],
            "signal":  [round(v, 2) for v in data[F]["signal"]],
            "vib_arr": calc_vib_array(F),
        }
        for F in panel_levels
    })
    js_levels = json.dumps([str(F) for F in panel_levels])

    # cum_height direkt aus erstem Fraktal — für Y-Anzeige
    F_first = panel_levels[0]
    cum_h_for_js = sampled[F_first]["cum_height"].fillna(0.0).round(2).tolist()         if "cum_height" in sampled[F_first].columns else [0.0] * min_len
    js_cumh  = json.dumps(cum_h_for_js)
    # X-Achse Timestamps fuer JS Status-Panel
    js_x0 = json.dumps(x_dates_str if 'x_dates_str' in dir() else [])

    status_html = f"""
<div id="faib-status" style="
    position:fixed; top:60px; right:10px; width:320px;
    background:#0a0a0a; border:1px solid #333; border-radius:8px;
    padding:12px; font-family:monospace; font-size:12px; color:#aaa;
    z-index:9999; box-shadow:0 0 20px rgba(0,0,0,0.8);
    max-height:90vh; overflow-y:auto;
">
    <div style="color:#4488ff;font-weight:bold;font-size:14px;text-align:center;margin-bottom:8px;">
        ◈ FAIB STATUS
    </div>
    <div id="faib-index" style="color:#ffff00;text-align:center;margin-bottom:2px;font-size:13px;">
        Index: —
    </div>


    <hr style="border-color:#333;margin:6px 0;">
    <div style="display:grid;grid-template-columns:60px 1fr 1fr;gap:4px;font-size:10px;color:#555;margin-bottom:4px;">
        <span>Fraktal</span>
        <span style="text-align:center;">Dominanz</span>
        <span style="text-align:center;">Exp/Komp</span>
    </div>
    <div id="faib-rows"></div>
    <hr style="border-color:#333;margin:8px 0;">

    <hr style="border-color:#333;margin:8px 0;">
    <div id="faib-macd-vals" style="font-size:10px;"></div>
</div>

<script>
(function(){{
    var faibData   = {js_data};
    var faibLevels = {js_levels};
    var cumHeight  = {js_cumh};
    var x0arr      = {js_x0};

    function updateStatus(xi){{
        var maxIdx = faibData[faibLevels[0]].dom.length - 1;
        var idx = Math.max(0, Math.min(Math.round(xi), maxIdx));
        // Datum anzeigen wenn x-Achse Timestamps hat
        var xVal = (typeof x0arr !== 'undefined' && x0arr && x0arr[idx])
                   ? x0arr[idx] : ('Index: ' + idx);
        document.getElementById('faib-index').textContent = xVal;


        var rowsHtml = '';
        var macdHtml = '';

        faibLevels.forEach(function(F){{
            var d   = faibData[F];
            var dom = d.dom[idx];
            var exp = d.exp[idx];
            var mv  = d.macd[idx];
            var sv  = d.signal[idx];
            var dv  = (mv - sv).toFixed(2);

            var combKey = (dom===1?'L':'S') + (exp===1?'E':'C');
            var combMap = {{
                'LE': ['#00cc00',   'LONG ▲▲'],
                'LC': ['#1f5a1f',   'LONG ▲○'],
                'SE': ['#ff0000',   'SHORT ▼▼'],   // hellrot = Short Expansion
                'SC': ['#7a1f1f',   'SHORT ▼○']    // dunkelrot = Short Kompression
            }};
            var domColor = combMap[combKey][0];
            var domLabel = combMap[combKey][1];
            var expColor = exp===1 ? '#1a6faf' : '#ffd700';
            var expLabel = exp===1 ? 'EXP ↑'   : 'COMP ↓';
            var dvColor  = parseFloat(dv)>=0 ? '#1f7a1f' : '#b22222';

            // Vibration aus rolierendem Array — aendert sich mit Mausposition!
            var vibArr  = d.vib_arr !== undefined ? d.vib_arr : [];
            var vibIdx  = Math.max(0, Math.min(idx, vibArr.length - 1));
            var vib     = vibArr.length > 0 ? (vibArr[vibIdx] || 0) : 0;
            var vibPct  = Math.round(vib * 100);
            var vibW    = Math.round(vib * 300);
            var vibCol  = vib >= 0.25 ? '#00aaff' : vib >= 0.15 ? '#0066cc' : '#003366';
            // Resonantestes Fraktal = hoechste Vibration am aktuellen Index
            var allVibs = faibLevels.map(function(lv){{
                var arr = faibData[lv] && faibData[lv].vib_arr ? faibData[lv].vib_arr : [];
                var vi  = Math.max(0, Math.min(idx, arr.length - 1));
                return arr.length > 0 ? (arr[vi] || 0) : 0;
            }});
            var maxVib  = Math.max.apply(null, allVibs);
            var vibStar = (maxVib > 0 && Math.abs(vib - maxVib) < 0.001) ? ' ★' : '';

            rowsHtml +=
                '<div style="display:grid;grid-template-columns:60px 1fr 1fr;gap:4px;margin-bottom:2px;align-items:center;">' +
                '<span style="color:#aaa;font-weight:bold;">F=' + F + '</span>' +
                '<span style="background:' + domColor + ';color:#fff;text-align:center;border-radius:4px;padding:2px 0;font-size:10px;font-weight:bold;">' + domLabel + '</span>' +
                '<span style="background:' + expColor + ';color:#fff;text-align:center;border-radius:4px;padding:2px 0;font-size:10px;font-weight:bold;">' + expLabel + '</span>' +
                '</div>' +
                '<div style="display:grid;grid-template-columns:60px 1fr;gap:4px;margin-bottom:4px;align-items:center;">' +
                '<span style="color:#555;font-size:9px;">Impulstärke</span>' +
                '<div style="position:relative;height:8px;background:#1a1a1a;border-radius:4px;overflow:hidden;">' +
                '<div style="position:absolute;left:0;top:0;height:100%;width:' + vibW + 'px;background:' + vibCol + ';border-radius:4px;"></div>' +
                '<span style="position:absolute;right:4px;top:-1px;font-size:8px;color:#aaa;">' + vibPct + '%' + vibStar + '</span>' +
                '</div>' +
                '</div>';

            macdHtml +=
                '<div style="margin-bottom:3px;color:#555;">F' + F +
                ': M=<span style="color:#4488ff;">' + mv.toFixed(1) + '</span>' +
                ' S=<span style="color:#888;">'     + sv.toFixed(1) + '</span>' +
                ' Δ=<span style="color:' + dvColor + ';font-weight:bold;">' + dv + '</span></div>';
        }});

        document.getElementById('faib-rows').innerHTML      = rowsHtml;
        document.getElementById('faib-macd-vals').innerHTML = macdHtml;
    }}

    // Messung: Alt+Klick
    var measureStart = null;

    function showMeasure(idx1, idx2){{
        var ticks = Math.abs(idx2 - idx1);
        var d = document.getElementById('faib-measure');
        if(!d) return;
        d.style.display = 'block';
        d.innerHTML =
            '<div style="color:#ffff00;font-weight:bold;margin-bottom:4px;">📏 MESSUNG</div>' +
            '<div>Von: <span style="color:#4488ff;">' + Math.min(idx1,idx2) + '</span></div>' +
            '<div>Bis: <span style="color:#4488ff;">' + Math.max(idx1,idx2) + '</span></div>' +
            '<div style="color:#ffffff;font-weight:bold;font-size:14px;margin-top:6px;">Ticks: ' + ticks + '</div>' +
            '<div style="color:#555;font-size:10px;margin-top:2px;">Alt+Klick = neu starten</div>';
    }}

    var lastHoverIdx = 0;
    var lastHoverY   = 0;
    var measureStartY = null;

    function attachHover(){{
        var divs = document.querySelectorAll('.plotly-graph-div');
        if(divs.length===0){{ setTimeout(attachHover,500); return; }}
        var mainDiv = divs[0];

        mainDiv.on('plotly_hover', function(ev){{
            if(ev && ev.points && ev.points.length>0){{
                var xi = ev.points[0].x;
                if(xi!==undefined && xi!==null){{
                    lastHoverIdx = Math.round(xi);
                    // Y direkt aus cum_height Array — immer korrekt
                    var cidx = Math.max(0, Math.min(lastHoverIdx, cumHeight.length-1));
                    lastHoverY = cumHeight[cidx];
                    updateStatus(xi);
                    if(measureStart !== null){{
                        var ticks  = Math.abs(lastHoverIdx - measureStart);
                        var dyRaw  = lastHoverY - measureStartY;
                        var dy     = dyRaw.toFixed(2);
                        var dyCol  = dyRaw >= 0 ? '#1f7a1f' : '#b22222';
                        var dySign = dyRaw >= 0 ? '↑' : '↓';
                        var d = document.getElementById('faib-measure');
                        if(d) d.innerHTML =
                            '<div style="color:#1f7a1f;">▶ START: ' + measureStartY.toFixed(2) + '</div>' +
                            '<div style="color:#ffff00;">◉ NOW: &nbsp; ' + lastHoverY.toFixed(2) + '</div>' +
                            '<hr style="border-color:#333;margin:4px 0;">' +
                            '<div style="color:' + dyCol + ';font-weight:bold;font-size:16px;">' + dySign + ' ' + Math.abs(dyRaw).toFixed(2) + ' Ticks</div>';
                    }}
                }}
            }}
        }});

        // Alt+Klick = Messpunkt setzen
        mainDiv.addEventListener('click', function(e){{
            if(!e.altKey) return;
            e.preventDefault();
            e.stopPropagation();
            if(measureStart === null){{
                measureStart = lastHoverIdx;
                var d = document.getElementById('faib-measure');
                if(d){{
                    d.style.display = 'block';
                    d.innerHTML =
                        '<div style="color:#ffff00;font-weight:bold;">📏 MESSUNG</div>' +
                        '<div>Start: <span style="color:#4488ff;">' + measureStart + '</span></div>' +
                        '<div style="color:#888;font-size:10px;">Alt+Klick = Endpunkt</div>';
                }}
            }} else {{
                showMeasure(measureStart, lastHoverIdx);
                measureStart = null;
            }}
        }});
    }}

    // Globale Messfunktionen für Buttons
    window.setMeasureStart = function(){{
        measureStart  = lastHoverIdx;
        measureStartY = lastHoverY;
        var d = document.getElementById('faib-measure');
        if(d){{
            d.style.borderColor = '#1f7a1f';
            d.innerHTML =
                '<div style="color:#1f7a1f;font-weight:bold;">▶ START: ' + measureStart + '</div>' +
                '<div style="color:#888;font-size:10px;">Maus bewegen, dann END drücken</div>';
        }}
        document.getElementById('btn-start').style.background = '#2a5a2a';
    }};

    window.setMeasureEnd = function(){{
        if(measureStart === null){{
            var d = document.getElementById('faib-measure');
            if(d) d.innerHTML = '<div style="color:#b22222;">Zuerst START drücken!</div>';
            return;
        }}
        var ticks  = Math.abs(lastHoverIdx - measureStart);
        var dyRaw  = lastHoverY - measureStartY;
        var dy     = dyRaw.toFixed(2);
        var dyCol  = dyRaw >= 0 ? '#1f7a1f' : '#b22222';
        var dySign = dyRaw >= 0 ? '↑' : '↓';
        var d = document.getElementById('faib-measure');
        if(d){{
            d.style.borderColor = '#4488ff';
            d.innerHTML =
                '<div style="color:#1f7a1f;">▶ START: ' + measureStartY.toFixed(2) + '</div>' +
                '<div style="color:#4488ff;">■ END: &nbsp; ' + lastHoverY.toFixed(2) + '</div>' +
                '<hr style="border-color:#333;margin:4px 0;">' +
                '<div style="color:' + dyCol + ';font-weight:bold;font-size:18px;">' + dySign + ' ' + Math.abs(dyRaw).toFixed(2) + ' Ticks</div>';
        }}
        measureStart = null;
        document.getElementById('btn-start').style.background = '#1a3a1a';
    }};

    window.resetMeasure = function(){{
        measureStart = null;
        var d = document.getElementById('faib-measure');
        if(d){{
            d.style.borderColor = '#333';
            d.innerHTML = '<div style="color:#555;text-align:center;">START drücken, dann Maus bewegen</div>';
        }}
        document.getElementById('btn-start').style.background = '#1a3a1a';
    }};

    setTimeout(attachHover, 1000);
    updateStatus(Math.floor(faibData[faibLevels[0]].dom.length/2));
}})();
</script>
"""

    # HTML ausgeben
    out_path = out_dir / f"{name}_interactive_plotly.html"
    html_str = fig.to_html(
        include_plotlyjs="cdn", full_html=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawrect",
                "eraseshape"
            ],
            "modeBarButtonsToRemove": ["lasso2d","select2d"],
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "FAIB_" + name,
                "width": 1800, "scale": 2
            }
        }
    )
    html_str = html_str.replace("</body>", status_html + "\n</body>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"[viz_plotly] Gespeichert: {out_path}")

    try:
        import webbrowser
        if VP_OPEN_BROWSER:
            webbrowser.open(str(out_path))
    except Exception:
        pass