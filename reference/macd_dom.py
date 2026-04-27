import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 6, slow: int = 26, signal_span: int = 5) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal = ema(macd_line, signal_span)
    hist = macd_line - signal

    return pd.DataFrame(
        {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd": macd_line,
            "signal": signal,
            "hist": hist,
        },
        index=series.index,
    )


def macd_on_change(series: pd.Series,
                   fast: int = 6,
                   slow: int = 26,
                   signal_span: int = 5) -> pd.DataFrame:
    """
    Event-basierter MACD:
    EMA/MACD werden nur dann aktualisiert, wenn sich der Wert von series ändert.
    Dazwischen werden die letzten Werte einfach fortgeschrieben.
    """
    s = series.astype(float).to_numpy()
    n = len(s)
    idx = series.index

    ema_fast_vals = [float("nan")] * n
    ema_slow_vals = [float("nan")] * n
    macd_vals = [float("nan")] * n
    signal_vals = [float("nan")] * n
    hist_vals = [float("nan")] * n

    alpha_fast = 2.0 / (fast + 1.0)
    alpha_slow = 2.0 / (slow + 1.0)
    alpha_sig = 2.0 / (signal_span + 1.0)

    ema_f = None
    ema_s = None
    macd_prev = None
    sig_prev = None

    last_val = None

    for i in range(n):
        x = s[i]
        changed = (last_val is None) or (x != last_val)

        if changed:
            # Initialisierung beim ersten Wert
            if ema_f is None:
                ema_f = x
                ema_s = x
                macd_val = 0.0
                sig_val = 0.0
            else:
                ema_f = alpha_fast * x + (1.0 - alpha_fast) * ema_f
                ema_s = alpha_slow * x + (1.0 - alpha_slow) * ema_s
                macd_val = ema_f - ema_s
                sig_val = alpha_sig * macd_val + (1.0 - alpha_sig) * sig_prev

            hist_val = macd_val - sig_val

            macd_prev = macd_val
            sig_prev = sig_val
        else:
            # kein neuer Close/kein neuer Wert -> alles einfrieren
            macd_val = macd_prev if macd_prev is not None else 0.0
            sig_val = sig_prev if sig_prev is not None else 0.0
            hist_val = macd_val - sig_val

        ema_fast_vals[i] = ema_f if ema_f is not None else x
        ema_slow_vals[i] = ema_s if ema_s is not None else x
        macd_vals[i] = macd_val
        signal_vals[i] = sig_val
        hist_vals[i] = hist_val

        last_val = x

    return pd.DataFrame(
        {
            "ema_fast": pd.Series(ema_fast_vals, index=idx),
            "ema_slow": pd.Series(ema_slow_vals, index=idx),
            "macd": pd.Series(macd_vals, index=idx),
            "signal": pd.Series(signal_vals, index=idx),
            "hist": pd.Series(hist_vals, index=idx),
        },
        index=idx,
    )


def dominance_binary(macd_df: pd.DataFrame) -> pd.Series:
    """Binäre Dominanz: 1 = Long, 0 = Short."""
    return (macd_df["ema_fast"] >= macd_df["ema_slow"]).astype(int)