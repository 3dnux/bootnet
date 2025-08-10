from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
import random
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

__version__ = "0.3.0"


# ----------------------------- Utilidades ------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0:
            return default
        return a / b
    except Exception:
        return default


def to_float(x: Any, default: float = 0.0) -> float:
    """Convierte valores (incluyendo strings como "1,234.56") a float de forma robusta."""
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            if s == "":
                return default
            return float(s)
        return default
    except Exception:
        return default


def fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        absx = abs(x)
        if absx >= 1_000_000_000:
            return f"{x/1_000_000_000:.2f}B"
        if absx >= 1_000_000:
            return f"{x/1_000_000:.2f}M"
        if absx >= 1_000:
            return f"{x/1_000:.2f}K"
        return f"{x:.2f}"
    except Exception:
        return str(x)


def fmt_usd(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"
    except Exception:
        return f"${x}"


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        if not math.isfinite(x):
            return "-"
        return f"{x*100:.2f}%"
    except Exception:
        return "-"


# ----------------------------- Cliente CoinGecko ------------------------------

class CoinGeckoClient:
    def __init__(self, base_url: str = "https://api.coingecko.com/api/v3", timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.user_agent = (
            f"MemecoinBot/{__version__} (+https://www.jetbrains.com) Python-urllib"
        )

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            qs = urllib.parse.urlencode(params, doseq=True)
            url = f"{url}?{qs}"
        last_err = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url)
                req.add_header("Accept", "application/json")
                req.add_header("User-Agent", self.user_agent)
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        status = resp.getcode()
                        data = resp.read()
                        if status == 200:
                            return json.loads(data.decode("utf-8"))
                        elif status in (429, 500, 502, 503, 504):
                            retry_after = resp.headers.get("Retry-After")
                            base_sleep = 1.5 * (2 ** attempt)
                            if retry_after:
                                try:
                                    base_sleep = max(base_sleep, float(retry_after))
                                except Exception:
                                    pass
                            jitter = random.uniform(0.1, 0.5)
                            time.sleep(base_sleep + jitter)
                            continue
                        else:
                            raise RuntimeError(f"HTTP {status} for {url}")
                except urllib.error.HTTPError as he:
                    status = getattr(he, "code", None)
                    if status in (429, 500, 502, 503, 504):
                        retry_after = he.headers.get("Retry-After") if hasattr(he, "headers") else None
                        base_sleep = 1.5 * (2 ** attempt)
                        if retry_after:
                            try:
                                base_sleep = max(base_sleep, float(retry_after))
                            except Exception:
                                pass
                        jitter = random.uniform(0.1, 0.5)
                        time.sleep(base_sleep + jitter)
                        continue
                    raise
            except Exception as e:
                last_err = e
                time.sleep(1.0 + attempt + random.uniform(0.0, 0.25))
        raise RuntimeError(f"Fallo al obtener {url}: {last_err}")

    def categories_list(self) -> List[Dict[str, Any]]:
        return self._get("coins/categories/list")

    def search_trending(self) -> Dict[str, Any]:
        return self._get("search/trending")

    def markets_by_category(self, category_id: str, vs: str = "usd", per_page: int = 50, page: int = 1) -> List[Dict[str, Any]]:
        return self._get(
            "coins/markets",
            {
                "vs_currency": vs,
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
                "price_change_percentage": "1h,24h,7d,30d",
                "category": category_id,
            },
        )

    def markets_by_ids(self, ids: List[str], vs: str = "usd") -> List[Dict[str, Any]]:
        if not ids:
            return []
        ids_param = ",".join(ids)
        return self._get(
            "coins/markets",
            {
                "vs_currency": vs,
                "order": "market_cap_desc",
                "per_page": min(250, max(1, len(ids))),
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "1h,24h,7d,30d",
                "ids": ids_param,
            },
        )

    def community_data(self, coin_id: str) -> Optional[Dict[str, Any]]:
        try:
            data = self._get(
                f"coins/{coin_id}",
                {
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "false",
                    "community_data": "true",
                    "developer_data": "false",
                    "sparkline": "false",
                },
            )
            return data.get("community_data") if isinstance(data, dict) else None
        except Exception:
            return None

    def market_chart(self, coin_id: str, vs: str = "usd", days: int = 90) -> Optional[Dict[str, Any]]:
        try:
            return self._get(
                f"coins/{coin_id}/market_chart",
                {
                    "vs_currency": vs,
                    "days": days,
                    "interval": "daily" if days >= 90 else "hourly",
                },
            )
        except Exception:
            return None

    def coin_tickers(self, coin_id: str, page: int = 1) -> List[Dict[str, Any]]:
        try:
            data = self._get(
                f"coins/{coin_id}/tickers",
                {
                    "page": page,
                },
            )
            if isinstance(data, dict) and isinstance(data.get("tickers"), list):
                return data["tickers"]
            return []
        except Exception:
            return []


# ----------------------------- Indicadores -----------------------------------

def sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    return sum(values[-period:]) / float(period)


def stdev(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 1:
        return None
    subset = values[-period:]
    mean = sum(subset) / period
    var = sum((x - mean) ** 2 for x in subset) / (period - 1)
    return math.sqrt(max(0.0, var))


def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    n = len(closes)
    if n < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    # Wilder smoothing
    for i in range(period + 1, n):
        delta = closes[i] - closes[i - 1]
        gain = max(0.0, delta)
        loss = max(0.0, -delta)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger_bands(closes: List[float], period: int = 20, k: float = 2.0) -> Optional[Tuple[float, float, float]]:
    if len(closes) < period:
        return None
    mid = sma(closes, period)
    sd = stdev(closes, period)
    if mid is None or sd is None:
        return None
    upper = mid + k * sd
    lower = mid - k * sd
    return lower, mid, upper


def pct_changes(closes: List[float], period: int) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    start = closes[-period - 1]
    end = closes[-1]
    if start == 0:
        return None
    return (end - start) / start


def daily_return_volatility(closes: List[float], lookback: int = 20) -> Optional[float]:
    if len(closes) < lookback + 1:
        return None
    rets = []
    for i in range(-lookback, 0):
        prev = closes[i - 1]
        cur = closes[i]
        if prev == 0:
            continue
        rets.append((cur - prev) / prev)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(max(0.0, var))


# ----------------------------- Análisis y Señales ----------------------------

def social_score(comm: Optional[Dict[str, Any]]) -> float:
    if not comm:
        return 0.0
    # Señales disponibles en CoinGecko community_data (valores pueden venir como string)
    tw = to_float(comm.get("twitter_followers"))
    rd = to_float(comm.get("reddit_subscribers"))
    tg = to_float(comm.get("telegram_channel_user_count"))
    fb = to_float(comm.get("facebook_likes"))
    rd_posts = to_float(comm.get("reddit_average_posts_48h"))
    rd_comments = to_float(comm.get("reddit_average_comments_48h"))
    rd_active = to_float(comm.get("reddit_accounts_active_48h"))

    # Alcance: escalado logarítmico suave (10^6 -> ~1.0)
    reach_tw = clamp(math.log10(tw + 1.0) / 6.0, 0.0, 1.0)
    reach_rd = clamp(math.log10(rd + 1.0) / 6.0, 0.0, 1.0)
    reach_tg = clamp(math.log10(max(0.0, tg) + 1.0) / 6.0, 0.0, 1.0) if tg > 0 else 0.0
    reach_fb = clamp(math.log10(max(0.0, fb) + 1.0) / 6.0, 0.0, 1.0) if fb > 0 else 0.0
    reach = clamp(0.55 * reach_tw + 0.30 * reach_rd + 0.10 * reach_tg + 0.05 * reach_fb, 0.0, 1.0)

    # Engagement: comentarios por post, cuentas activas relativas y "buzz" 48h
    eng_ratio = 0.0
    if rd_posts > 0.0:
        c_per_p = rd_comments / max(1.0, rd_posts)
        # 10 comentarios/post ~ 1.0 (cap a 1.0)
        eng_ratio = clamp(c_per_p / 10.0, 0.0, 1.0)
    active_ratio = 0.0
    if rd > 0.0:
        active_ratio = clamp((rd_active / rd) if rd_active > 0 else 0.0, 0.0, 1.0)
    # Suma de publicaciones+comentarios en 48h; 50 eventos ~ 1.0
    buzz = clamp((rd_posts + rd_comments) / 50.0, 0.0, 1.0)

    engagement = clamp(0.50 * eng_ratio + 0.30 * active_ratio + 0.20 * buzz, 0.0, 1.0)

    # Combinación con pequeño bonus cuando alcance y engagement son altos
    base = 0.65 * reach + 0.35 * engagement
    if reach > 0.6 and engagement > 0.6:
        base = clamp(base + 0.05, 0.0, 1.0)
    if reach < 0.1 and engagement < 0.1:
        base = base * 0.5

    return clamp(base, 0.0, 1.0)


def liquidity_score(volume: Optional[float], market_cap: Optional[float]) -> float:
    if not volume or not market_cap or market_cap <= 0:
        return 0.0
    vm = volume / market_cap
    # vm 0.2 -> 1.0, 0.05 -> 0.5, 0.01 -> 0.2, etc.
    return clamp(vm / 0.2, 0.0, 1.0)


def risk_score(volatility: Optional[float]) -> float:
    # Volatilidad diaria de 15% -> riesgoso (0), 5% -> 0.66, 2% -> ~0.93
    if volatility is None:
        return 0.5
    return clamp(1.0 - (volatility / 0.15), 0.0, 1.0)


def momentum_score(rsi_value: Optional[float]) -> float:
    if rsi_value is None:
        return 0.5
    # Mapear RSI a [0,1], máximo alrededor de 65 (tendencia saludable)
    if rsi_value >= 80:
        return 0.2
    if rsi_value >= 70:
        return 0.5
    if rsi_value >= 60:
        return 0.9
    if rsi_value >= 50:
        return 0.75
    if rsi_value >= 40:
        return 0.55
    if rsi_value >= 30:
        return 0.4
    return 0.3


def trend_score(sma20: Optional[float], sma50: Optional[float]) -> float:
    if sma20 is None or sma50 is None or sma50 == 0:
        return 0.5
    ratio = sma20 / sma50
    if ratio >= 1.1:
        return 1.0
    if ratio >= 1.02:
        return 0.85
    if ratio >= 1.0:
        return 0.7
    if ratio >= 0.98:
        return 0.5
    if ratio >= 0.95:
        return 0.35
    return 0.2


def compute_signals(current_price: float,
                     market_cap: Optional[float],
                     volume: Optional[float],
                     closes: List[float],
                     pct_24h: Optional[float],
                     community: Optional[Dict[str, Any]],
                     social_weight: Optional[float] = None,
                     disable_social: bool = False) -> Dict[str, Any]:
    rsi_val = rsi(closes, 14)
    sma20_val = sma(closes, 20)
    sma50_val = sma(closes, 50)
    bb = bollinger_bands(closes, 20, 2.0)
    vol20 = daily_return_volatility(closes, 20)

    soc_s = 0.0 if disable_social else social_score(community)
    liq_s = liquidity_score(volume, market_cap)
    risk_s = risk_score(vol20)
    mom_s = momentum_score(rsi_val)
    trd_s = trend_score(sma20_val, sma50_val)

    # Pesos base y ajustes según flags CLI (con autoajuste social opcional)
    w_trd = 0.30
    w_mom = 0.25
    auto_soc_note = None
    if disable_social:
        w_soc = 0.0
    else:
        if social_weight is None:
            # Autoajuste: más peso social cuando el puntaje social y la liquidez son altos y el riesgo es elevado, y cuando la capitalización es baja.
            base_soc = 0.12 + 0.28 * soc_s + 0.10 * liq_s + 0.10 * (1.0 - risk_s)
            small_cap_boost = 0.10 if (market_cap is not None and market_cap > 0 and market_cap < 50_000_000) else 0.0
            w_soc = clamp(base_soc + small_cap_boost, 0.10, 0.50)
            auto_soc_note = f"Peso social autoajustado a {w_soc:.2f} según señales sociales, liquidez, riesgo y capitalización."
        else:
            w_soc = clamp(social_weight, 0.0, 0.6)
    w_liq = 0.15
    w_risk = 0.10
    w_sum = max(1e-9, (w_trd + w_mom + w_soc + w_liq + w_risk))
    wn_trd = w_trd / w_sum
    wn_mom = w_mom / w_sum
    wn_soc = w_soc / w_sum
    wn_liq = w_liq / w_sum
    wn_risk = w_risk / w_sum

    # Ponderación total normalizada
    buy_score = clamp(wn_trd * trd_s + wn_mom * mom_s + wn_soc * soc_s + wn_liq * liq_s + wn_risk * risk_s, 0.0, 1.0)

    # Reglas heurísticas
    avoid = False
    reasons: List[str] = []

    # Señales sociales explícitas
    if not disable_social:
        if soc_s >= 0.75:
            reasons.append("Fuerte tracción social reciente (alcance/engagement altos).")
        elif soc_s <= 0.15:
            reasons.append("Tracción social débil: comunidad con baja actividad.")

    if (market_cap or 0) < 5_000_000 or (volume or 0) < 250_000 or len(closes) < 30:
        avoid = True
        reasons.append("Riesgo de liquidez/capitalización muy baja o datos insuficientes.")

    action = "Retener"

    # Señal de Vender (sobreextensión)
    if rsi_val is not None and bb is not None and pct_24h is not None:
        lower, mid, upper = bb
        bb_pos = safe_div(current_price - lower, (upper - lower) or 1.0, 0.5)
        if rsi_val > 75 and bb_pos > 0.9 and pct_24h > 0.20:
            action = "Vender"
            reasons.append("RSI > 75 y precio cerca de banda superior tras fuerte subida: tomar ganancias/ajustar trailing stop.")

    # Señal de Comprar (tendencia + momentum saludables)
    if action != "Vender":
        if buy_score >= 0.62 and (rsi_val is None or 45 <= rsi_val <= 70) and (sma20_val is None or current_price >= sma20_val) and (sma50_val is None or sma20_val is None or sma20_val >= sma50_val):
            action = "Comprar"
            reasons.append("Tendencia y momentum saludables con liquidez/social aceptables.")
        elif rsi_val is not None and rsi_val < 35 and liq_s > 0.5 and soc_s > 0.6 and not disable_social:
            action = "Retener"
            reasons.append("Sobreventa con buen soporte social/liquidez: acumular gradualmente.")
        else:
            action = "Retener"
            reasons.append("Señales mixtas: esperar confirmación o mantener posición.")

    if avoid:
        action = "Evitar"

    # Anotar autoajuste de peso social si aplica
    if auto_soc_note:
        try:
            reasons.append(auto_soc_note)
        except Exception:
            pass

    # Niveles guía: basados en volatilidad reciente
    vol = vol20 or 0.05
    stop_loss_pct = clamp(2.0 * vol, 0.05, 0.15)  # 5% a 15%
    take_profit_pct = clamp(2.0 * vol, 0.08, 0.30)  # 8% a 30%

    # Posición relativa a Bandas de Bollinger
    bb_pos_str = "-"
    if bb is not None:
        lower, mid, upper = bb
        width = (upper - lower) or 1.0
        bb_pos = clamp((current_price - lower) / width, 0.0, 1.0)
        bb_pos_str = f"{bb_pos*100:.1f}% dentro del canal"

    # Estrategia textual según señales y niveles guía
    if action == "Vender":
        strategy_text = f"Toma de ganancias/gestión de riesgo: TP ~ {take_profit_pct*100:.1f}%, SL ~ {stop_loss_pct*100:.1f}%."
    elif action == "Comprar":
        strategy_text = f"Tendencia y momentum: entrar y gestionar con TP ~ {take_profit_pct*100:.1f}% y SL ~ {stop_loss_pct*100:.1f}%."
    elif action == "Retener":
        strategy_text = f"Acumulación/espera: mantener y evaluar; TP ~ {take_profit_pct*100:.1f}%, SL ~ {stop_loss_pct*100:.1f}%."
    else:
        strategy_text = "Preservación de capital: evitar por condiciones de riesgo."

    return {
        "rsi": rsi_val,
        "sma20": sma20_val,
        "sma50": sma50_val,
        "bb": bb,
        "volatility20": vol20,
        "social_score": soc_s,
        "liquidity_score": liq_s,
        "risk_score": risk_s,
        "momentum_score": mom_s,
        "trend_score": trd_s,
        "buy_score": buy_score,
        "action": action,
        "strategy": strategy_text,
        "reasons": reasons,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "bb_position": bb_pos_str,
    }


# ----------------------------- IA Avanzada (Logistic Regression) --------------

class _LogRegGD:
    def __init__(self, lr: float = 0.1, epochs: int = 150, l2: float = 0.001):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.w: Optional[List[float]] = None  # incluye intercepto w0 al inicio
        self.mu: Optional[List[float]] = None
        self.sigma: Optional[List[float]] = None

    @staticmethod
    def _sigmoid(z: float) -> float:
        # Evitar overflow
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def _standardize(self, X: List[List[float]]) -> List[List[float]]:
        if self.mu is None or self.sigma is None:
            # calcular
            n_feat = len(X[0]) if X else 0
            mu = [0.0] * n_feat
            sigma = [0.0] * n_feat
            if n_feat > 0:
                for j in range(n_feat):
                    col = [x[j] for x in X]
                    m = sum(col) / len(col)
                    v = sum((v - m) ** 2 for v in col) / max(1, (len(col) - 1))
                    s = math.sqrt(max(1e-12, v))
                    mu[j] = m
                    sigma[j] = s
            self.mu = mu
            self.sigma = sigma
        # aplicar
        Xs: List[List[float]] = []
        for row in X:
            Xs.append([(row[j] - self.mu[j]) / (self.sigma[j] if self.sigma[j] != 0 else 1.0) for j in range(len(row))])
        return Xs

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        if not X or not y or len(X) != len(y):
            # nada que entrenar
            self.w = None
            return
        # estandarizar
        Xs = self._standardize(X)
        n = len(Xs)
        d = len(Xs[0]) if Xs else 0
        # inicializar pesos (incluye intercepto)
        self.w = [0.0] * (d + 1)
        lr = self.lr
        l2 = self.l2
        for _ in range(self.epochs):
            # gradientes
            grad = [0.0] * (d + 1)
            for i in range(n):
                z = self.w[0]
                xi = Xs[i]
                for j in range(d):
                    z += self.w[j + 1] * xi[j]
                p = self._sigmoid(z)
                err = p - float(y[i])
                grad[0] += err
                for j in range(d):
                    grad[j + 1] += err * xi[j]
            # regularización L2 (sin intercepto)
            for j in range(1, d + 1):
                grad[j] += l2 * self.w[j]
            inv_n = 1.0 / max(1, n)
            for j in range(d + 1):
                self.w[j] -= lr * grad[j] * inv_n

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        if not X:
            return []
        if self.w is None:
            # modelo no entrenado: prob neutral
            return [0.5 for _ in X]
        # usar medias/sigmas aprendidas
        Xs: List[List[float]] = []
        for row in X:
            if self.mu is None or self.sigma is None:
                Xs.append(row[:])
            else:
                Xs.append([(row[j] - self.mu[j]) / (self.sigma[j] if self.sigma[j] != 0 else 1.0) for j in range(len(row))])
        probs: List[float] = []
        d = len(self.w) - 1
        for xi in Xs:
            z = self.w[0]
            for j in range(min(d, len(xi))):
                z += self.w[j + 1] * xi[j]
            probs.append(self._sigmoid(z))
        return probs


def _sma_ratio_norm(sma20_val: Optional[float], sma50_val: Optional[float]) -> float:
    if sma20_val is None or sma50_val is None or sma50_val == 0:
        return 0.5
    ratio = sma20_val / sma50_val
    # mapear ~[0.8, 1.4] -> [0,1]
    return clamp((ratio - 0.8) / 0.6, 0.0, 1.0)


def _bb_position(closes: List[float]) -> float:
    bb = bollinger_bands(closes, 20, 2.0)
    if not bb:
        return 0.5
    lower, mid, upper = bb
    width = (upper - lower) or 1.0
    price = closes[-1]
    return clamp((price - lower) / width, 0.0, 1.0)


def _vol_norm(closes: List[float]) -> float:
    v = daily_return_volatility(closes, 20)
    return clamp((v or 0.0) / 0.15, 0.0, 1.0)


def _recent_ret(closes: List[float], k: int) -> float:
    if len(closes) < k + 1:
        return 0.0
    a = closes[-k - 1]
    b = closes[-1]
    return (b - a) / a if a != 0 else 0.0


def _features_at_index(full_closes: List[float], idx: int, comm: Optional[Dict[str, Any]], market_cap: Optional[float], volume: Optional[float]) -> List[float]:
    # Usar serie hasta idx (incluido)
    if idx < 0:
        idx = 0
    closes = full_closes[:idx + 1]
    rsi_val = rsi(closes, 14)
    sma20_val = sma(closes, 20)
    sma50_val = sma(closes, 50)
    rsi_norm = clamp(((rsi_val if rsi_val is not None else 50.0) / 100.0), 0.0, 1.0)
    sma_ratio_n = _sma_ratio_norm(sma20_val, sma50_val)
    bb_pos = _bb_position(closes)
    vol_n = _vol_norm(closes)
    risk_s = risk_score(daily_return_volatility(closes, 20))
    # Retornos recientes (sin normalizar; estandarización interna del modelo)
    ret1 = 0.0
    if len(closes) >= 2:
        prev = closes[-2]
        cur = closes[-1]
        ret1 = (cur - prev) / prev if prev != 0 else 0.0
    ret3 = _recent_ret(closes, 3)
    ret7 = _recent_ret(closes, 7)
    soc_s = social_score(comm) if comm else 0.0
    liq_s = liquidity_score(volume, market_cap)
    return [rsi_norm, sma_ratio_n, bb_pos, vol_n, risk_s, ret1, ret3, ret7, soc_s, liq_s]


def _build_dataset_for_coin(closes: List[float], comm: Optional[Dict[str, Any]], market_cap: Optional[float], volume: Optional[float], horizon: int = 3, max_samples: int = 400) -> Tuple[List[List[float]], List[int]]:
    X: List[List[float]] = []
    y: List[int] = []
    if not closes or len(closes) < 30 + horizon:
        return X, y
    start = 30  # asegurar mínimos para indicadores simples
    end = len(closes) - horizon - 1
    if end <= start:
        return X, y
    n_points = end - start + 1
    step = max(1, n_points // max_samples)
    for idx in range(start, end + 1, step):
        feats = _features_at_index(closes, idx, comm, market_cap, volume)
        # etiqueta: retorno futuro positivo a horizonte fijo
        a = closes[idx]
        b = closes[idx + horizon]
        lbl = 1 if (a > 0 and (b - a) / a > 0.0) else 0
        X.append(feats)
        y.append(lbl)
    return X, y


def sizing_recommendations(results: List[Dict[str, Any]], budget: float, risk: str, concentrate_enabled: bool = True, concentrate_threshold: float = 0.85) -> List[Dict[str, Any]]:
    risk = (risk or "moderado").lower()
    if risk not in ("bajo", "moderado", "alto"):
        risk = "moderado"

    # Asegurar umbral válido
    concentrate_threshold = clamp(float(concentrate_threshold or 0.0), 0.0, 1.0)

    if risk == "bajo":
        total_invest_pct = 0.30
        max_coin_pct = 0.05
    elif risk == "alto":
        total_invest_pct = 0.70
        max_coin_pct = 0.20
    else:  # moderado
        total_invest_pct = 0.50
        max_coin_pct = 0.10

    # Seleccionar invertibles (Comprar y algunos Retener de alto puntaje)
    investables = [r for r in results if r.get("analysis", {}).get("action") in ("Comprar", "Retener") and r.get("analysis", {}).get("buy_score", 0) >= 0.5 and (r.get("market_cap") or 0) >= 5_000_000 and (r.get("volume") or 0) >= 250_000]

    if not investables:
        return []

    # Opción de concentración: si el mejor candidato es MUY bueno, concentrar todo el capital objetivo en él
    if concentrate_enabled:
        best = max(investables, key=lambda x: x.get("analysis", {}).get("buy_score", 0.0))
        best_score = float(best.get("analysis", {}).get("buy_score", 0.0) or 0.0)
        best_action = (best.get("analysis", {}).get("action") or "").strip()
        if best_action == "Comprar" and best_score >= concentrate_threshold:
            amount = budget * total_invest_pct
            return [{
                "id": best["id"],
                "symbol": best["symbol"],
                "name": best["name"],
                "price": best["price"],
                "target_pct": total_invest_pct,
                "amount": amount,
                "stop_loss_price": best["price"] * (1 - best["analysis"]["stop_loss_pct"]),
                "take_profit_price": best["price"] * (1 + best["analysis"]["take_profit_pct"]),
                "concentrated": True,
                "concentrate_threshold": concentrate_threshold,
                "buy_score": best_score,
            }]

    # Distribución proporcional por buy_score si no hay concentración
    total_score = sum(r["analysis"]["buy_score"] for r in investables)
    allocations: List[Dict[str, Any]] = []
    remaining_budget = budget * total_invest_pct

    for r in sorted(investables, key=lambda x: x["analysis"]["buy_score"], reverse=True):
        weight = r["analysis"]["buy_score"] / total_score if total_score > 0 else 0
        target_pct = clamp(weight * total_invest_pct, 0.0, max_coin_pct)
        amount = remaining_budget * target_pct / total_invest_pct  # proporcional dentro del monto a invertir
        # Mínimo razonable si es "Comprar"
        min_pct = 0.02 if r["analysis"]["action"] == "Comprar" else 0.0
        target_pct = max(target_pct, min_pct)
        amount = budget * target_pct

        allocations.append({
            "id": r["id"],
            "symbol": r["symbol"],
            "name": r["name"],
            "price": r["price"],
            "target_pct": target_pct,
            "amount": amount,
            "stop_loss_price": r["price"] * (1 - r["analysis"]["stop_loss_pct"]),
            "take_profit_price": r["price"] * (1 + r["analysis"]["take_profit_pct"]),
        })

    # Re-escalar si excede total_invest_pct
    total_pct = sum(a["target_pct"] for a in allocations)
    if total_pct > total_invest_pct > 0:
        scale = total_invest_pct / total_pct
        for a in allocations:
            a["target_pct"] *= scale
            a["amount"] = budget * a["target_pct"]

    return allocations


def discover_meme_category_id(client: CoinGeckoClient) -> Optional[str]:
    try:
        cats = client.categories_list()
        # Buscar categoría que contenga 'meme'
        for c in cats:
            if "meme" in (c.get("category_id") or "").lower() or "meme" in (c.get("name") or "").lower():
                return c.get("category_id")
        # Fallbacks conocidos
        return "memes"
    except Exception:
        return "memes"


def fetch_markets(client: CoinGeckoClient, vs: str, top: int) -> List[Dict[str, Any]]:
    # Intentar por categoría
    cat = discover_meme_category_id(client)
    markets: List[Dict[str, Any]] = []
    try:
        markets = client.markets_by_category(cat or "memes", vs=vs, per_page=min(250, max(10, top)))
    except Exception:
        markets = []
    # Si vacío, usar trending como fallback
    if not markets:
        try:
            tr = client.search_trending()
            ids = [item["item"]["id"] for item in tr.get("coins", []) if isinstance(item, dict) and item.get("item", {}).get("id")]
            markets = client.markets_by_ids(ids, vs=vs)
        except Exception:
            markets = []
    return markets[:top]


def is_listed_on_bit2me(client: CoinGeckoClient, coin_id: str) -> bool:
    """Comprueba vía CoinGecko si una moneda tiene algún ticker en el exchange Bit2Me.
    La verificación es por nombre o identificador del exchange, insensible a mayúsculas.
    """
    if not coin_id:
        return False
    try:
        tickers = client.coin_tickers(coin_id)
        for t in tickers:
            mkt = t.get("market") or {}
            name = (mkt.get("name") or "").lower()
            ident = (mkt.get("identifier") or "").lower()
            if "bit2me" in name or "bit2me" in ident:
                return True
        return False
    except Exception:
        return False


def filter_bit2me_markets(client: CoinGeckoClient, markets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Filtra mercados a los que estén listados en Bit2Me. Devuelve (filtrados, ids_excluidos)."""
    filtered: List[Dict[str, Any]] = []
    removed: List[str] = []
    for m in markets:
        cid = m.get("id")
        if is_listed_on_bit2me(client, cid):
            filtered.append(m)
        else:
            if cid:
                removed.append(cid)
    return filtered, removed


def analyze_markets(client: CoinGeckoClient, markets: List[Dict[str, Any]], vs: str, days: int, social_weight: Optional[float] = None, disable_social: bool = False, ai_mode: Optional[str] = "advanced", ai_weight: float = 0.35) -> List[Dict[str, Any]]:
    # Normalizar flags IA
    ai_mode_l = (ai_mode or "advanced").lower()
    if ai_mode_l not in ("off", "basic", "advanced"):
        ai_mode_l = "advanced"
    ai_w = clamp(float(ai_weight or 0.0), 0.0, 1.0)
    horizon = None if ai_mode_l == "off" else (1 if ai_mode_l == "basic" else 3)

    results: List[Dict[str, Any]] = []
    # Dataset IA y contexto de predicción
    X_all: List[List[float]] = []
    y_all: List[int] = []
    ctx_list: List[Dict[str, Any]] = []

    for m in markets:
        coin_id = m.get("id")
        symbol = (m.get("symbol") or "").upper()
        name = m.get("name")
        price = m.get("current_price")
        market_cap = m.get("market_cap")
        volume = m.get("total_volume")
        pct_24h = m.get("price_change_percentage_24h_in_currency")

        comm = None
        chart = None
        try:
            comm = client.community_data(coin_id)
        except Exception:
            comm = None
        try:
            chart = client.market_chart(coin_id, vs=vs, days=days)
        except Exception:
            chart = None

        closes: List[float] = []
        used_fallback = False
        if chart and isinstance(chart.get("prices"), list):
            closes = [float(p[1]) for p in chart["prices"] if isinstance(p, list) and len(p) >= 2]
            # Asegurar que no haya valores no positivos que rompan cálculos
            closes = [c for c in closes if isinstance(c, (int, float)) and c > 0]

        if not closes and price:
            # fallback mínimo con precio actual repetido (evita errores pero no da señales sólidas)
            closes = [float(price)] * max(50, days)
            used_fallback = True

        analysis = compute_signals(
            price or 0.0,
            market_cap,
            volume,
            closes,
            (pct_24h / 100.0) if isinstance(pct_24h, (int, float)) else None,
            comm,
            social_weight=social_weight,
            disable_social=disable_social,
        )
        # Anotar advertencias de calidad de datos
        try:
            if used_fallback:
                analysis.setdefault("reasons", []).append(
                    "Datos de precios insuficientes: usando sustitución con precio actual repetido (menor confianza técnica)."
                )
            if not disable_social and comm is None:
                analysis.setdefault("reasons", []).append(
                    "Sin datos sociales de CoinGecko para esta moneda (componente social menos preciso)."
                )
            if disable_social:
                analysis.setdefault("reasons", []).append(
                    "Componente social desactivado por configuración del usuario."
                )
        except Exception:
            pass

        result_item = {
            "id": coin_id,
            "symbol": symbol,
            "name": name,
            "price": float(price) if price is not None else None,
            "market_cap": float(market_cap) if market_cap is not None else None,
            "volume": float(volume) if volume is not None else None,
            "change_1h": (m.get("price_change_percentage_1h_in_currency") or 0) / 100.0,
            "change_24h": (m.get("price_change_percentage_24h_in_currency") or 0) / 100.0,
            "change_7d": (m.get("price_change_percentage_7d_in_currency") or 0) / 100.0,
            "change_30d": (m.get("price_change_percentage_30d_in_currency") or 0) / 100.0,
            "analysis": analysis,
            "community": comm or {},
        }
        results.append(result_item)

        # Construir dataset para IA
        if horizon is not None:
            comm_ai = None if disable_social else comm
            Xi, yi = _build_dataset_for_coin(closes, comm_ai, market_cap, volume, horizon=horizon, max_samples=400)
            if Xi and yi:
                X_all.extend(Xi)
                y_all.extend(yi)
            # Guardar contexto para predicción posterior
            ctx_list.append({
                "analysis": analysis,
                "price": float(price) if price is not None else None,
                "closes": closes,
                "comm": comm_ai,
                "market_cap": market_cap,
                "volume": volume,
            })

    # Entrenar y aplicar IA si corresponde
    if horizon is not None and len(X_all) >= 30 and ctx_list:
        try:
            if ai_mode_l == "basic":
                model = _LogRegGD(lr=0.12, epochs=80, l2=0.0005)
            else:
                model = _LogRegGD(lr=0.12, epochs=150, l2=0.001)
            model.fit(X_all, y_all)
            for ctx in ctx_list:
                a = ctx["analysis"]
                closes = ctx["closes"]
                price_now = ctx["price"] or (closes[-1] if closes else 0.0)
                feats_now = _features_at_index(closes, len(closes) - 1, ctx["comm"], ctx["market_cap"], ctx["volume"]) if closes else [0.0]*10
                prob = model.predict_proba([feats_now])[0]
                base = float(a.get("buy_score") or 0.0)
                a["buy_score_base"] = base
                a["buy_score_ai"] = clamp(prob, 0.0, 1.0)
                a["buy_score"] = clamp((1.0 - ai_w) * base + ai_w * prob, 0.0, 1.0)
                # Añadir razón IA
                try:
                    if isinstance(a.get("reasons"), list):
                        a["reasons"].append(f"IA avanzada: probabilidad de subida ~ {prob*100:.1f}% (peso {ai_w:.2f}).")
                except Exception:
                    pass
                # Ajuste de acción sólo si no es Evitar/Vender
                if a.get("action") not in ("Evitar", "Vender"):
                    rsi_v = a.get("rsi")
                    sma20_v = a.get("sma20")
                    sma50_v = a.get("sma50")
                    if a.get("buy_score", 0.0) >= 0.62 and (rsi_v is None or 45 <= rsi_v <= 70) and (sma20_v is None or (price_now is not None and price_now >= sma20_v)) and (sma50_v is None or sma20_v is None or sma20_v >= sma50_v):
                        a["action"] = "Comprar"
                        a["strategy"] = f"Tendencia y momentum: entrar y gestionar con TP ~ {a.get('take_profit_pct', 0.0)*100:.1f}% y SL ~ {a.get('stop_loss_pct', 0.0)*100:.1f}%."
        except Exception:
            # si IA falla, ignorar silenciosamente para no romper el flujo
            pass

    # Anotar razones globales relacionadas con IA
    try:
        if ai_mode_l == "off":
            for r in results:
                r.get("analysis", {}).setdefault("reasons", []).append("IA desactivada por configuración del usuario.")
        elif horizon is not None and (len(X_all) < 30 or not ctx_list):
            for r in results:
                r.get("analysis", {}).setdefault("reasons", []).append("IA no aplicada por datos insuficientes (muestras < 30); se usa buy_score base.")
    except Exception:
        pass

    return results


# ----------------------------- Impresión -------------------------------------

def print_report(results: List[Dict[str, Any]], allocations: List[Dict[str, Any]], budget: float, risk: str, vs: str):
    now = dt.datetime.now()
    print("")
    print("=" * 88)
    print(f"Análisis de Memecoins — {now.strftime('%Y-%m-%d %H:%M')}  |  Moneda base: {vs.upper()}  |  Perfil: {risk.capitalize()}")
    print("=" * 88)
    print("")

    # Presupuesto y uso objetivo según perfil
    risk_l = (risk or "moderado").lower()
    if risk_l == "bajo":
        invest_pct = 0.30
    elif risk_l == "alto":
        invest_pct = 0.70
    else:
        invest_pct = 0.50
    print(f"Presupuesto de trabajo: {fmt_usd(budget)}  |  Uso objetivo según perfil: {fmt_pct(invest_pct)} -> {fmt_usd(budget * invest_pct)}")
    print("")

    if not results:
        print("No se obtuvieron resultados. Verifique su conexión a Internet o intente más tarde.")
        return

    # Mapa para buscar estrategia por id
    analysis_by_id: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if r.get("id"):
            analysis_by_id[r["id"]] = r.get("analysis", {})

    for r in results:
        a = r["analysis"]
        price = r.get("price")
        print(f"{r['name']} ({r['symbol']})  Precio: {fmt_usd(price)}  MC: {fmt_num(r.get('market_cap'))}  Vol: {fmt_num(r.get('volume'))}")
        print(f"  Cambios: 1h {fmt_pct(r.get('change_1h'))} | 24h {fmt_pct(r.get('change_24h'))} | 7d {fmt_pct(r.get('change_7d'))} | 30d {fmt_pct(r.get('change_30d'))}")
        rsi_str = f"{a['rsi']:.1f}" if isinstance(a.get('rsi'), (int, float)) and a.get('rsi') is not None else "-"
        sma20_str = f"{a['sma20']:.6f}" if a.get('sma20') is not None else "-"
        sma50_str = f"{a['sma50']:.6f}" if a.get('sma50') is not None else "-"
        print(f"  RSI14: {rsi_str}  | SMA20: {sma20_str}  | SMA50: {sma50_str}  | BB Pos: {a['bb_position']}")
        ai_v = a.get('buy_score_ai')
        base_v = a.get('buy_score_base')
        ai_str = f"{ai_v:.2f}" if isinstance(ai_v, (int, float)) else "-"
        base_str = f"{base_v:.2f}" if isinstance(base_v, (int, float)) else "-"
        print(f"  Puntajes — Buy: {a['buy_score']:.2f} (IA {ai_str} | Base {base_str})  | Tendencia: {a['trend_score']:.2f}  | Momentum: {a['momentum_score']:.2f}  | Social: {a['social_score']:.2f}  | Liquidez: {a['liquidity_score']:.2f}  | Riesgo: {a['risk_score']:.2f}")
        print(f"  Recomendación: {a['action']}  | Stop Loss ~ {fmt_pct(a['stop_loss_pct'])}  | Take Profit ~ {fmt_pct(a['take_profit_pct'])}")
        print(f"  Estrategia: {a.get('strategy', '-')} ")
        for reason in a.get("reasons", [])[:3]:
            print(f"   - {reason}")
        print("")

    if allocations:
        print("Asignación sugerida (NO es asesoría financiera):")
        # Nota de concentración si aplica
        try:
            if len(allocations) == 1 and allocations[0].get("concentrated"):
                _a0 = allocations[0]
                _bs = _a0.get("buy_score")
                _thr = _a0.get("concentrate_threshold")
                _pct = _a0.get("target_pct")
                print(f"  [Modo concentración] Se destina {fmt_pct(_pct)} del presupuesto a {_a0['name']} ({_a0['symbol']}) por BuyScore {_bs:.2f} ≥ umbral {_thr:.2f}. Use --no-concentrate para desactivar.")
        except Exception:
            pass
        total_amt = 0.0
        total_gain = 0.0
        total_loss = 0.0
        for alloc in allocations:
            total_amt += alloc["amount"]
        for alloc in allocations:
            pct_str = fmt_pct(alloc["target_pct"]) if alloc["target_pct"] is not None else "-"
            print(f"  - {alloc['name']} ({alloc['symbol']}): {pct_str} del presupuesto -> {fmt_usd(alloc['amount'])} @ {fmt_usd(alloc['price'])}")
            print(f"      Stop {fmt_usd(alloc['stop_loss_price'])}  |  TP {fmt_usd(alloc['take_profit_price'])}")
            price = alloc.get('price') or 0.0
            tp = alloc.get('take_profit_price') or 0.0
            sl = alloc.get('stop_loss_price') or 0.0
            amount = alloc.get('amount') or 0.0
            tp_pct = (tp / price - 1.0) if price else 0.0
            sl_pct = (1.0 - sl / price) if price else 0.0
            gain_usd = max(0.0, amount * tp_pct)
            loss_usd = max(0.0, amount * sl_pct)
            rr = (gain_usd / loss_usd) if loss_usd > 0 else None
            strategy_text = analysis_by_id.get(alloc.get('id'), {}).get('strategy') or "-"
            print(f"      Estrategia: {strategy_text}")
            rr_str = f"{rr:.2f}" if rr is not None else "-"
            print(f"      + Ganancia potencial: {fmt_usd(gain_usd)} (+{fmt_pct(tp_pct)})  |  - Pérdida potencial: {fmt_usd(loss_usd)} (-{fmt_pct(sl_pct)})  |  R/R: {rr_str}")
            total_gain += gain_usd
            total_loss += loss_usd
        print(f"  Total asignado: {fmt_usd(total_amt)} de {fmt_usd(budget)}")
        rr_total = (total_gain / total_loss) if total_loss > 0 else None
        rr_total_str = f"{rr_total:.2f}" if rr_total is not None else "-"
        print(f"  Escenario de la canasta — +Ganancia potencial: {fmt_usd(total_gain)}  |  -Pérdida potencial: {fmt_usd(total_loss)}  |  R/R total: {rr_total_str}")
    else:
        print("Por ahora no hay una canasta clara de compra según su perfil. Considere retener efectivo y esperar confirmación.")

    print("")
    print("Aviso: Este bot no garantiza resultados ni sustituye asesoría profesional. Las memecoins son extremadamente volátiles.")


# ------------------------ Simulación (Paper Trading) --------------------------

def normalize_pct(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x <= 0:
        return 0.0
    # Acepta notación en % (10 => 10%) o decimal (0.10 => 10%)
    return clamp(x / 100.0, 0.0001, 0.95) if x > 1.0 else clamp(x, 0.0001, 0.95)


def simulate_trade(
    client: CoinGeckoClient,
    coin_id: str,
    vs: str = "usd",
    days: int = 30,
    initial_balance: float = 100.0,
    tp_pct: float = 0.10,
    sl_pct: float = 0.08,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Emula una operación simple con TP/SL sobre datos reales de CoinGecko.

    Reglas:
    - Entra al primer precio del periodo.
    - Recorre la serie y decide el primer evento alcanzado: TP (éxito) o SL (pérdida).
    - Si SL: descuenta del balance y reintenta con lo que queda, desde el siguiente punto.
    - Se detiene al primer TP o al agotar intentos/datos.
    - Si no toca TP/SL, cierra al último precio (resultado FORCED con PnL real del periodo).
    """
    data = client.market_chart(coin_id, vs=vs, days=days)
    attempts: List[Dict[str, Any]] = []
    if not isinstance(data, dict) or not data.get("prices"):
        return {
            "ok": False,
            "error": "No se pudo obtener datos de precios.",
            "attempts": attempts,
            "final_balance": initial_balance,
        }
    series = data.get("prices") or []
    prices: List[Tuple[int, float]] = [
        (int(p[0]), to_float(p[1])) for p in series if isinstance(p, list) and len(p) >= 2
    ]
    prices = [p for p in prices if p[1] > 0]
    if len(prices) < 5:
        return {
            "ok": False,
            "error": "Datos de precios insuficientes para simular.",
            "attempts": attempts,
            "final_balance": initial_balance,
        }

    balance = max(0.0, float(initial_balance))
    idx = 0
    att = 0
    success = False

    while att < max_attempts and idx < len(prices) - 1 and balance > 0.0:
        entry_ts, entry_price = prices[idx]
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)
        exit_idx: Optional[int] = None
        outcome = "NONE"

        for j in range(idx + 1, len(prices)):
            _, px = prices[j]
            if px >= tp_price:  # prioriza TP si ambos se cruzan entre puntos discretos
                exit_idx = j
                outcome = "TP"
                break
            if px <= sl_price:
                exit_idx = j
                outcome = "SL"
                break

        if exit_idx is None:
            # cierre forzado al último precio
            exit_idx = len(prices) - 1
            outcome = "FORCED"

        exit_ts, exit_price = prices[exit_idx]
        pnl_pct = (exit_price / entry_price - 1.0) if entry_price > 0 else 0.0
        if outcome == "SL":
            pnl_pct = -sl_pct
        elif outcome == "TP":
            pnl_pct = tp_pct
        pnl_usd = balance * pnl_pct
        balance_after = balance + pnl_usd

        attempts.append(
            {
                "attempt": att + 1,
                "entry_ts": entry_ts,
                "entry_price": entry_price,
                "exit_ts": exit_ts,
                "exit_price": exit_price,
                "outcome": outcome,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "balance_before": balance,
                "balance_after": balance_after,
            }
        )

        balance = max(0.0, balance_after)
        idx = exit_idx + 1
        att += 1

        if outcome == "TP":
            success = True
            break
        if idx >= len(prices) - 1:
            break

    return {
        "ok": True,
        "coin_id": coin_id,
        "vs": vs,
        "days": days,
        "attempts": attempts,
        "success": success,
        "final_balance": balance,
        "start_price": prices[0][1],
        "end_price": prices[-1][1],
    }


def print_simulation_report(result: Dict[str, Any]) -> None:
    print("")
    print("=" * 72)
    print("Simulación de operación (paper trading)")
    print("=" * 72)
    if not result.get("ok"):
        print(result.get("error") or "Error desconocido.")
        return
    coin_id = result.get("coin_id")
    vs = (result.get("vs") or "usd").upper()
    days = result.get("days")
    attempts = result.get("attempts") or []
    print(f"Moneda: {coin_id}  |  Base: {vs}  |  Días: {days}")
    print("")
    for a in attempts:
        et = dt.datetime.fromtimestamp(a["entry_ts"] / 1000.0)
        xt = dt.datetime.fromtimestamp(a["exit_ts"] / 1000.0)
        print(
            f"Intento #{a['attempt']}: {et.strftime('%Y-%m-%d %H:%M')} -> {xt.strftime('%Y-%m-%d %H:%M')}  |  {a['outcome']}"
        )
        print(
            f"  Entrada: {fmt_usd(a['entry_price'])}  |  Salida: {fmt_usd(a['exit_price'])}  |  PnL: {fmt_pct(a['pnl_pct'])} -> {fmt_usd(a['pnl_usd'])}"
        )
        print(
            f"  Balance: {fmt_usd(a['balance_before'])} -> {fmt_usd(a['balance_after'])}"
        )
    print("")
    print(f"Éxito por TP alcanzado: {'Sí' if result.get('success') else 'No'}")
    print(f"Balance final: {fmt_usd(result.get('final_balance'))}")
    print("")
    print(
        "Nota: Esta simulación usa cierres discretos (horarios/diarios). Si TP y SL se cruzan entre puntos, se prioriza TP al evaluar. Es una aproximación; no refleja ejecución real."
    )
    print("")

# ----------------------------- CLI / Main ------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bot de análisis de memecoins (CoinGecko)")
    parser.add_argument("--budget", "--capital", dest="budget", type=float, default=1000.0, help="Presupuesto total a invertir en la canasta sugerida (por defecto 1000). Alias: --capital")
    parser.add_argument("--risk", type=str, default="moderado", help="Perfil de riesgo: bajo | moderado | alto")
    parser.add_argument("--top", type=int, default=8, help="Número de memecoins a analizar (por defecto 8)")
    parser.add_argument("--days", type=int, default=90, help="Días de historial para análisis técnico (por defecto 90)")
    parser.add_argument("--vs", type=str, default="usd", help="Moneda base para precios (por defecto usd)")
    parser.add_argument("--offline", action="store_true", help="Modo offline: no realiza solicitudes HTTP (solo muestra aviso)")
    parser.add_argument("--coins", type=str, default="", help="IDs específicos de CoinGecko separados por coma (opcional)")
    parser.add_argument("--social-weight", dest="social_weight", type=float, default=None, help="Peso del componente social en el buy_score (0.0–0.6). Si no se especifica, el peso se autoajusta (≈0.10–0.50) según señales sociales, liquidez, riesgo y capitalización.")
    parser.add_argument("--disable-social", dest="disable_social", action="store_true", help="Desactiva el componente social (solo señales técnicas/liquidez/riesgo)")
    parser.add_argument("--concentrate", dest="concentrate", action="store_true", default=True, help="Activa la concentración de capital en una sola operación cuando la mejor candidata es muy buena (por defecto activado)")
    parser.add_argument("--no-concentrate", dest="concentrate", action="store_false", help="Desactiva la concentración de capital")
    parser.add_argument("--concentrate-threshold", dest="concentrate_threshold", type=float, default=0.85, help="Umbral del buy_score (0.0–1.0) para considerar 'muy buena'. Por defecto 0.85")
    parser.add_argument("--ai-mode", dest="ai_mode", type=str, default="advanced", help="IA: off | basic | advanced (por defecto advanced)")
    parser.add_argument("--ai-weight", dest="ai_weight", type=float, default=0.35, help="Peso de la IA al combinar con buy_score (0.0–1.0). Por defecto 0.35")
    # Filtro Bit2Me
    parser.add_argument("--only-bit2me", dest="only_bit2me", action="store_true", default=True, help="Restringe las propuestas a monedas listadas en Bit2Me (por defecto activado).")
    parser.add_argument("--no-bit2me", dest="only_bit2me", action="store_false", help="Desactiva el filtro de Bit2Me (analiza aunque no estén listadas).")
    # Simulación (paper trading)
    parser.add_argument("--simulate", dest="simulate", action="store_true", help="Modo simulación (paper trading) con TP/SL sobre datos reales.")
    parser.add_argument("--sim-coin", dest="sim_coin", type=str, default="", help="ID de la moneda en CoinGecko para simular (ej. pepe, shiba-inu).")
    parser.add_argument("--sim-days", dest="sim_days", type=int, default=30, help="Días de historial para simular (por defecto 30).")
    parser.add_argument("--sim-balance", dest="sim_balance", type=float, default=100.0, help="Balance inicial para simular (por defecto 100).")
    parser.add_argument("--sim-tp", dest="sim_tp", type=float, default=10.0, help="Take Profit como % o decimal (ej. 10 o 0.10).")
    parser.add_argument("--sim-sl", dest="sim_sl", type=float, default=8.0, help="Stop Loss como % o decimal (ej. 8 o 0.08).")
    parser.add_argument("--sim-attempts", dest="sim_attempts", type=int, default=3, help="Intentos máximos tras SL para reintentar con balance restante (por defecto 3).")

    args = parser.parse_args(argv)

    if args.offline:
        print("Modo offline: no se realizarán solicitudes a CoinGecko. Conéctese a Internet para análisis en vivo.")
        print("Ejemplo: python main.py --capital 1500 --risk moderado --top 10 --days 120")
        print("Consulta README.md para más opciones y explicación de estrategia/TP/SL.")
        return 0

    # Saneamiento y validaciones de argumentos
    risk_use = (getattr(args, "risk", "moderado") or "moderado").lower()
    if risk_use not in ("bajo", "moderado", "alto"):
        print(f"Aviso: --risk '{args.risk}' no reconocido. Se usará 'moderado'.")
        risk_use = "moderado"

    top_raw = int(getattr(args, "top", 8))
    top_use = int(min(250, max(1, top_raw)))
    if top_use != top_raw:
        print(f"Aviso: --top {top_raw} fuera de rango [1,250]. Ajustado a {top_use}.")

    days_raw = int(getattr(args, "days", 90))
    days_use = int(min(365, max(30, days_raw)))
    if days_use != days_raw:
        print(f"Aviso: --days {days_raw} ajustado a {days_use} (rango 30–365).")

    vs_raw = getattr(args, "vs", "usd") or "usd"
    vs_use = str(vs_raw).lower()
    if vs_use != vs_raw:
        print(f"Aviso: --vs normalizado a '{vs_use}'.")

    budget_raw = float(getattr(args, "budget", 1000.0))
    budget_use = max(0.0, budget_raw)
    if budget_use != budget_raw:
        print(f"Aviso: --budget negativo no válido. Ajustado a {budget_use}.")

    ai_w_raw = float(getattr(args, "ai_weight", 0.35))
    ai_weight_use = clamp(ai_w_raw, 0.0, 1.0)
    if ai_weight_use != ai_w_raw:
        print(f"Aviso: --ai-weight {ai_w_raw} fuera de [0.0,1.0]; ajustado a {ai_weight_use}.")

    conc_thr_raw = float(getattr(args, "concentrate_threshold", 0.85))
    concentrate_threshold_use = clamp(conc_thr_raw, 0.0, 1.0)
    if concentrate_threshold_use != conc_thr_raw:
        print(f"Aviso: --concentrate-threshold {conc_thr_raw} fuera de [0.0,1.0]; ajustado a {concentrate_threshold_use}.")

    disable_social = bool(getattr(args, "disable_social", False))
    social_weight_use: Optional[float]
    if disable_social:
        if getattr(args, "social_weight", None) not in (None, 0.0):
            print("Aviso: --disable-social activo; se ignorará --social-weight.")
        social_weight_use = None  # compute_signals ya forzará w_soc=0
    else:
        if getattr(args, "social_weight", None) is not None:
            sw_raw = float(getattr(args, "social_weight"))
            sw_use = clamp(sw_raw, 0.0, 0.6)
            if sw_use != sw_raw:
                print(f"Aviso: --social-weight {sw_raw} fuera de [0.0,0.6]; ajustado a {sw_use}.")
            social_weight_use = sw_use
        else:
            social_weight_use = None  # activa autoajuste

    concentrate_enabled = bool(getattr(args, "concentrate", True))
    only_bit2me = bool(getattr(args, "only_bit2me", True))

    client = CoinGeckoClient()

    # Modo simulación (paper trading)
    if getattr(args, "simulate", False):
        sim_coin = (getattr(args, "sim_coin", "") or "").strip()
        if not sim_coin:
            print("Para usar --simulate necesitas indicar --sim-coin (ID de CoinGecko). Ej.: --sim-coin pepe")
            return 2
        tp = normalize_pct(getattr(args, "sim_tp", 10.0))
        sl = normalize_pct(getattr(args, "sim_sl", 8.0))
        res = simulate_trade(
            client,
            sim_coin,
            vs=vs_use,
            days=int(max(1, getattr(args, "sim_days", 30))),
            initial_balance=max(0.0, float(getattr(args, "sim_balance", 100.0))),
            tp_pct=tp,
            sl_pct=sl,
            max_attempts=int(max(1, getattr(args, "sim_attempts", 3)))
        )
        print_simulation_report(res)
        return 0

    try:
        if args.coins.strip():
            ids = [x.strip() for x in args.coins.split(",") if x.strip()]
            markets = client.markets_by_ids(ids, vs=vs_use)
            # Si algún id no devolvió datos, intentamos unir con categoría memes
            if len(markets) < len(ids):
                extra = fetch_markets(client, vs_use, top=max(0, top_use - len(markets)))
                # Evitar duplicados
                existing_ids = {m.get("id") for m in markets}
                for e in extra:
                    if e.get("id") not in existing_ids:
                        markets.append(e)
                markets = markets[:top_use]
        else:
            markets = fetch_markets(client, vs_use, top_use)
    except Exception as e:
        print(f"Error al obtener mercados: {e}")
        return 1

    # Aplicar filtro de Bit2Me si está activado
    if only_bit2me:
        markets_bit2me, removed_ids = filter_bit2me_markets(client, markets)
        if not markets_bit2me:
            print("Filtro Bit2Me: ninguna de las monedas candidatas está listada en Bit2Me. Usa --no-bit2me para desactivar este filtro.")
        else:
            if removed_ids:
                try:
                    print(f"Filtro Bit2Me: se excluyeron {len(removed_ids)} monedas no listadas en Bit2Me.")
                except Exception:
                    pass
        markets = markets_bit2me

    try:
        results = analyze_markets(client, markets, vs_use, days_use, social_weight=social_weight_use, disable_social=disable_social, ai_mode=getattr(args, 'ai_mode', 'advanced'), ai_weight=ai_weight_use)
    except Exception as e:
        print(f"Error en el análisis: {e}")
        return 1

    try:
        allocations = sizing_recommendations(
            results,
            budget_use,
            risk_use,
            concentrate_enabled=concentrate_enabled,
            concentrate_threshold=concentrate_threshold_use
        )
    except Exception as e:
        print(f"Error al calcular asignaciones: {e}")
        allocations = []

    print_report(results, allocations, budget_use, risk_use, vs_use)

    return 0


if __name__ == "__main__":
    sys.exit(main())
