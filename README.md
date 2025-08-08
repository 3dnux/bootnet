# Bot de análisis de Memecoins

AVISO: Este software es únicamente informativo/educativo y NO es asesoría financiera. Las criptomonedas son altamente volátiles; invierta bajo su propio riesgo.

## ¿Qué hace este bot?

- Consulta datos públicos de CoinGecko (sin librerías externas) para analizar memecoins.
- Calcula señales técnicas y de entorno: RSI, SMA20/50, Bandas de Bollinger, volatilidad de retornos, puntajes de liquidez/social, y un puntaje de compra (buy_score).
- Emite una recomendación por moneda: Comprar, Retener, Vender o Evitar.
- Sugiere una asignación de presupuesto según su perfil de riesgo y muestra escenarios estimados de ganancia potencial (Take Profit) y pérdida potencial (Stop Loss) por moneda y para la canasta total, junto con el ratio riesgo/beneficio (R/R).

## Requisitos

- Python 3.8+ (sin dependencias externas).
- Conexión a Internet para análisis en vivo (a menos que use `--offline`).

## Uso rápido

Desde la carpeta del proyecto:

```
python main.py --capital 1500 --risk moderado --top 8 --days 90 --vs usd
```

Parámetros principales:

- `--capital` (alias de `--budget`): presupuesto total con el que desea trabajar (por defecto 1000). Ej.: `--capital 1500`.
- `--risk`: perfil de riesgo: `bajo` | `moderado` | `alto` (por defecto `moderado`). Define el porcentaje objetivo del presupuesto que el bot intenta asignar:
  - bajo: ~30%
  - moderado: ~50%
  - alto: ~70%
- `--top`: número de memecoins a analizar (por defecto 8).
- `--days`: días de historial para análisis técnico (por defecto 90; mínimo interno 30).
- `--vs`: moneda base de precios (por defecto `usd`).
- `--coins`: IDs específicos de CoinGecko separados por coma para forzar el análisis de esas monedas (opcional). Ej.: `--coins pepe,dogecoin,shiba-inu`.
- `--social-weight`: peso del componente social en el buy_score (0.0–0.6). Por defecto 0.20 si no se especifica.
- `--disable-social`: desactiva el componente social (usa solo señales técnicas/liquidez/riesgo).
- `--concentrate` / `--no-concentrate`: activa o desactiva la concentración de capital en una sola operación cuando la mejor candidata es muy buena (por defecto activado).
- `--concentrate-threshold`: umbral del `buy_score` (0.0–1.0) para considerar "muy buena" y concentrar. Por defecto 0.85.
- `--ai-mode`: IA: `off` | `basic` | `advanced` (por defecto `advanced`).
- `--ai-weight`: peso de la IA al combinar con `buy_score` (0.0–1.0). Por defecto 0.35.
- `--offline`: evita llamadas HTTP y muestra un ejemplo/ayuda.

Ejemplos:

```
# Moderado, 1200 USD, analiza 10 monedas, 120 días de historial
python main.py --capital 1200 --risk moderado --top 10 --days 120

# Alto riesgo, 500 USD, solo estas monedas
python main.py --capital 500 --risk alto --coins pepe,dogecoin,shiba-inu
```

## ¿Cómo calcula la estrategia y los escenarios TP/SL?

Para cada moneda, el bot:

1. Calcula indicadores (RSI, SMA20/50, Bandas de Bollinger, volatilidad de retornos, puntajes sociales y de liquidez).
2. Combina estos factores en un `buy_score` y aplica reglas heurísticas para definir la acción:
   - Comprar: tendencia y momentum saludables, con métricas de liquidez/social aceptables.
   - Retener: señales mixtas o sobreventa con buen soporte; sugiere acumular o esperar confirmación.
   - Vender: sobreextensión (p. ej., RSI alto y precio pegado a banda superior tras subidas fuertes).
   - Evitar: riesgo elevado (liquidez/capitalización muy baja o datos insuficientes).
3. Determina niveles guía de gestión de riesgo según volatilidad reciente:
   - `stop_loss_pct` ≈ entre 5% y 15%.
   - `take_profit_pct` ≈ entre 8% y 30%.
4. Genera un texto de estrategia, visible en el reporte: por ejemplo “Tendencia y momentum: entrar y gestionar con TP ~ X% y SL ~ Y%”.

### Asignación y escenarios de P/L

- El bot intenta distribuir el presupuesto según el perfil de riesgo y el `buy_score` relativo de cada moneda, respetando límites por moneda.
- Para cada asignación, el reporte muestra:
  - Estrategia (texto),
  - Precio de Stop Loss y Take Profit,
  - Ganancia potencial (si llega a TP) y Pérdida potencial (si toca SL), en dinero y porcentaje,
  - Ratio R/R (riesgo/beneficio), y totales para la canasta.

Nota: Los escenarios TP/SL son estimaciones simples basadas en variaciones porcentuales desde el precio actual; no garantizan que el precio alcance esos niveles.

## Concentración de capital (si la operación es muy buena)

- Cuando está activado (por defecto), si la mejor candidata tiene acción "Comprar" y su `buy_score` es mayor o igual al umbral `--concentrate-threshold`, el bot concentrará el porcentaje objetivo total de su presupuesto (según perfil de riesgo) en esa única moneda.
- Esto ignora el límite por moneda sólo en ese caso de convicción alta.
- Para desactivar este comportamiento: use `--no-concentrate`.
- Para ajustar el umbral de convicción: use `--concentrate-threshold` (ej., `0.90`).
- El reporte avisará explícitamente con la leyenda `[Modo concentración]` cuando se aplique.

Ejemplos:
```
# Concentrar sólo si la mejor supera umbral 0.90
python main.py --capital 1000 --risk moderado --concentrate --concentrate-threshold 0.90

# Desactivar concentración (distribución proporcional habitual)
python main.py --capital 1000 --risk moderado --no-concentrate
```

## Salida (extracto de ejemplo)

```
Análisis de Memecoins — 2025-08-07 21:27  |  Moneda base: USD  |  Perfil: Moderado
Presupuesto de trabajo: $1,200.00  |  Uso objetivo según perfil: 50.00% -> $600.00

Pepe (PEPE)  Precio: $0.00000234  MC: 1.25B  Vol: 123.45M
  Cambios: 1h +0.50% | 24h +2.10% | 7d +12.30% | 30d +45.00%
  RSI14: 58.2  | SMA20: 0.00000210  | SMA50: 0.00000190  | BB Pos: 63.4% dentro del canal
  Puntajes — Buy: 0.68  | Tendencia: 0.85  | Momentum: 0.90  | Social: 0.70  | Liquidez: 0.65  | Riesgo: 0.55
  Recomendación: Comprar  | Stop Loss ~ 8.00%  | Take Profit ~ 16.00%
  Estrategia: Tendencia y momentum: entrar y gestionar con TP ~ 16.0% y SL ~ 8.0%.
   - Tendencia y momentum saludables con liquidez/social aceptables.

Asignación sugerida (NO es asesoría financiera):
  - Pepe (PEPE): 6.00% del presupuesto -> $72.00 @ $0.00000234
      Stop $0.00000215  |  TP $0.00000271
      Estrategia: Tendencia y momentum: entrar y gestionar con TP ~ 16.0% y SL ~ 8.0%.
      + Ganancia potencial: $11.52 (+16.00%)  |  - Pérdida potencial: $5.76 (-8.00%)  |  R/R: 2.00
  Total asignado: $600.00 de $1,200.00
  Escenario de la canasta — +Ganancia potencial: $150.00  |  -Pérdida potencial: $90.00  |  R/R total: 1.67
```

## Señales sociales avanzadas (redes, no noticias)

Este bot no lee noticias. Para “buscar en redes”, utiliza únicamente `community_data` de CoinGecko como proxy de actividad en redes y comunidades de cada proyecto. Los campos considerados (cuando están disponibles) incluyen:
- Twitter/X: `twitter_followers` (alcance)
- Reddit: `reddit_subscribers` (alcance), `reddit_average_posts_48h`, `reddit_average_comments_48h`, `reddit_accounts_active_48h` (engagement y actividad reciente)
- Telegram: `telegram_channel_user_count` (alcance)
- Facebook: `facebook_likes` (alcance)

Técnicas utilizadas para un puntaje social más avanzado:
- Alcance (reach) con escalado logarítmico: followers/subscribers se llevan a [0,1] usando log10 (p. ej. 10^6 -> ~1.0) y ponderaciones por plataforma.
- Engagement: comentarios por post (10 c/p ~ 1.0), cuentas activas/total y “buzz” (posts+comentarios en 48h), todo normalizado a [0,1].
- Combinación: 65% alcance + 35% engagement, con un pequeño “bonus” si ambos son altos y penalización si ambos son muy bajos.
- Control del peso social en el `buy_score` con `--social-weight` (0.0–0.6) o desactivación total con `--disable-social`.

Limitaciones conocidas:
- `community_data` puede ser nulo o estar desactualizado para algunas monedas.
- No se usan APIs de X/Reddit ni scraping; sólo los agregados que expone CoinGecko.
- No se consumen fuentes de noticias.

Ejemplos:
```
# Dar más peso a redes (40%)
python main.py --social-weight 0.4

# Analizar solo señales técnicas/liquidez/riesgo, sin redes
python main.py --disable-social
```

## IA avanzada (opcional)

El bot incorpora una capa de IA ligera sin dependencias externas: un modelo de regresión logística entrenado al vuelo con el historial reciente de precios e indicadores por moneda. Su objetivo es estimar la probabilidad de que el precio suba en el corto plazo y mezclar dicha probabilidad con el `buy_score` heurístico.

Cómo funciona:
- Características (features) por punto temporal: RSI normalizado, relación SMA20/50 normalizada, posición en Bandas de Bollinger, volatilidad normalizada, `risk_score`, retornos recientes (1, 3 y 7 pasos), `social_score` y `liquidity_score`.
- Etiquetas (targets): subida (> 0%) del precio a un horizonte fijo.
  - Modo `basic`: horizonte 1.
  - Modo `advanced`: horizonte 3.
- Entrenamiento: gradiente descendente con regularización L2 y estandarización interna de features, usando muestras agregadas de todas las monedas analizadas en la corrida actual.
- Mezcla con la heurística: `buy_final = (1 - ai_weight) * buy_base + ai_weight * P(subida)`.
- Reporte: verás `Buy: X.XX (IA Y.YY | Base Z.ZZ)` y una razón adicional del tipo “IA avanzada: probabilidad de subida ~ 65.0% (peso 0.35)”.

Parámetros de IA:
- `--ai-mode`: `off` | `basic` | `advanced` (por defecto `advanced`).
- `--ai-weight`: peso de la IA al combinar con `buy_score` (0.0–1.0). Por defecto `0.35`.

Notas:
- Si usas `--disable-social`, la IA no utilizará las señales sociales para sus features.
- En modo `--offline`, no se ejecuta el análisis ni el entrenamiento de IA (se muestra ayuda/ejemplo).

Ejemplos:
```
# IA avanzada (default) con más influencia (50%)
python main.py --ai-mode advanced --ai-weight 0.5

# IA básica (horizonte más corto) con menor peso
python main.py --ai-mode basic --ai-weight 0.25

# Desactivar IA por completo (usa sólo heurística base)
python main.py --ai-mode off
```

## Consejos y limitaciones

- Si no aparece una canasta clara, el bot sugerirá esperar confirmación.
- El mercado cripto es extremadamente volátil; considere incorporar stops y un tamaño de posición responsable.
- La API pública de CoinGecko puede imponer límites de frecuencia (HTTP 429). El bot ya implementa reintentos con backoff simple.

## Modo offline

Para ver ayuda rápida sin llamar a la API:

```
python main.py --offline
```

## Licencia

Uso personal/educativo. Sin garantías. No responsabilidad por pérdidas.


## Modo simulación (paper trading)

Emula una operación con Take Profit (TP) y Stop Loss (SL) usando datos reales de CoinGecko, sin ejecutar órdenes reales. Si el precio alcanza el TP, marca éxito y suma la ganancia al balance; si toca SL, descuenta la pérdida del balance y reintenta con lo que queda hasta agotar intentos.

Uso básico:

```
python main.py --simulate --sim-coin pepe --sim-balance 100 --sim-tp 10 --sim-sl 8 --sim-days 30 --sim-attempts 3 --vs usd
```

Parámetros:
- `--simulate`: activa el modo simulación.
- `--sim-coin`: ID de la moneda en CoinGecko (ej. `pepe`, `shiba-inu`).
- `--sim-balance`: balance inicial para la simulación (por defecto 100).
- `--sim-tp`: Take Profit en % o decimal (ej. `10` o `0.10`).
- `--sim-sl`: Stop Loss en % o decimal (ej. `8` o `0.08`).
- `--sim-days`: días de historial a usar (por defecto 30). Para < 90 días CoinGecko devuelve precios por hora; para ≥ 90 días, diarios.
- `--sim-attempts`: intentos máximos tras SL para reintentar con el balance restante (por defecto 3).

Notas:
- La simulación recorre la serie de precios y, entre puntos discretos, prioriza TP si se cruzaran TP y SL simultáneamente.
- Si no se alcanza TP ni SL dentro del periodo, cierra al último precio y reporta el PnL real del periodo con resultado `FORCED`.
- Este es un modelo simple de papel; no considera deslizamiento, comisiones ni liquidez real.
