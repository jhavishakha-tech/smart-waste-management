# Intelligent Waste Segregation & Smart Collection - AI + IoT (simulation)

Bengaluru-based simulation of smart bins (virtual ESP32 + ultrasonic sensors) -> MQTT -> forecasting -> capacitated VRP routing,
with an AI waste-classification interface (Phase 2) and a Streamlit dashboard.

## Quick start
```
pip install -r requirements.txt
streamlit run app.py                 # dashboard (first start trains the forecaster, ~10 s)
python experiments.py all            # full paper experiments (~40-60 min on 1 core; resumable)
python experiments.py all --quick    # 2-minute smoke test
```
MQTT demo (needs a broker, e.g. `docker run -p 1883:1883 eclipse-mosquitto:2` with allow_anonymous, or docker-compose):
```
python -m iot.collector --duration 60 --expected 960 &
python -m iot.virtual_bins --bins 20 --hours 48 --speed 0.5
```

## Layout
config.py (ALL assumptions) | simulator.py (bins, sensing, policies, metrics) | forecasting.py | routing.py |
experiments.py | app.py | iot/ (MQTT) | esp32_wokwi/ (sketch) | classifier/ (Phase 2 interface) | results/ (CSV + figures)

## Policies compared (same OR-Tools solver; they differ only in WHICH bins are collected)
fixed_daily, fixed_alt (alternate-day by area), threshold (sensed fill >= tau), proposed (sensed + forecast 24 h accumulation >= theta).

## Results summary (10 seeds, 14 days, mean +/- std; see results/)
- Proposed vs fixed-daily: distance -24 to -28 % (p=0.002, Wilcoxon), overflow 0.18-0.21 % vs 0.07-0.08 % (slightly worse).
- Proposed vs threshold tau=0.7: +11 to +14 % distance, overflow ~94 % lower. **But tau=0.7 is a weak baseline.**
- Pareto sweep (100 bins, 5 seeds): threshold tau=0.5 gives 191 km/day at 0.38 % overflow; proposed theta=0.9 gives 191 km/day at 0.19 %.
  The forecast advantage over a *tuned* threshold is real but modest (~half the overflow at equal distance). Report it this way.
- Forecast (24 h accumulation): Gradient Boosting / MLP beat persistence baseline (see results/forecast_models.csv).
- Robustness: proposed degrades with sensor noise (0.19 % -> 0.90 % at SD 10 %) but stays below threshold (3.7 %); packet loss barely matters.

## Honest limitations (put these in the paper)
1. Fill-level data is synthetic; rate calibration (RATE_SCALE etc.) is an assumption, not municipal data.
2. Distances = haversine x 1.35, not real roads (osm_distance_matrix() in routing.py is provided but untested here).
3. Collection is instantaneous at 06:00; waste generated during the route is ignored.
4. Forecaster trained on the same generator family it is tested on (different seeds) - optimistic vs real data.
5. No GRU/LSTM here (PyTorch not installable in authoring sandbox); add as extra baseline in Colab.
6. Wokwi sketch and docker-compose were NOT tested. Streamlit app tested headless (no exceptions); MQTT tested with a local amqtt broker (60/60 delivered, ~3.7 ms).
7. Classifier tab is in demo mode until Phase 2 weights are added.
