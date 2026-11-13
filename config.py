"""Global configuration & assumptions for the Smart Waste simulation.
Every number here is an ASSUMPTION and must be stated in the paper (Simulation Setup section).
"""
import numpy as np

# ---- Study area: Bengaluru (real area centres; bins are scattered around them) ----
DEPOT = dict(name="Central Depot (Yeshwanthpur)", lat=13.0285, lon=77.5409)

# kind -> behaviour; rate = mean fill gain in % of bin capacity per hour (before profile)
AREAS = {
    "Majestic / KR Market": dict(lat=12.9767, lon=77.5713, kind="market",      rate=2.8, spread_km=1.0, weight=1.3),
    "Malleshwaram":         dict(lat=13.0035, lon=77.5647, kind="residential", rate=2.0, spread_km=1.2, weight=1.0),
    "Indiranagar":          dict(lat=12.9784, lon=77.6408, kind="commercial",  rate=2.6, spread_km=1.2, weight=1.1),
    "Koramangala":          dict(lat=12.9352, lon=77.6245, kind="commercial",  rate=2.5, spread_km=1.3, weight=1.2),
    "Jayanagar":            dict(lat=12.9250, lon=77.5938, kind="residential", rate=2.0, spread_km=1.3, weight=1.0),
    "HSR Layout":           dict(lat=12.9116, lon=77.6474, kind="residential", rate=2.1, spread_km=1.4, weight=0.9),
    "Whitefield":           dict(lat=12.9698, lon=77.7500, kind="tech",        rate=2.2, spread_km=1.8, weight=0.9),
    "Electronic City":      dict(lat=12.8452, lon=77.6602, kind="tech",        rate=2.3, spread_km=1.8, weight=0.8),
    "Hebbal":               dict(lat=13.0358, lon=77.5970, kind="residential", rate=1.9, spread_km=1.3, weight=0.7),
    "Yelahanka":            dict(lat=13.1007, lon=77.5963, kind="residential", rate=1.8, spread_km=1.6, weight=0.6),
}
KINDS = ["market", "commercial", "residential", "tech"]

WEEKEND_FACTOR = dict(market=1.15, commercial=1.10, residential=1.25, tech=0.35)

def _profile(peaks):
    """24-value hourly profile (mean 1) built from Gaussian peaks [(hour, width, height), ...]."""
    h = np.arange(24)
    p = np.full(24, 0.25)
    for mu, w, a in peaks:
        d = np.minimum(abs(h - mu), 24 - abs(h - mu))
        p = p + a * np.exp(-0.5 * (d / w) ** 2)
    return p / p.mean()

HOURLY_PROFILE = dict(
    market=_profile([(8, 2.5, 2.0), (17, 2.0, 1.2)]),
    commercial=_profile([(13, 2.5, 1.4), (20, 2.5, 1.6)]),
    residential=_profile([(8, 2.0, 1.8), (19, 2.5, 1.5)]),
    tech=_profile([(13, 1.8, 1.8), (18, 2.0, 1.2)]),
)

# ---- Bin / sensor / truck assumptions ----
BIN_VOLUME_L = 1100          # standard 1100 L municipal container
SENSOR_NOISE_SD = 0.02       # HC-SR04 style ultrasonic: ~2 % of bin height
PACKET_LOSS = 0.0            # probability a hourly reading is lost (MQTT/LoRa)
RATE_SCALE = 0.55            # global calibration: mean bin gains ~25-35 % of capacity per day
RATE_MULT_RANGE = (0.3, 1.8) # per-bin heterogeneity (log-normal, clipped)
RATE_MULT_SIGMA = 0.45
DAY_FACTOR_SIGMA = 0.12      # day-to-day area-level variation (AR(1))
DAY_FACTOR_RHO = 0.5
SPIKE_PROB = 0.02            # prob. per bin-day of an event spike (festival, market day)
SPIKE_MULT = 2.0

TRUCK_CAPACITY_L = 16000     # effective (compacted) capacity in bin-litres
MAX_ROUTE_KM = 150.0         # per-vehicle distance limit
ROAD_FACTOR = 1.35           # road distance / straight-line distance (urban circuity)
AVG_SPEED_KMPH = 22.0        # Bengaluru average truck speed
SERVICE_MIN_PER_BIN = 4.0
FUEL_L_PER_KM = 0.5          # diesel collection truck (assumption)
CO2_KG_PER_L = 2.68

# ---- Simulation ----
DECISION_HOUR = 6            # routes planned/dispatched at 06:00
WARMUP_DAYS = 2              # all policies use daily collection during warm-up
POLICIES = ["fixed_daily", "fixed_alt", "threshold", "proposed"]
POLICY_LABEL = {
    "fixed_daily": "Fixed (daily, all bins)",
    "fixed_alt":   "Fixed (alternate-day)",
    "threshold":   "Threshold (fill ≥ 70 %)",
    "proposed":    "Proposed (forecast-based)",
}
THRESHOLD = 0.70
THETA = 0.90                 # proposed: collect if predicted fill at next dispatch >= THETA
