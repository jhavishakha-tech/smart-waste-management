"""Virtual smart-bin field + discrete-time (1 h) simulation engine."""
import numpy as np
import pandas as pd
import config as C
import routing as R


class BinField:
    """Places n bins around real Bengaluru areas and generates hourly waste-generation series."""

    def __init__(self, n_bins, seed=0):
        self.n = n_bins
        self.seed = seed
        rng = np.random.default_rng(seed)
        names = list(C.AREAS)
        w = np.array([C.AREAS[a]["weight"] for a in names]); w = w / w.sum()
        area_idx = rng.choice(len(names), size=n_bins, p=w)
        lat, lon, kind, base = [], [], [], []
        for a in area_idx:
            A = C.AREAS[names[a]]
            dlat = rng.normal(0, A["spread_km"] / 111.0)
            dlon = rng.normal(0, A["spread_km"] / (111.0 * np.cos(np.radians(A["lat"]))))
            lat.append(A["lat"] + dlat); lon.append(A["lon"] + dlon)
            kind.append(A["kind"])
            m = np.clip(rng.lognormal(0, C.RATE_MULT_SIGMA), *C.RATE_MULT_RANGE)
            base.append(A["rate"] / 100.0 * C.RATE_SCALE * m)          # fraction of capacity per hour
        self.df = pd.DataFrame(dict(
            bin_id=[f"BIN-{i+1:03d}" for i in range(n_bins)],
            area=[names[a] for a in area_idx], kind=kind, lat=lat, lon=lon, base_rate=base))
        self.area_idx = area_idx
        self.kind_idx = np.array([C.KINDS.index(k) for k in kind])
        self.lat = np.array(lat); self.lon = np.array(lon)
        self.dist = R.distance_matrix_m(np.r_[C.DEPOT["lat"], self.lat], np.r_[C.DEPOT["lon"], self.lon])

    def generate(self, hours, seed=None):
        """Hourly generation (fraction of bin capacity) array (hours, n). Day 0 = Monday 00:00."""
        rng = np.random.default_rng(self.seed * 7919 + 13 if seed is None else seed)
        days = int(np.ceil(hours / 24))
        n_area = len(C.AREAS)
        # AR(1) day factor per area
        af = np.ones((days, n_area))
        z = rng.normal(0, C.DAY_FACTOR_SIGMA, n_area)
        for d in range(days):
            z = C.DAY_FACTOR_RHO * z + rng.normal(0, C.DAY_FACTOR_SIGMA, n_area)
            af[d] = np.exp(z)
        spikes = np.where(rng.random((days, self.n)) < C.SPIKE_PROB, C.SPIKE_MULT, 1.0)
        prof = np.stack([C.HOURLY_PROFILE[k] for k in C.KINDS])            # (4,24)
        wk = np.array([C.WEEKEND_FACTOR[k] for k in C.KINDS])
        g = np.zeros((days * 24, self.n))
        for d in range(days):
            weekend = (d % 7) >= 5
            fac = (wk[self.kind_idx] if weekend else 1.0)
            day_mult = af[d][self.area_idx] * spikes[d] * fac * self.df.base_rate.values
            hourly = prof[self.kind_idx][:, :].T                           # (24, n)
            hour_noise = rng.lognormal(0, 0.25, (24, self.n))
            g[d * 24:(d + 1) * 24] = hourly * hour_noise * day_mult
        return g[:hours]


class SmartWasteSystem:
    """Step-wise engine: sensors -> (forecast) -> collection decision -> VRP -> metrics."""

    def __init__(self, n_bins, seed=0, hours=24 * 16, warmup_days=C.WARMUP_DAYS,
                 noise=C.SENSOR_NOISE_SD, loss=C.PACKET_LOSS):
        self.field = BinField(n_bins, seed)
        self.n = n_bins
        self.T = hours
        self.rng = np.random.default_rng(seed + 101)
        self.gen = self.field.generate(hours)
        self.noise, self.loss = noise, loss
        self.warm = warmup_days * 24
        self.t = -1
        self.fill = np.zeros(n_bins)
        self.obs = np.zeros(n_bins)
        self.last_obs = np.zeros(n_bins)
        self.ghat = np.zeros((hours, n_bins))
        self.fill_hist = np.zeros((hours, n_bins))
        self.obs_hist = np.zeros((hours, n_bins))
        self.reset_metrics()

    def reset_metrics(self):
        self.m = dict(distance_km=0.0, trips=0, collections=0, overflow_bin_hours=0, overflow_events=0,
                      lost=0.0, generated=0.0, collected_L=0.0, fill_at_collect=[], dropped=0,
                      service_h=0.0, dispatch_days=0, daily=[], measured_hours=0, travel_h=0.0)

    # ---- one hour of waste generation + sensing ----
    def step(self):
        self.t += 1; t = self.t
        prev_full = self.fill >= 0.999
        new = self.fill + self.gen[t]
        lost = np.maximum(new - 1.0, 0.0)
        self.fill = np.minimum(new, 1.0)
        reading = np.clip(self.fill + self.rng.normal(0, self.noise, self.n), 0, 1) if self.noise > 0 else self.fill.copy()
        if self.loss > 0:
            ok = self.rng.random(self.n) >= self.loss
            reading = np.where(ok, reading, self.last_obs)
        self.obs = reading
        self.ghat[t] = self.obs - self.last_obs
        self.last_obs = self.obs.copy()
        self.fill_hist[t] = self.fill; self.obs_hist[t] = self.obs
        if t >= self.warm:
            full = self.fill >= 0.999
            self.m["overflow_bin_hours"] += int(full.sum())
            self.m["overflow_events"] += int((full & ~prev_full).sum())
            self.m["lost"] += float(lost.sum()); self.m["generated"] += float(self.gen[t].sum())
            self.m["measured_hours"] += 1

    def is_decision_time(self):
        return self.t % 24 == C.DECISION_HOUR

    # ---- policies ----
    def select(self, policy, forecaster=None, theta=C.THETA, threshold=C.THRESHOLD):
        """Return (indices of bins to collect, predicted 24h accumulation or None)."""
        day = self.t // 24
        pred = None
        if self.t < self.warm:
            return np.arange(self.n), pred
        if policy == "fixed_daily":
            sel = np.arange(self.n)
        elif policy == "fixed_alt":
            sel = np.where(self.field.area_idx % 2 == day % 2)[0]
        elif policy == "threshold":
            sel = np.where(self.obs >= threshold)[0]
        elif policy == "proposed":
            pred = forecaster.predict(self.ghat, self.t, self.field)
            sel = np.where(self.obs + pred >= theta)[0]
        else:
            raise ValueError(policy)
        return sel, pred

    def collect(self, sel, time_limit_s=0.3):
        """Plan routes for selected bins with the VRP solver, empty them, update metrics."""
        sel = np.asarray(sel, dtype=int)
        out = dict(routes=[], distance_km=0.0, trips=0, bins=[], dropped=0)
        if len(sel) == 0:
            if self.t >= self.warm: self.m["daily"].append((self.t // 24, 0, 0, 0.0))
            return out
        nodes = np.r_[0, sel + 1]
        D = self.field.dist[np.ix_(nodes, nodes)]
        demand = np.r_[0, np.maximum(np.ceil(np.clip(self.obs[sel], 0, 1) * C.BIN_VOLUME_L), 1)].astype(int)
        res = R.solve_vrp(D, demand, time_limit_s=time_limit_s)
        served_local = [k for r in res["routes"] for k in r]
        served = sel[np.array(served_local) - 1] if served_local else np.array([], dtype=int)
        dist_km = res["distance_m"] / 1000.0
        if self.t >= self.warm:
            self.m["distance_km"] += dist_km; self.m["trips"] += len(res["routes"])
            self.m["collections"] += len(served); self.m["dropped"] += len(res["dropped"])
            self.m["collected_L"] += float((self.fill[served] * C.BIN_VOLUME_L).sum())
            self.m["fill_at_collect"].extend(self.fill[served].tolist())
            self.m["service_h"] += len(served) * C.SERVICE_MIN_PER_BIN / 60.0
            self.m["travel_h"] += dist_km / C.AVG_SPEED_KMPH
            self.m["dispatch_days"] += 1
            self.m["daily"].append((self.t // 24, len(served), len(res["routes"]), dist_km))
        self.fill[served] = 0.0
        self.last_obs[served] = 0.0
        out.update(routes=[[int(sel[k - 1]) for k in r] for r in res["routes"]],
                   distance_km=dist_km, trips=len(res["routes"]), bins=served.tolist(),
                   dropped=len(res["dropped"]))
        return out

    def summary(self):
        m = self.m
        mh = max(m["measured_hours"], 1)
        days = max(mh / 24.0, 1e-9)
        fac = np.array(m["fill_at_collect"]) if m["fill_at_collect"] else np.array([np.nan])
        fuel = m["distance_km"] * C.FUEL_L_PER_KM
        return dict(
            distance_km=m["distance_km"], distance_km_per_day=m["distance_km"] / days,
            trucks_per_day=m["trips"] / days, bins_collected_per_day=m["collections"] / days,
            overflow_pct=100.0 * m["overflow_bin_hours"] / (mh * self.n),
            overflow_events_per_bin_week=m["overflow_events"] / self.n / (days / 7.0),
            lost_waste_pct=100.0 * m["lost"] / max(m["generated"], 1e-9),
            avg_fill_at_collection=float(np.nanmean(fac)),
            fuel_L=fuel, co2_kg=fuel * C.CO2_KG_PER_L,
            route_hours_per_day=(m["travel_h"] + m["service_h"]) / days,
            dropped=m["dropped"], days=days)


def run_simulation(n_bins, policy, seed, days=14, forecaster=None, theta=C.THETA,
                   noise=C.SENSOR_NOISE_SD, loss=C.PACKET_LOSS, time_limit_s=0.3, threshold=C.THRESHOLD):
    sysm = SmartWasteSystem(n_bins, seed, hours=(days + C.WARMUP_DAYS) * 24, noise=noise, loss=loss)
    for _ in range(sysm.T):
        sysm.step()
        if sysm.is_decision_time():
            sel, _ = sysm.select(policy, forecaster, theta, threshold)
            sysm.collect(sel, time_limit_s)
    return sysm.summary()
