"""Virtual ESP32 bins -> MQTT publisher (uses the same simulator as the experiments).
   python -m iot.virtual_bins --bins 20 --speed 0.5 --hours 48        (0.5 s wall-clock per simulated hour)
Topic: smartwaste/bin/<bin_id>/telemetry   payload JSON: {bin_id, seq, sim_hour, fill, battery, rssi, sent_at}"""
import argparse, json, time, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import paho.mqtt.client as mqtt
from simulator import SmartWasteSystem


def make_client(cid):
    try:                                   # paho-mqtt >= 2.0
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
    except AttributeError:                 # paho-mqtt 1.x
        return mqtt.Client(client_id=cid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--bins", type=int, default=20); ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--speed", type=float, default=0.5, help="seconds of wall-clock per simulated hour")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--qos", type=int, default=1)
    a = ap.parse_args()
    sysm = SmartWasteSystem(a.bins, a.seed, hours=a.hours)
    cli = make_client("virtual-bins"); cli.connect(a.host, a.port, 60); cli.loop_start()
    ids = sysm.field.df.bin_id.tolist(); battery = np.full(a.bins, 100.0); rng = np.random.default_rng(a.seed)
    seq = 0
    for _ in range(a.hours):
        sysm.step()
        if sysm.is_decision_time():                       # (optional) collection - threshold policy
            sysm.collect(sysm.select("threshold")[0], 0.1)
        battery -= rng.uniform(0.005, 0.02, a.bins)
        for i, b in enumerate(ids):
            seq += 1
            msg = dict(bin_id=b, seq=seq, sim_hour=sysm.t, fill=round(float(sysm.obs[i]), 4),
                       battery=round(float(battery[i]), 2), rssi=int(rng.integers(-90, -50)), sent_at=time.time())
            cli.publish(f"smartwaste/bin/{b}/telemetry", json.dumps(msg), qos=a.qos)
        time.sleep(a.speed)
    cli.loop_stop(); cli.disconnect(); print("published", seq, "messages")


if __name__ == "__main__":
    main()
