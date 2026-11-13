"""MQTT subscriber: stores telemetry in SQLite, reports latency and delivery ratio.
   python -m iot.collector --db results/telemetry.db --duration 60"""
import argparse, json, sqlite3, time, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from iot.virtual_bins import make_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--db", default="results/telemetry.db"); ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--expected", type=int, default=0, help="messages the publisher will send (for delivery ratio)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.db) or ".", exist_ok=True)
    db = sqlite3.connect(a.db, check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS telemetry(bin_id TEXT, seq INT, sim_hour INT, fill REAL, battery REAL, rssi INT, sent_at REAL, recv_at REAL)")
    lat, seqs = [], set()

    def on_msg(c, u, m):
        d = json.loads(m.payload); now = time.time()
        db.execute("INSERT INTO telemetry VALUES(?,?,?,?,?,?,?,?)",
                   (d["bin_id"], d["seq"], d["sim_hour"], d["fill"], d["battery"], d["rssi"], d["sent_at"], now))
        lat.append((now - d["sent_at"]) * 1000); seqs.add(d["seq"])

    def on_conn(c, u, f, rc, *args):
        c.subscribe("smartwaste/bin/+/telemetry", qos=1)
    cli = make_client("collector"); cli.on_connect = on_conn; cli.on_message = on_msg
    cli.connect(a.host, a.port, 60); cli.loop_start(); time.sleep(a.duration); cli.loop_stop(); db.commit()
    if lat:
        exp = a.expected or (max(seqs) if seqs else 1)
        print(f"received={len(seqs)}  delivery_ratio={len(seqs)/exp:.3f}  latency_ms mean={np.mean(lat):.2f} "
              f"p95={np.percentile(lat,95):.2f} max={np.max(lat):.2f}")
    else:
        print("no messages received")


if __name__ == "__main__":
    main()
