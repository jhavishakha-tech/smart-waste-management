"""Waste classifier interface.
PHASE 1: no trained weights are shipped -> `available` is False and the app lets you pick the class
manually (clearly labelled 'demo mode').
PHASE 2: train with train_classifier.py (Colab), copy `waste_model.tflite` / `waste_model.keras`
into classifier/ and `available` becomes True automatically."""
import os, json
import numpy as np

CLASSES = ["plastic", "paper", "metal", "glass", "organic", "hazardous", "other"]
# class -> smart-bin compartment (Indian dry / wet / hazardous segregation)
COMPARTMENT = {"plastic": "Dry", "paper": "Dry", "metal": "Dry", "glass": "Dry",
               "organic": "Wet", "hazardous": "Hazardous", "other": "Reject"}
COMPARTMENTS = ["Wet", "Dry", "Hazardous", "Reject"]
CONF_THRESHOLD = 0.60        # below this the item goes to the 'Reject / manual check' compartment
HERE = os.path.dirname(__file__)


def route_item(label, confidence, conf_threshold=CONF_THRESHOLD):
    """Confidence-based rejection: low-confidence items are not trusted to the automatic compartments."""
    return "Reject" if confidence < conf_threshold else COMPARTMENT[label]


class WasteClassifier:
    def __init__(self):
        self.model = None
        self.kind = None
        meta = os.path.join(HERE, "labels.json")
        self.classes = json.load(open(meta)) if os.path.exists(meta) else CLASSES
        tfl, keras_p = os.path.join(HERE, "waste_model.tflite"), os.path.join(HERE, "waste_model.keras")
        try:
            if os.path.exists(tfl):
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    from tensorflow.lite import Interpreter
                self.model = Interpreter(model_path=tfl); self.model.allocate_tensors(); self.kind = "tflite"
            elif os.path.exists(keras_p):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(keras_p); self.kind = "keras"
        except Exception as e:      # missing TF etc. -> stay in demo mode
            print("Classifier not loaded:", e); self.model = None

    @property
    def available(self):
        return self.model is not None

    def predict(self, pil_image, size=224):
        """Return (label, confidence, probs). Raises if no model is loaded."""
        if not self.available:
            raise RuntimeError("No trained model found (demo mode).")
        x = np.asarray(pil_image.convert("RGB").resize((size, size)), dtype=np.float32)[None]
        if self.kind == "keras":
            p = self.model.predict(x, verbose=0)[0]
        else:
            inp = self.model.get_input_details()[0]; out = self.model.get_output_details()[0]
            if inp["dtype"] == np.uint8:
                x = x.astype(np.uint8)
            self.model.set_tensor(inp["index"], x); self.model.invoke()
            p = self.model.get_tensor(out["index"])[0].astype(np.float32)
            if p.sum() > 1.01 or p.min() < 0:
                p = np.exp(p - p.max()); p /= p.sum()
        i = int(np.argmax(p))
        return self.classes[i], float(p[i]), p
