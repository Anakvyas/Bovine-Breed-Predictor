#!/usr/bin/env python3
"""
Final app.py (Render-compatible)

- Safe model loading: supports weights saved as state_dict OR as full model object.
- /predict : POST image + milk,lact,weight,disease -> returns top3, final_pred, crossbreed_flag,
             and URLs to heatmap and silhouette overlay images (saved in static/results).
- /feedback: POST JSON {image, predicted_final, action, scores} -> saved to feedback_log.csv
- CORS enabled.
- Compatible with Render dynamic ports (uses os.environ["PORT"]).
"""

import os, io, csv, json, datetime, traceback
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# --------- CONFIG ----------
BASE = os.getcwd()
WEIGHTS = os.path.join(BASE, "best_model.pth")
TRAITS_CSV = os.path.join(BASE, "breed_traits.csv")
SILH_DIR = os.path.join(BASE, "silhouettes")
RESULTS_DIR = os.path.join(BASE, "static", "results")
CROSSLOG = os.path.join(BASE, "crossbreed_log.csv")
FEEDBACK_LOG = os.path.join(BASE, "feedback_log.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SILH_DIR, exist_ok=True)

# Replace with your breed list (order must match model training class order)
BREEDS = ["Gir", "H_F", "Murrah", "jersey", "nili_ravi"]

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("Using device:", DEVICE)

# -------- transforms ----------
transform_input = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --------- model load ----------
def load_model(weights_path, num_classes, device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    data = torch.load(weights_path, map_location=device)
    if isinstance(data, dict) and not any(hasattr(v, '__call__') for v in data.values()):
        try:
            model.load_state_dict(data)
            print("Loaded weights as state_dict.")
        except Exception:
            new_sd = {k.replace("module.", ""): v for k, v in data.items()}
            model.load_state_dict(new_sd)
            print("Loaded weights as state_dict (module. prefix removed).")
    else:
        try:
            model = data
            print("Loaded full model object from checkpoint.")
            if isinstance(model, nn.Module):
                try:
                    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
                        if getattr(model.classifier[1], "out_features", None) != num_classes:
                            in_f = model.classifier[1].in_features
                            model.classifier[1] = nn.Linear(in_f, num_classes)
                            print("Replaced classifier to match num_classes.")
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError("Unknown weights format; please provide state_dict or full model.") from e

    model.to(device)
    model.eval()
    return model

if not os.path.isfile(WEIGHTS):
    raise FileNotFoundError(f"Model weights not found at: {WEIGHTS}")
model = load_model(WEIGHTS, len(BREEDS), DEVICE)

# -------- Grad-CAM helper ----------
def compute_gradcam(model, pil_img, class_idx):
    model.zero_grad()
    x = transform_input(pil_img).unsqueeze(0).to(DEVICE)
    features = grads = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    target_module = model.features[-1]
    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_full_backward_hook(backward_hook)

    out = model(x)
    out_tensor = out[0] if isinstance(out, tuple) else out
    score = out_tensor[0, class_idx]
    score.backward(retain_graph=False)

    fmap = features.detach().cpu().numpy()[0]
    g = grads.detach().cpu().numpy()
    weights = g.mean(axis=(1,2))
    cam = np.maximum(np.sum(weights[:, None, None] * fmap, axis=0), 0)
    cam = cam / (cam.max() + 1e-8)
    fh.remove(); bh.remove()
    return cam

def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.45):
    orig = np.array(pil_img)[:,:,::-1].copy()
    hm_u8 = np.uint8(255 * cv2.resize(heatmap, (orig.shape[1], orig.shape[0])))
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 1-alpha, hm_color, alpha, 0)
    return overlay

# --------- CSV helpers ----------
def append_csv(path, rowdict):
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rowdict.keys()))
        if not exists:
            w.writeheader()
        w.writerow(rowdict)

# ---------- Flask app ----------
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error":"no image uploaded"}), 400
        f = request.files['image']
        milk = float(request.form.get('milk', 0.0))
        lact = float(request.form.get('lact', 0.0))
        weight = float(request.form.get('weight', 0.0))
        disease = float(request.form.get('disease', 0.0))

        pil = Image.open(io.BytesIO(f.read())).convert("RGB")
        x = transform_input(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

        idxs = probs.argsort()[::-1][:3]
        top3 = [(BREEDS[i], float(probs[i])) for i in idxs]
        top1_idx = idxs[0]
        pred_breed = BREEDS[top1_idx]
        top1_prob = float(probs[top1_idx])

        cam = compute_gradcam(model, pil, top1_idx)
        heat_overlay = overlay_heatmap_on_image(pil, cam)

        stamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        base = f"{os.path.splitext(f.filename)[0]}_{stamp}"
        heat_fn = f"{base}_heat.jpg"
        cv2.imwrite(os.path.join(RESULTS_DIR, heat_fn), heat_overlay)

        cross_flag = top1_prob < 0.7 or (probs[idxs[0]] - probs[idxs[1]]) < 0.15
        append_csv(CROSSLOG, {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "image_filename": f.filename,
            "predicted_cnn": pred_breed,
            "pred_conf_cnn": top1_prob,
            "top3": json.dumps(top3),
            "milk": milk, "lact": lact, "weight": weight, "disease": disease,
            "cross_flag": bool(cross_flag)
        })

        return jsonify({
            "top3": top3,
            "final_pred": pred_breed,
            "cnn_conf": top1_prob,
            "crossbreed_flag": bool(cross_flag),
            "heatmap_url": f"/static/results/{heat_fn}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json(force=True)
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "image": data.get("image"),
            "predicted_final": data.get("predicted_final"),
            "action": data.get("action"),
            "scores": json.dumps(data.get("scores")) if data.get("scores") else ""
        }
        append_csv(FEEDBACK_LOG, entry)
        return jsonify({"status":"saved"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/static/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

@app.route("/support")
def support():
    return render_template("support.html")

if __name__ == "__main__":
    print("App root:", BASE)
    print("Results dir:", RESULTS_DIR)
    print("Silhouettes dir:", SILH_DIR)

    # ✅ Render-compatible dynamic port
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
