# #!/usr/bin/env python3
# """
# Final app.py (Render-compatible, hardened)

# - Safe model loading: supports weights saved as state_dict OR as full model object.
# - /predict : POST image + milk,lact,weight,disease -> returns top3, final_pred, crossbreed_flag,
#              and a URL to a heatmap overlay (saved in static/results). If Grad-CAM fails, it gracefully skips.
# - /feedback: POST JSON {image, predicted_final, action, scores} -> saved to feedback_log.csv
# - CORS enabled.
# - Compatible with Render dynamic ports (uses os.environ["PORT"]).
# """

# import os, io, csv, json, datetime, traceback
# from typing import Tuple
# from flask import Flask, request, jsonify, send_from_directory, render_template
# from flask_cors import CORS
# from werkzeug.utils import secure_filename
# from PIL import Image, UnidentifiedImageError
# import numpy as np
# import cv2
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms

# # --------- CONFIG ----------
# BASE = os.getcwd()
# WEIGHTS = os.path.join(BASE, "best_model.pth")
# TRAITS_CSV = os.path.join(BASE, "breed_traits.csv")  # (not used here but kept)
# SILH_DIR = os.path.join(BASE, "silhouettes")
# STATIC_DIR = os.path.join(BASE, "static")
# TEMPLATES_DIR = os.path.join(BASE, "templates")
# RESULTS_DIR = os.path.join(STATIC_DIR, "results")
# CROSSLOG = os.path.join(BASE, "crossbreed_log.csv")
# FEEDBACK_LOG = os.path.join(BASE, "feedback_log.csv")

# os.makedirs(RESULTS_DIR, exist_ok=True)
# os.makedirs(SILH_DIR, exist_ok=True)
# os.makedirs(STATIC_DIR, exist_ok=True)
# os.makedirs(TEMPLATES_DIR, exist_ok=True)

# # Replace with your breed list (order must match model training class order)
# BREEDS = ["Gir", "H_F", "Murrah", "jersey", "nili_ravi"]

# ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# # Device
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
# print("Using device:", DEVICE)

# # -------- transforms ----------
# transform_input = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # --------- utilities ----------
# def now_utc_iso() -> str:
#     return datetime.datetime.utcnow().isoformat()

# def append_csv(path, rowdict):
#     exists = os.path.isfile(path)
#     with open(path, "a", newline="", encoding="utf8") as f:
#         w = csv.DictWriter(f, fieldnames=list(rowdict.keys()))
#         if not exists:
#             w.writeheader()
#         w.writerow(rowdict)

# def ensure_uint8_single_channel(arr: np.ndarray) -> np.ndarray:
#     """
#     Ensure a [H,W] single-channel uint8 array for cv2.applyColorMap.
#     """
#     if arr is None:
#         arr = np.zeros((2, 2), dtype=np.float32)
#     arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

#     # If it's not 2D, squeeze/convert
#     if arr.ndim == 3:
#         # If HxWxC, reduce with mean over channels
#         arr = arr.mean(axis=2)
#     elif arr.ndim == 1:
#         # 1D -> make tiny 2D
#         arr = np.expand_dims(arr, 0)
#     elif arr.ndim == 0:
#         arr = np.array([[float(arr)]], dtype=np.float32)

#     # Normalize 0..1 safely
#     mn, mx = float(arr.min()), float(arr.max())
#     if mx > mn:
#         arr = (arr - mn) / (mx - mn)
#     else:
#         arr = np.zeros_like(arr, dtype=np.float32)

#     # Scale to 0..255 and convert to uint8
#     arr_u8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
#     return arr_u8

# # --------- model load ----------
# def load_model(weights_path, num_classes, device):
#     model = models.efficientnet_b0(weights=None)
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(in_features, num_classes)

#     data = torch.load(weights_path, map_location=device)
#     # Heuristic: if it's a dict of tensors (state_dict)
#     if isinstance(data, dict) and all(hasattr(v, "dtype") for v in data.values() if hasattr(v, "dtype")):
#         try:
#             model.load_state_dict(data, strict=False)
#             print("Loaded weights as state_dict.")
#         except Exception:
#             new_sd = {k.replace("module.", ""): v for k, v in data.items()}
#             model.load_state_dict(new_sd, strict=False)
#             print("Loaded weights as state_dict (module. prefix removed).")
#     else:
#         # Try full model
#         if isinstance(data, nn.Module):
#             model = data
#             print("Loaded full model object from checkpoint.")
#             # Ensure classifier matches num_classes
#             try:
#                 if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
#                     if getattr(model.classifier[1], "out_features", None) != num_classes:
#                         in_f = model.classifier[1].in_features
#                         model.classifier[1] = nn.Linear(in_f, num_classes)
#                         print("Replaced classifier to match num_classes.")
#             except Exception:
#                 pass
#         else:
#             raise RuntimeError("Unknown weights format; please provide state_dict or full model.")

#     model.to(device)
#     model.eval()
#     return model

# if not os.path.isfile(WEIGHTS):
#     raise FileNotFoundError(f"Model weights not found at: {WEIGHTS}")
# model = load_model(WEIGHTS, len(BREEDS), DEVICE)

# # -------- Grad-CAM helper ----------
# def compute_gradcam(model, pil_img, class_idx) -> np.ndarray:
#     """
#     Returns a 2D float heatmap (H,W) in [0,1]. If anything fails, returns zeros.
#     """
#     try:
#         model.zero_grad()
#         x = transform_input(pil_img).unsqueeze(0).to(DEVICE)

#         features = None
#         grads = None

#         def forward_hook(module, input, output):
#             nonlocal features
#             features = output

#         def backward_hook(module, grad_in, grad_out):
#             nonlocal grads
#             grads = grad_out[0]

#         target_module = model.features[-1]
#         fh = target_module.register_forward_hook(forward_hook)
#         bh = target_module.register_full_backward_hook(backward_hook)

#         out = model(x)
#         out_tensor = out[0] if isinstance(out, tuple) else out
#         score = out_tensor[0, class_idx]
#         # Important: enable grads for backward
#         model.zero_grad(set_to_none=True)
#         score.backward(retain_graph=False)

#         fh.remove(); bh.remove()

#         if features is None or grads is None:
#             return np.zeros((pil_img.height, pil_img.width), dtype=np.float32)

#         fmap = features.detach().cpu().numpy()[0]     # [C,H,W]
#         g = grads.detach().cpu().numpy()              # [C,H,W]
#         if fmap.ndim != 3 or g.ndim != 3:
#             return np.zeros((pil_img.height, pil_img.width), dtype=np.float32)

#         weights = g.mean(axis=(1, 2))                 # [C]
#         cam = np.maximum(np.sum(weights[:, None, None] * fmap, axis=0), 0)  # [H,W]
#         mx = cam.max()
#         if mx > 1e-8:
#             cam = cam / mx
#         else:
#             cam = np.zeros_like(cam, dtype=np.float32)
#         return cam.astype(np.float32)
#     except Exception as _:
#         # Don’t crash prediction if CAM fails
#         return np.zeros((pil_img.height, pil_img.width), dtype=np.float32)

# def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.45) -> np.ndarray:
#     """
#     Returns a BGR overlay image (uint8). Never throws on bad input.
#     """
#     # Convert PIL → OpenCV BGR
#     try:
#         orig = np.array(pil_img)
#     except Exception:
#         # fallback
#         orig = np.zeros((224, 224, 3), dtype=np.uint8)
#     if orig.ndim == 2:  # grayscale -> 3ch
#         orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
#     elif orig.shape[2] == 4:  # RGBA -> RGB
#         orig = cv2.cvtColor(orig, cv2.COLOR_RGBA2RGB)
#     # RGB -> BGR
#     orig = orig[:, :, ::-1].copy()

#     # Ensure heatmap single-channel uint8
#     heat_u8 = ensure_uint8_single_channel(heatmap)
#     # Resize to match original size
#     heat_u8 = cv2.resize(heat_u8, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
#     # Apply colormap (expects CV_8UC1 or CV_8UC3)
#     hm_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
#     # Blend
#     overlay = cv2.addWeighted(orig, 1.0 - float(alpha), hm_color, float(alpha), 0)
#     return overlay

# # ---------- Flask app ----------
# app = Flask(__name__, static_folder="static", template_folder="templates")
# CORS(app, resources={r"/*": {"origins": "*"}})

# @app.route("/health")
# def health():
#     return jsonify({"status": "ok", "time": now_utc_iso()})

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         if 'image' not in request.files:
#             return jsonify({"error": "no image uploaded"}), 400

#         f = request.files['image']
#         original_filename = secure_filename(f.filename or "upload.jpg")
#         ext = os.path.splitext(original_filename)[1].lower()
#         if ext not in ALLOWED_EXTS:
#             return jsonify({"error": f"unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}"}), 400

#         # Parse numeric traits
#         def to_float(v, default=0.0):
#             try:
#                 return float(v)
#             except Exception:
#                 return default

#         milk = to_float(request.form.get('milk', 0.0))
#         lact = to_float(request.form.get('lact', 0.0))
#         weight = to_float(request.form.get('weight', 0.0))
#         disease = to_float(request.form.get('disease', 0.0))

#         # Load image safely
#         file_bytes = f.read()
#         if not file_bytes:
#             return jsonify({"error": "empty image file"}), 400
#         try:
#             pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
#         except UnidentifiedImageError:
#             return jsonify({"error": "could not decode image"}), 400

#         # Save original upload (optional, helpful for QA)
#         stamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
#         base = f"{os.path.splitext(original_filename)[0]}_{stamp}"
#         orig_path = os.path.join(RESULTS_DIR, f"{base}_orig.jpg")
#         pil.save(orig_path, format="JPEG", quality=92)

#         # Model forward (probs)
#         x = transform_input(pil).unsqueeze(0).to(DEVICE)
#         with torch.no_grad():
#             out = model(x)
#             probs = torch.softmax(out, dim=1).cpu().numpy()[0]

#         # Top-k
#         idxs = probs.argsort()[::-1][:3]
#         top3 = [(BREEDS[i], float(probs[i])) for i in idxs]
#         top1_idx = int(idxs[0])
#         pred_breed = BREEDS[top1_idx]
#         top1_prob = float(probs[top1_idx])

#         # Grad-CAM and overlay (graceful on failure)
#         cam = compute_gradcam(model, pil, top1_idx)  # 2D float
#         heat_overlay_bgr = overlay_heatmap_on_image(pil, cam)
#         heat_fn = f"{base}_heat.jpg"
#         heat_path = os.path.join(RESULTS_DIR, heat_fn)
#         # Ensure write
#         ok = cv2.imwrite(heat_path, heat_overlay_bgr)
#         heat_url = f"/static/results/{heat_fn}" if ok else None

#         # Crossbreed heuristic
#         margin = float(probs[idxs[0]] - probs[idxs[1]]) if len(idxs) > 1 else 1.0
#         cross_flag = (top1_prob < 0.7) or (margin < 0.15)

#         append_csv(CROSSLOG, {
#             "timestamp": now_utc_iso(),
#             "image_filename": original_filename,
#             "saved_original": os.path.basename(orig_path),
#             "predicted_cnn": pred_breed,
#             "pred_conf_cnn": top1_prob,
#             "top3": json.dumps(top3),
#             "milk": milk, "lact": lact, "weight": weight, "disease": disease,
#             "cross_flag": bool(cross_flag)
#         })

#         resp = {
#             "top3": top3,
#             "final_pred": pred_breed,
#             "cnn_conf": top1_prob,
#             "crossbreed_flag": bool(cross_flag),
#             "heatmap_url": heat_url
#         }
#         return jsonify(resp)

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

# @app.route("/feedback", methods=["POST"])
# def feedback():
#     try:
#         data = request.get_json(force=True) or {}
#         entry = {
#             "timestamp": now_utc_iso(),
#             "image": data.get("image"),
#             "predicted_final": data.get("predicted_final"),
#             "action": data.get("action"),
#             "scores": json.dumps(data.get("scores")) if data.get("scores") else ""
#         }
#         append_csv(FEEDBACK_LOG, entry)
#         return jsonify({"status": "saved"})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

# @app.route("/static/results/<path:filename>")
# def serve_result(filename):
#     return send_from_directory(RESULTS_DIR, filename)

# @app.route("/")
# def home():
#     # If you don't have templates, return a simple message.
#     try:
#         return render_template("index.html")
#     except Exception:
#         return "Bovine Breed Predictor API is running.", 200

# @app.route("/demo")
# def demo():
#     try:
#         return render_template("demo.html")
#     except Exception:
#         return "Demo page not set up.", 200

# @app.route("/documentation")
# def documentation():
#     try:
#         return render_template("documentation.html")
#     except Exception:
#         return "Documentation page not set up.", 200

# @app.route("/support")
# def support():
#     try:
#         return render_template("support.html")
#     except Exception:
#         return "Support page not set up.", 200

# if __name__ == "__main__":
#     print("App root:", BASE)
#     print("Results dir:", RESULTS_DIR)
#     print("Silhouettes dir:", SILH_DIR)
#     port = int(os.environ.get("PORT", 5000))  # Render dyno port
#     app.run(host="0.0.0.0", port=port, debug=False)

#!/usr/bin/env python3

import os, io, csv, json, datetime, traceback
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# --------- CONFIG ----------
BASE = os.getcwd()
WEIGHTS = os.path.join(BASE, "best_model.pth")
RESULTS_DIR = os.path.join(BASE, "static", "results")
CROSSLOG = os.path.join(BASE, "crossbreed_log.csv")
FEEDBACK_LOG = os.path.join(BASE, "feedback_log.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

BREEDS = ["Gir", "H_F", "Murrah", "jersey", "nili_ravi"]
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

transform_input = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def append_csv(path, rowdict):
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rowdict.keys()))
        if not exists:
            w.writeheader()
        w.writerow(rowdict)

def load_model(weights_path, num_classes, device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    state = torch.load(weights_path, map_location=device)
    if "module." in list(state.keys())[0]:
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

if not os.path.isfile(WEIGHTS):
    raise FileNotFoundError("best_model.pth missing.")

model = load_model(WEIGHTS, len(BREEDS), DEVICE)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error":"No image file received"}), 400

        f = request.files["image"]
        fname = secure_filename(f.filename)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_EXTS:
            return jsonify({"error": "Unsupported image format"}), 400

        milk = float(request.form.get("milk", 0))
        lact = float(request.form.get("lact", 0))
        weight = float(request.form.get("weight", 0))
        disease = float(request.form.get("disease", 0))

        img = Image.open(io.BytesIO(f.read())).convert("RGB")

        x = transform_input(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

        idxs = probs.argsort()[::-1][:3]
        top3 = [(BREEDS[i], float(probs[i])) for i in idxs]

        pred = BREEDS[idxs[0]]
        conf = float(probs[idxs[0]])

        cross_flag = conf < 0.7 or (probs[idxs[0]] - probs[idxs[1]]) < 0.15

        append_csv(CROSSLOG, {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "image_filename": fname,
            "predicted_cnn": pred,
            "pred_conf_cnn": conf,
            "top3": json.dumps(top3),
            "milk": milk, "lact": lact, "weight": weight, "disease": disease,
            "cross_flag": bool(cross_flag)
        })

        return jsonify({
            "top3": top3,
            "final_pred": pred,
            "cnn_conf": conf,
            "crossbreed_flag": bool(cross_flag),
            "heatmap_url": None  # removed to avoid memory crash
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json(force=True)
        append_csv(FEEDBACK_LOG, {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "image": data.get("image"),
            "predicted_final": data.get("predicted_final"),
            "action": data.get("action"),
            "scores": json.dumps(data.get("scores"))
        })
        return jsonify({"status":"saved"})
    except:
        return jsonify({"error":"Feedback save failed"}), 500

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

