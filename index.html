# IronBody - Flask app with Gemini AI for local + cloud deployment
# Put app.py and index.html in the same folder.
#
# First-time setup:
#   python -m pip install flask flask-cors python-dotenv requests
#
# Gemini API key is read from GEMINI_API_KEY.
# Default model: gemini-3.6-flash

import os
import time
import threading
import webbrowser
import requests
import re
from dotenv import load_dotenv

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from the same folder as app.py.
# override=True also fixes cases where Windows has an empty/stale
# GEMINI_API_KEY environment variable that would otherwise hide .env.
ENV_FILE = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_FILE, override=True)

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)

PORT = int(os.environ.get("PORT", "3001"))

# Accept the project's normal name plus common Gemini/Google aliases.
# The actual key is never printed or returned to the browser.
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GOOGLE_GEMINI_API_KEY")
    or ""
).strip().strip('"').strip("'")

MODEL = os.environ.get("GEMINI_MODEL", "").strip() or "gemini-3.6-flash"

# Automatically migrate the old model name if it is still present in
# Render/environment variables.
if MODEL in {"gemini-2.5-flash", "gemini-2.5-flash-001"}:
    MODEL = "gemini-3.6-flash"

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

print(f"[CONFIG] .env found: {os.path.isfile(ENV_FILE)}")
print(f"[CONFIG] Gemini API key loaded: {bool(GEMINI_API_KEY)}")
print(f"[CONFIG] Gemini model: {MODEL}")

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 30
_hits = {}

def rate_limited(ip):
    now = time.time()
    entry = _hits.get(ip, {"count": 0, "start": now})
    if now - entry["start"] > WINDOW_SECONDS:
        entry = {"count": 0, "start": now}
    entry["count"] += 1
    _hits[ip] = entry
    return entry["count"] > MAX_REQUESTS_PER_WINDOW

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "ai": "gemini", "model": MODEL, "configured": bool(GEMINI_API_KEY)})

@app.route("/api/analyze-food", methods=["POST"])
def analyze_food():
    """Analyze a food photo with Gemma 3 vision."""
    if rate_limited(request.remote_addr):
        return jsonify({"error": "Too many requests. Please slow down."}), 429

    body = request.get_json(silent=True) or {}
    image = body.get("image", "")
    image_mime_type = str(body.get("mimeType", "image/jpeg")).strip().lower()
    if not image_mime_type.startswith("image/"):
        image_mime_type = "image/jpeg"
    expected_food = str(body.get("expectedFood", "")).strip()

    if not image or not isinstance(image, str):
        return jsonify({"error": "Please provide a food image."}), 400
    if image.startswith("data:"):
        image = image.split(",", 1)[1]
    if len(image) > 12_000_000:
        return jsonify({"error": "Image is too large. Please use an image under about 9 MB."}), 413

    expected_instruction = (
        f' The user thinks the food is "{expected_food}". Check whether the photo matches that food.'
        if expected_food else
        " Identify the most likely food or dish visible in the image."
    )
    prompt = f"""You are IronBody's food-photo recognition model. Analyze ONLY the provided food image.
{expected_instruction}
Also estimate the nutrition for a typical single serving of the identified dish.
Return ONLY valid JSON with these keys:
food_name: short name of the most likely food/dish
confidence: integer from 0 to 100 representing your visual recognition confidence
is_food: true or false
match: true or false or null (null when no expected food was supplied)
reason: one short sentence explaining the visual evidence
serving_size: short description of the estimated serving shown, e.g. "1 plate (~250g)"
calories: estimated integer kcal for the serving shown
protein_g: estimated integer grams of protein for the serving shown
carbs_g: estimated integer grams of carbohydrate for the serving shown
fat_g: estimated integer grams of fat for the serving shown
fiber_g: estimated integer grams of fiber for the serving shown
nutrition_confidence: integer from 0 to 100 for how confident you are in the nutrition estimate specifically (this is usually lower than visual recognition confidence, since portion size is hard to judge from a photo)
Do not claim that any confidence score is scientific accuracy. If the image is unclear, say so and lower confidence. If is_food is false, still return zeroes for the nutrition fields."""

    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key is missing. Put GEMINI_API_KEY=YOUR_KEY in the .env file beside app.py, then restart Flask."}), 503

    payload = {
        "systemInstruction": {
            "parts": [{"text": "You are IronBody's food-photo recognition model. Return only valid JSON."}]
        },
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": image_mime_type, "data": image}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 600,
            "responseMimeType": "application/json"
        }
    }

    try:
        upstream = requests.post(
            GEMINI_URL,
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=180
        )
    except requests.exceptions.Timeout:
        return jsonify({"error": "Food analysis took too long to respond."}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Gemini connection error: {e}"}), 502

    try:
        data = upstream.json()
    except ValueError:
        return jsonify({"error": "Gemini returned an invalid response."}), 502
    if not upstream.ok:
        err = data.get("error", {})
        message = err.get("message", "Gemini returned an error.") if isinstance(err, dict) else str(err)
        return jsonify({"error": message}), upstream.status_code

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return jsonify({"error": "Gemini returned an empty food analysis."}), 502
    try:
        import json
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"\\s*```$", "", cleaned)
        result = json.loads(cleaned)
    except Exception:
        return jsonify({"error": "The food model returned an unreadable result."}), 502

    def to_int(value, lo=None, hi=None, default=0):
        try:
            n = int(round(float(value)))
        except (TypeError, ValueError):
            n = default
        if lo is not None:
            n = max(lo, n)
        if hi is not None:
            n = min(hi, n)
        return n

    confidence = to_int(result.get("confidence", 0), lo=0, hi=100)
    nutrition_confidence = to_int(result.get("nutrition_confidence", max(0, confidence - 15)), lo=0, hi=100)

    return jsonify({
        "food_name": str(result.get("food_name", "Unknown food")),
        "confidence": confidence,
        "is_food": bool(result.get("is_food", True)),
        "match": result.get("match"),
        "reason": str(result.get("reason", "Visual evidence was limited.")),
        "serving_size": str(result.get("serving_size", "1 serving (estimated)")),
        "nutrition": {
            "calories": to_int(result.get("calories", 0), lo=0),
            "protein_g": to_int(result.get("protein_g", 0), lo=0),
            "carbs_g": to_int(result.get("carbs_g", 0), lo=0),
            "fat_g": to_int(result.get("fat_g", 0), lo=0),
            "fiber_g": to_int(result.get("fiber_g", 0), lo=0),
            "confidence": nutrition_confidence,
        },
        "note": "Recognition and nutrition values are the model's visual estimates, not a validated real-world accuracy percentage — actual serving size and ingredients can shift these numbers."
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    if rate_limited(request.remote_addr):
        return jsonify({"error": "Too many requests. Please slow down."}), 429

    body = request.get_json(silent=True) or {}
    system_prompt = body.get("systemPrompt")
    messages = body.get("messages")
    max_tokens = body.get("maxTokens") or 1000
    temperature = body.get("temperature", 0.7)

    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages array is required"}), 400

    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key is missing. Put GEMINI_API_KEY=YOUR_KEY in the .env file beside app.py, then restart Flask."}), 503

    gemini_contents = []
    for message in messages:
        role = message.get("role", "user")
        if role == "assistant":
            role = "model"
        elif role not in ["user", "model"]:
            role = "user"

        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append({"text": str(item.get("text", ""))})
                else:
                    parts.append({"text": str(item)})
        else:
            parts = [{"text": str(content)}]

        gemini_contents.append({"role": role, "parts": parts})

    payload = {
        "contents": gemini_contents,
        "generationConfig": {
            "maxOutputTokens": int(max_tokens)
        }
    }

    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": str(system_prompt)}]
        }

    try:
        upstream = requests.post(
            GEMINI_URL,
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=180
        )
    except requests.exceptions.Timeout:
        return jsonify({"error": "Gemini took too long to respond."}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Gemini connection error: {e}"}), 502

    try:
        data = upstream.json()
    except ValueError:
        return jsonify({"error": "Gemini returned an invalid response."}), 502

    if not upstream.ok:
        err = data.get("error", {})
        message = err.get("message", "Gemini returned an error.") if isinstance(err, dict) else str(err)
        return jsonify({"error": message}), upstream.status_code

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return jsonify({"error": "Gemini returned an empty response."}), 502

    return jsonify({"content": [{"type": "text", "text": text}]})

def open_browser():
    time.sleep(1.2)
    webbrowser.open(f"http://127.0.0.1:{PORT}/")

if __name__ == "__main__":
    print()
    print("==============================================")
    print("        IRONBODY - GYM PERFORMANCE HUB")
    print("==============================================")
    print(f" Website : http://127.0.0.1:{PORT}/")
    print(f" Health  : http://127.0.0.1:{PORT}/health")
    print(" AI      : Gemini")
    print(f" Model   : {MODEL}")
    print(" Keep this window open while using IronBody.")
    print("==============================================")
    print()
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
