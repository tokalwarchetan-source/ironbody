# IronBody - One-click Flask app with local Ollama AI
# Put app.py and index.html in the same folder.
#
# First-time setup:
#   python -m pip install flask flask-cors python-dotenv requests
#
# Ollama must be installed and running locally.
# Default model: gemma3:4b

import os
import time
import threading
import webbrowser
import requests

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)

PORT = int(os.environ.get("PORT", "3001"))
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

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
    return jsonify({"status": "ok", "ai": "ollama", "model": MODEL})

@app.route("/api/analyze-food", methods=["POST"])
def analyze_food():
    """Analyze a food photo with Gemma 3 vision."""
    if rate_limited(request.remote_addr):
        return jsonify({"error": "Too many requests. Please slow down."}), 429

    body = request.get_json(silent=True) or {}
    image = body.get("image", "")
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

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [image]}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 400},
    }

    try:
        upstream = requests.post(OLLAMA_URL, json=payload, timeout=180)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Ollama is not running. Open Ollama and try again."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Food analysis took too long to respond."}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Ollama connection error: {e}"}), 502

    try:
        data = upstream.json()
    except ValueError:
        return jsonify({"error": "Ollama returned an invalid response."}), 502
    if not upstream.ok:
        return jsonify({"error": data.get("error", "Ollama returned an error.")}), upstream.status_code

    text = data.get("message", {}).get("content", "")
    if not text:
        return jsonify({"error": "Ollama returned an empty food analysis."}), 502
    try:
        import json
        result = json.loads(text)
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

    ollama_messages = []
    if system_prompt:
        ollama_messages.append({"role": "system", "content": str(system_prompt)})

    for message in messages:
        role = message.get("role", "user")
        if role not in ["system", "user", "assistant"]:
            role = "user"
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            content = "\n".join(parts)
        ollama_messages.append({"role": role, "content": str(content)})

    payload = {
        "model": MODEL,
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        upstream = requests.post(OLLAMA_URL, json=payload, timeout=180)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Ollama is not running. Open Ollama and try again."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Ollama took too long to respond."}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Ollama connection error: {e}"}), 502

    try:
        data = upstream.json()
    except ValueError:
        return jsonify({"error": "Ollama returned an invalid response."}), 502

    if not upstream.ok:
        return jsonify({"error": data.get("error", "Ollama returned an error.")}), upstream.status_code

    text = data.get("message", {}).get("content", "")
    if not text:
        return jsonify({"error": "Ollama returned an empty response."}), 502

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
    print(" AI      : Ollama")
    print(f" Model   : {MODEL}")
    print(" Keep this window open while using IronBody.")
    print("==============================================")
    print()
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)
