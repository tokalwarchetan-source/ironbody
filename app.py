# IronBody - Flask app with Gemini AI
# Gemini 3.6 Flash + Gemini 3.5 Flash-Lite fallback
# Put app.py and index.html in the same folder.
#
# First-time setup:
#   python -m pip install flask flask-cors python-dotenv requests
#
# Environment:
#   GEMINI_API_KEY=YOUR_KEY
#   Optional: GEMINI_MODEL=gemini-3.6-flash

import os
import time
import threading
import webbrowser
import requests
import re
import json

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# ---------------------------------------------------------------------
# APP / CONFIG
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_FILE, override=True)

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)

PORT = int(os.environ.get("PORT", "3001"))

GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GOOGLE_GEMINI_API_KEY")
    or ""
).strip().strip('"').strip("'")

# Main model. Keep this configurable from Render/.env.
MODEL = os.environ.get("GEMINI_MODEL", "").strip() or "gemini-3.6-flash"

# Cheaper/faster fallback used only when the main model returns a quota
# (HTTP 429) error. This helps the demo continue when one model's free
# tier request limit is reached.
FALLBACK_MODEL = os.environ.get(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.5-flash-lite"
).strip()

# Do not keep an old/deprecated model name from an older deployment.
if MODEL in {"gemini-2.5-flash", "gemini-2.5-flash-001"}:
    MODEL = "gemini-3.6-flash"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

print(f"[CONFIG] .env found: {os.path.isfile(ENV_FILE)}")
print(f"[CONFIG] Gemini API key loaded: {bool(GEMINI_API_KEY)}")
print(f"[CONFIG] Primary model: {MODEL}")
print(f"[CONFIG] Fallback model: {FALLBACK_MODEL}")


# ---------------------------------------------------------------------
# SIMPLE LOCAL RATE LIMIT
# Google free-tier limits can be lower than a website's button rate.
# Keep this slightly below the common 20 requests/minute limit shown
# by the user's current error, so accidental repeated clicks are reduced.
# ---------------------------------------------------------------------

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 18
_hits = {}


def rate_limited(ip):
    now = time.time()
    entry = _hits.get(ip, {"count": 0, "start": now})

    if now - entry["start"] > WINDOW_SECONDS:
        entry = {"count": 0, "start": now}

    entry["count"] += 1
    _hits[ip] = entry
    return entry["count"] > MAX_REQUESTS_PER_WINDOW


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def gemini_url(model):
    return f"{GEMINI_BASE_URL}/{model}:generateContent"


def extract_error_message(data, status_code):
    """Turn Google's long technical error into a short user-facing message."""
    err = data.get("error", {}) if isinstance(data, dict) else {}
    raw = err.get("message", "") if isinstance(err, dict) else str(err)

    raw = str(raw or "").strip()

    # Extract Google's retry delay when present.
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", raw, re.I)
    if match:
        seconds = max(1, int(round(float(match.group(1)))))
        return (
            f"Gemini is temporarily rate-limited. "
            f"Please wait about {seconds} seconds and try again."
        )

    if status_code == 429 or "quota" in raw.lower() or "rate limit" in raw.lower():
        return (
            "Gemini has reached the current API quota. "
            "Please wait a little and try again."
        )

    if status_code in (401, 403):
        return (
            "Gemini API access was rejected. "
            "Check the GEMINI_API_KEY in Render Environment Variables."
        )

    if status_code == 400:
        return "Gemini rejected the request. Please try again."

    if raw:
        # Keep the UI clean instead of dumping Google's full error payload.
        return raw[:500]

    return "Gemini returned an error."


def parse_retry_seconds(data):
    """Read a retry delay from Google's error message if available."""
    err = data.get("error", {}) if isinstance(data, dict) else {}
    raw = err.get("message", "") if isinstance(err, dict) else str(err)
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", str(raw), re.I)
    if match:
        return max(1, int(round(float(match.group(1)))))
    return None


def post_gemini(model, payload, timeout=180):
    """
    Make one Gemini REST request.

    Returns:
        (response_json, status_code, None)
    or:
        (None, status_code, user_facing_error)
    """
    try:
        upstream = requests.post(
            gemini_url(model),
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        return None, 504, "Gemini took too long to respond."
    except requests.RequestException:
        return None, 502, "Could not connect to Gemini. Please try again."

    try:
        data = upstream.json()
    except ValueError:
        return None, upstream.status_code, "Gemini returned an invalid response."

    if not upstream.ok:
        return data, upstream.status_code, extract_error_message(
            data, upstream.status_code
        )

    return data, upstream.status_code, None


def call_gemini_with_fallback(payload, timeout=180):
    """
    Try the primary model first.
    If Google returns HTTP 429/quota exceeded, try the fallback model.
    """
    data, status, error = post_gemini(MODEL, payload, timeout)

    if data is not None and status < 400:
        return data, status, None, MODEL

    # Only switch models for quota/rate-limit errors.
    if status == 429 and FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
        print(
            f"[GEMINI] {MODEL} quota/rate limit reached; "
            f"trying fallback {FALLBACK_MODEL}"
        )

        fallback_data, fallback_status, fallback_error = post_gemini(
            FALLBACK_MODEL, payload, timeout
        )

        if fallback_data is not None and fallback_status < 400:
            return fallback_data, fallback_status, None, FALLBACK_MODEL

        # If fallback also fails, return the cleanest quota message.
        if fallback_status == 429:
            return (
                None,
                fallback_status,
                extract_error_message(
                    fallback_data or {},
                    fallback_status,
                ),
                FALLBACK_MODEL,
            )

    return None, status, error or "Gemini returned an error.", MODEL


def get_response_text(data):
    """Safely extract normal text from Gemini's candidate response."""
    if not isinstance(data, dict):
        return ""

    candidates = data.get("candidates") or []
    if not candidates:
        return ""

    candidate = candidates[0] or {}
    content = candidate.get("content") or {}
    parts = content.get("parts") or []

    texts = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])

    return "\n".join(texts).strip()


def parse_json_object(text):
    """
    Robust JSON parser for structured food analysis.

    Gemini is instructed to return JSON, but this also handles:
      - ```json ... ```
      - accidental text before/after the JSON object
      - whitespace/newlines
    """
    if not isinstance(text, str):
        raise ValueError("No text returned.")

    cleaned = text.strip()

    # Remove markdown fences.
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    # First attempt: exact JSON.
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Second attempt: decode the first JSON object embedded in text.
    start = cleaned.find("{")
    if start >= 0:
        decoder = json.JSONDecoder()
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    # Third attempt: take the outermost {...} block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON.")


def to_int(value, lo=None, hi=None, default=0):
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = default

    if lo is not None:
        number = max(lo, number)
    if hi is not None:
        number = min(hi, number)

    return number


# ---------------------------------------------------------------------
# PAGES / HEALTH
# ---------------------------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "ai": "gemini",
            "model": MODEL,
            "fallback_model": FALLBACK_MODEL,
            "configured": bool(GEMINI_API_KEY),
        }
    )


# ---------------------------------------------------------------------
# FOOD PHOTO ANALYSIS
# ---------------------------------------------------------------------

@app.route("/api/analyze-food", methods=["POST"])
def analyze_food():
    """
    Analyze a food photo using Gemini vision.

    The old version could show:
      "The food model returned an unreadable result."

    This version:
      - uses structured JSON output
      - removes deprecated sampling parameters
      - has a more robust JSON parser
      - uses Gemini fallback on quota errors
      - returns clean errors to the frontend
    """
    if rate_limited(request.remote_addr):
        return (
            jsonify(
                {
                    "error": (
                        "Too many requests. Please wait about one minute "
                        "before trying again."
                    )
                }
            ),
            429,
        )

    body = request.get_json(silent=True) or {}

    image = body.get("image", "")
    image_mime_type = str(
        body.get("mimeType", "image/jpeg")
    ).strip().lower()
    expected_food = str(body.get("expectedFood", "")).strip()

    # Prefer the MIME type encoded in the data URL when available.
    # This prevents PNG/WebP/JPEG bytes from being mislabeled as another format.
    if isinstance(image, str) and image.startswith("data:"):
        header, _, payload_data = image.partition(",")
        match = re.match(r"data:(image/[a-zA-Z0-9.+-]+);base64$", header, re.I)
        if match:
            image_mime_type = match.group(1).lower()
        image = payload_data

    if not image or not isinstance(image, str):
        return jsonify({"error": "Please provide a food image."}), 400

    if image_mime_type not in {
        "image/jpeg", "image/jpg", "image/png", "image/webp", "image/heic", "image/heif"
    }:
        image_mime_type = "image/jpeg"

    if len(image) > 12_000_000:
        return (
            jsonify(
                {
                    "error": (
                        "Image is too large. Please use an image "
                        "under about 9 MB."
                    )
                }
            ),
            413,
        )

    expected_instruction = (
        f'The user thinks the food is "{expected_food}". '
        "Check whether the photo matches that food."
        if expected_food
        else "Identify the most likely food or dish visible in the image."
    )

    prompt = f"""
You are IronBody's food-photo recognition model.
Analyze ONLY the provided food image.

{expected_instruction}

Estimate nutrition for the visible serving.

Return ONLY one valid JSON object.
Do not use Markdown.
Do not put any explanation before or after the JSON.

Use exactly these keys:
food_name: short string
confidence: integer 0-100 for visual recognition confidence
is_food: boolean
match: boolean or null
reason: one short sentence
serving_size: short string
calories: integer kcal
protein_g: integer grams
carbs_g: integer grams
fat_g: integer grams
fiber_g: integer grams
nutrition_confidence: integer 0-100

If the image is unclear, lower confidence.
If is_food is false, return zeroes for nutrition fields.
Do not claim confidence is scientific accuracy.
"""

    if not GEMINI_API_KEY:
        return (
            jsonify(
                {
                    "error": (
                        "Gemini API key is missing. Put GEMINI_API_KEY=YOUR_KEY "
                        "in the .env file beside app.py, then restart Flask."
                    )
                }
            ),
            503,
        )

    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You are IronBody's food-photo recognition model. "
                        "Return exactly one valid JSON object and nothing else."
                    )
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": image_mime_type,
                            "data": image,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 800,
            "responseMimeType": "application/json",
        },
    }

    print(
        f"[FOOD] request: mime={image_mime_type}, base64_chars={len(image)}, "
        f"expected={expected_food or '(none)'}"
    )

    data, status, error, used_model = call_gemini_with_fallback(
        payload, timeout=180
    )

    if data is None:
        print(f"[FOOD] Gemini error: status={status}, error={error}")
        return jsonify({"error": error}), status

    text = get_response_text(data)

    if not text:
        return (
            jsonify(
                {
                    "error": (
                        "Gemini did not return a readable food analysis. "
                        "Please try a clearer food photo."
                    )
                }
            ),
            502,
        )

    try:
        result = parse_json_object(text)
    except ValueError:
        print("[FOOD] Unreadable model output:", repr(text[:1500]))
        return (
            jsonify(
                {
                    "error": (
                        "The food analysis format was invalid. "
                        "Please try the photo again."
                    )
                }
            ),
            502,
        )

    confidence = to_int(
        result.get("confidence", 0),
        lo=0,
        hi=100,
    )

    nutrition_confidence = to_int(
        result.get(
            "nutrition_confidence",
            max(0, confidence - 15),
        ),
        lo=0,
        hi=100,
    )

    return jsonify(
        {
            "food_name": str(
                result.get("food_name", "Unknown food")
            ),
            "confidence": confidence,
            "is_food": bool(result.get("is_food", True)),
            "match": result.get("match"),
            "reason": str(
                result.get(
                    "reason",
                    "Visual evidence was limited.",
                )
            ),
            "serving_size": str(
                result.get(
                    "serving_size",
                    "1 serving (estimated)",
                )
            ),
            "nutrition": {
                "calories": to_int(
                    result.get("calories", 0),
                    lo=0,
                ),
                "protein_g": to_int(
                    result.get("protein_g", 0),
                    lo=0,
                ),
                "carbs_g": to_int(
                    result.get("carbs_g", 0),
                    lo=0,
                ),
                "fat_g": to_int(
                    result.get("fat_g", 0),
                    lo=0,
                ),
                "fiber_g": to_int(
                    result.get("fiber_g", 0),
                    lo=0,
                ),
                "confidence": nutrition_confidence,
            },
            "model": used_model,
            "note": (
                "Recognition and nutrition values are visual estimates "
                "from the model, not validated measurements. Actual "
                "ingredients and serving size can change the values."
            ),
        }
    )


# ---------------------------------------------------------------------
# AI DIET COACH CHAT
# ---------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    if rate_limited(request.remote_addr):
        return (
            jsonify(
                {
                    "error": (
                        "Too many requests. Please wait about one minute "
                        "before trying again."
                    )
                }
            ),
            429,
        )

    body = request.get_json(silent=True) or {}

    system_prompt = body.get("systemPrompt")
    messages = body.get("messages")
    max_tokens = body.get("maxTokens") or 1800

    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages array is required"}), 400

    if not GEMINI_API_KEY:
        return (
            jsonify(
                {
                    "error": (
                        "Gemini API key is missing. Put GEMINI_API_KEY=YOUR_KEY "
                        "in the .env file beside app.py, then restart Flask."
                    )
                }
            ),
            503,
        )

    # Keep the response useful but prevent a frontend mistake from asking
    # for an enormous output on every request.
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = 1800

    max_tokens = max(300, min(max_tokens, 4000))

    gemini_contents = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")

        if role == "assistant":
            role = "model"
        elif role not in ["user", "model"]:
            role = "user"

        content = message.get("content", "")

        if isinstance(content, list):
            parts = []

            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):
                    parts.append(
                        {
                            "text": str(
                                item.get("text", "")
                            )
                        }
                    )
                else:
                    parts.append(
                        {
                            "text": str(item)
                        }
                    )
        else:
            parts = [{"text": str(content)}]

        if parts:
            gemini_contents.append(
                {
                    "role": role,
                    "parts": parts,
                }
            )

    if not gemini_contents:
        return jsonify({"error": "No usable messages were supplied."}), 400

    # Gemini 3.x does not allow a prefilled model turn as the final turn.
    # If the browser accidentally sends one, remove trailing model turns.
    while (
        gemini_contents
        and gemini_contents[-1].get("role") == "model"
    ):
        gemini_contents.pop()

    if not gemini_contents:
        return jsonify({"error": "Please send a new user message."}), 400

    payload = {
        "contents": gemini_contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        },
    }

    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [
                {
                    "text": str(system_prompt)
                }
            ]
        }

    print(
        f"[FOOD] request: mime={image_mime_type}, base64_chars={len(image)}, "
        f"expected={expected_food or '(none)'}"
    )

    data, status, error, used_model = call_gemini_with_fallback(
        payload, timeout=180
    )

    if data is None:
        print(f"[FOOD] Gemini error: status={status}, error={error}")
        return jsonify({"error": error}), status

    text = get_response_text(data)

    if not text:
        return (
            jsonify(
                {
                    "error": (
                        "Gemini returned an empty answer. "
                        "Please try again."
                    )
                }
            ),
            502,
        )

    return jsonify(
        {
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
            "model": used_model,
        }
    )


# ---------------------------------------------------------------------
# LOCAL START
# ---------------------------------------------------------------------

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
    print(f" Primary : {MODEL}")
    print(f" Fallback: {FALLBACK_MODEL}")
    print(" Keep this window open while using IronBody.")
    print("==============================================")
    print()

    threading.Thread(
        target=open_browser,
        daemon=True,
    ).start()

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False,
    )
