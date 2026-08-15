# IronBody - Flask app with Gemini AI
# Local + Render deployment
#
# Put app.py and index.html in the same folder.
#
# Install:
#   pip install flask flask-cors python-dotenv requests
#
# Environment variable:
#   GEMINI_API_KEY=your_key
#
# Optional:
#   GEMINI_MODEL=gemini-3.6-flash
#   PORT=3001

import os
import time
import threading
import webbrowser
import json
import requests

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_FILE = os.path.join(BASE_DIR, ".env")

# Load .env and allow it to override stale Windows environment variables.
load_dotenv(ENV_FILE, override=True)

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

CORS(app)

PORT = int(os.environ.get("PORT", "3001"))


# ============================================================
# GEMINI API CONFIGURATION
# ============================================================

# Accept several common environment variable names.
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GOOGLE_GEMINI_API_KEY")
    or ""
).strip().strip('"').strip("'")

# Current stable Gemini model.
MODEL = (
    os.environ.get("GEMINI_MODEL", "").strip()
    or "gemini-3.6-flash"
)

# Automatically migrate old model names.
OLD_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-001",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-09-25",
}

if MODEL in OLD_MODELS:
    MODEL = "gemini-3.6-flash"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{MODEL}:generateContent"
)


print(f"[CONFIG] .env found: {os.path.isfile(ENV_FILE)}")
print(f"[CONFIG] Gemini API key loaded: {bool(GEMINI_API_KEY)}")
print(f"[CONFIG] Gemini model: {MODEL}")


# ============================================================
# RATE LIMITING
# ============================================================

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 30

_hits = {}


def rate_limited(ip):
    now = time.time()

    entry = _hits.get(
        ip,
        {
            "count": 0,
            "start": now
        }
    )

    if now - entry["start"] > WINDOW_SECONDS:
        entry = {
            "count": 0,
            "start": now
        }

    entry["count"] += 1
    _hits[ip] = entry

    return entry["count"] > MAX_REQUESTS_PER_WINDOW


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "ai": "gemini",
        "model": MODEL,
        "configured": bool(GEMINI_API_KEY)
    })


# ============================================================
# FOOD IMAGE ANALYZER
# ============================================================

@app.route("/api/analyze-food", methods=["POST"])
def analyze_food():

    if rate_limited(request.remote_addr):
        return jsonify({
            "error": "Too many requests. Please slow down."
        }), 429

    body = request.get_json(silent=True) or {}

    image = body.get("image", "")
    expected_food = str(
        body.get("expectedFood", "")
    ).strip()

    if not image or not isinstance(image, str):
        return jsonify({
            "error": "Please provide a food image."
        }), 400

    # Remove data URL prefix if present.
    if image.startswith("data:"):
        try:
            image = image.split(",", 1)[1]
        except Exception:
            return jsonify({
                "error": "Invalid image data."
            }), 400

    # Prevent extremely large requests.
    if len(image) > 12_000_000:
        return jsonify({
            "error": "Image is too large. Please use an image under about 9 MB."
        }), 413

    if not GEMINI_API_KEY:
        return jsonify({
            "error": (
                "Gemini API key is missing. "
                "Set GEMINI_API_KEY in Render Environment Variables."
            )
        }), 503

    if expected_food:
        expected_instruction = (
            f'The user thinks the food is "{expected_food}". '
            f'Check whether the image matches that food.'
        )
    else:
        expected_instruction = (
            "Identify the most likely food or dish visible in the image."
        )

    prompt = f"""
You are IronBody's food-photo recognition model.

Analyze ONLY the provided food image.

{expected_instruction}

Estimate nutrition for the visible serving.

Return ONLY valid JSON with these keys:

food_name:
Short name of the most likely food/dish.

confidence:
Integer from 0 to 100 representing visual recognition confidence.

is_food:
true or false.

match:
true, false, or null.

reason:
One short sentence explaining the visual evidence.

serving_size:
Short description such as "1 plate (~250g)".

calories:
Estimated integer kcal.

protein_g:
Estimated grams of protein.

carbs_g:
Estimated grams of carbohydrates.

fat_g:
Estimated grams of fat.

fiber_g:
Estimated grams of fiber.

nutrition_confidence:
Integer from 0 to 100 representing confidence in the nutrition estimate.

Do not claim that confidence is scientific accuracy.

If the image is unclear, lower the confidence.

If is_food is false, return zeroes for nutrition values.
"""

    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You are IronBody's food-photo recognition model. "
                        "Return ONLY valid JSON."
                    )
                }
            ]
        },

        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image
                        }
                    }
                ]
            }
        ],

        "generationConfig": {
            "maxOutputTokens": 500,
            "responseMimeType": "application/json"
        }
    }

    try:
        upstream = requests.post(
            GEMINI_URL,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=180
        )

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Food analysis took too long to respond."
        }), 504

    except requests.RequestException as e:
        return jsonify({
            "error": f"Gemini connection error: {e}"
        }), 502

    try:
        data = upstream.json()

    except ValueError:
        return jsonify({
            "error": "Gemini returned an invalid response."
        }), 502

    if not upstream.ok:

        err = data.get("error", {})

        if isinstance(err, dict):
            message = err.get(
                "message",
                "Gemini returned an error."
            )
        else:
            message = str(err)

        return jsonify({
            "error": message
        }), upstream.status_code

    try:
        parts = (
            data["candidates"][0]
            ["content"]
            ["parts"]
        )

        text = "".join(
            str(part.get("text", ""))
            for part in parts
            if isinstance(part, dict)
        )

    except (KeyError, IndexError, TypeError):

        return jsonify({
            "error": "Gemini returned an empty food analysis."
        }), 502

    try:
        result = json.loads(text)

    except Exception:

        return jsonify({
            "error": "The food model returned an unreadable result."
        }), 502

    def to_int(
        value,
        lo=None,
        hi=None,
        default=0
    ):

        try:
            number = int(round(float(value)))

        except (TypeError, ValueError):
            number = default

        if lo is not None:
            number = max(lo, number)

        if hi is not None:
            number = min(hi, number)

        return number

    confidence = to_int(
        result.get("confidence", 0),
        lo=0,
        hi=100
    )

    nutrition_confidence = to_int(
        result.get(
            "nutrition_confidence",
            max(0, confidence - 15)
        ),
        lo=0,
        hi=100
    )

    return jsonify({

        "food_name": str(
            result.get(
                "food_name",
                "Unknown food"
            )
        ),

        "confidence": confidence,

        "is_food": bool(
            result.get(
                "is_food",
                True
            )
        ),

        "match": result.get("match"),

        "reason": str(
            result.get(
                "reason",
                "Visual evidence was limited."
            )
        ),

        "serving_size": str(
            result.get(
                "serving_size",
                "1 serving (estimated)"
            )
        ),

        "nutrition": {

            "calories": to_int(
                result.get("calories", 0),
                lo=0
            ),

            "protein_g": to_int(
                result.get("protein_g", 0),
                lo=0
            ),

            "carbs_g": to_int(
                result.get("carbs_g", 0),
                lo=0
            ),

            "fat_g": to_int(
                result.get("fat_g", 0),
                lo=0
            ),

            "fiber_g": to_int(
                result.get("fiber_g", 0),
                lo=0
            ),

            "confidence": nutrition_confidence
        },

        "note": (
            "Recognition and nutrition values are model estimates. "
            "Actual serving size and ingredients can change these values."
        )
    })


# ============================================================
# AI DIET COACH CHAT
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():

    # -----------------------------
    # Rate limit
    # -----------------------------

    if rate_limited(request.remote_addr):
        return jsonify({
            "error": "Too many requests. Please slow down."
        }), 429


    # -----------------------------
    # Read request
    # -----------------------------

    body = request.get_json(silent=True) or {}

    system_prompt = body.get("systemPrompt")

    messages = body.get("messages")

    # IMPORTANT:
    # Old version used 1000.
    # New version uses 2000 by default.
    max_tokens = body.get("maxTokens") or 2000

    # Keep a safe upper limit.
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = 2000

    max_tokens = max(
        256,
        min(max_tokens, 4000)
    )


    # -----------------------------
    # Validate messages
    # -----------------------------

    if not isinstance(messages, list) or not messages:

        return jsonify({
            "error": "messages array is required"
        }), 400


    # -----------------------------
    # Check API key
    # -----------------------------

    if not GEMINI_API_KEY:

        return jsonify({
            "error": (
                "Gemini API key is missing. "
                "Set GEMINI_API_KEY in Render Environment Variables."
            )
        }), 503


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    completion_instruction = """
You are IronBody AI Diet Coach.

Always provide a COMPLETE answer.

Never stop in the middle of a sentence.
Never stop in the middle of a bullet point.
Never stop in the middle of a meal plan.
Never leave an unfinished section.

Give practical, easy-to-understand nutrition advice.

When the user asks for a meal plan:
- Give complete meals.
- Include approximate protein where useful.
- Include calories when relevant.
- Consider the user's stated goal.
- Prefer Indian foods when appropriate.
- Keep recommendations practical.

When the user asks for pre-workout or post-workout meals:
- Clearly separate PRE-WORKOUT and POST-WORKOUT.
- Explain timing briefly.
- Include protein and carbohydrate sources.

When the user asks about protein:
- Explain the target clearly.
- Show how the remaining protein can be completed.

When the user asks for a personalized plan:
- Use the information provided by the user.
- Do not invent missing personal details.

Keep responses structured with headings and bullet points.

Most importantly:
FINISH THE ENTIRE RESPONSE.
"""

    if system_prompt:

        system_prompt = (
            str(system_prompt)
            + "\n\n"
            + completion_instruction
        )

    else:

        system_prompt = completion_instruction


    # ========================================================
    # CONVERT MESSAGES TO GEMINI FORMAT
    # ========================================================

    gemini_contents = []

    for message in messages:

        if not isinstance(message, dict):
            continue

        role = message.get(
            "role",
            "user"
        )

        # Gemini uses "model" instead of "assistant".
        if role == "assistant":
            role = "model"

        elif role not in ["user", "model"]:
            role = "user"

        content = message.get(
            "content",
            ""
        )


        # -----------------------------
        # Handle text content
        # -----------------------------

        if isinstance(content, list):

            parts = []

            for item in content:

                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):

                    parts.append({
                        "text": str(
                            item.get(
                                "text",
                                ""
                            )
                        )
                    })

                else:

                    parts.append({
                        "text": str(item)
                    })

        else:

            parts = [
                {
                    "text": str(content)
                }
            ]


        if not parts:
            continue


        gemini_contents.append({

            "role": role,

            "parts": parts

        })


    if not gemini_contents:

        return jsonify({
            "error": "No valid messages were provided."
        }), 400


    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    payload = {

        "contents": gemini_contents,

        "systemInstruction": {

            "parts": [

                {
                    "text": system_prompt
                }

            ]

        },

        "generationConfig": {

            # Gemini 3.6 uses its own reasoning behavior.
            # Do not send deprecated temperature/top-p parameters.

            "maxOutputTokens": max_tokens

        }
    }


    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        upstream = requests.post(

            GEMINI_URL,

            headers={

                "x-goog-api-key":
                    GEMINI_API_KEY,

                "Content-Type":
                    "application/json"

            },

            json=payload,

            timeout=180

        )

    except requests.exceptions.Timeout:

        return jsonify({
            "error": "Gemini took too long to respond."
        }), 504

    except requests.RequestException as e:

        return jsonify({
            "error": f"Gemini connection error: {e}"
        }), 502


    # ========================================================
    # PARSE RESPONSE
    # ========================================================

    try:

        data = upstream.json()

    except ValueError:

        return jsonify({
            "error": "Gemini returned an invalid response."
        }), 502


    # ========================================================
    # GEMINI ERROR
    # ========================================================

    if not upstream.ok:

        err = data.get(
            "error",
            {}
        )

        if isinstance(err, dict):

            message = err.get(
                "message",
                "Gemini returned an error."
            )

        else:

            message = str(err)

        return jsonify({
            "error": message
        }), upstream.status_code


    # ========================================================
    # EXTRACT COMPLETE TEXT
    # ========================================================

    try:

        candidates = data.get(
            "candidates",
            []
        )

        if not candidates:

            return jsonify({
                "error": "Gemini returned no candidates."
            }), 502


        candidate = candidates[0]

        content = candidate.get(
            "content",
            {}
        )

        parts = content.get(
            "parts",
            []
        )


        # IMPORTANT:
        # Do NOT only read parts[0].
        # Gemini can return multiple text parts.

        text_parts = []

        for part in parts:

            if isinstance(part, dict):

                text_value = part.get(
                    "text"
                )

                if text_value:
                    text_parts.append(
                        str(text_value)
                    )


        text = "\n".join(
            text_parts
        ).strip()


        if not text:

            return jsonify({
                "error": "Gemini returned an empty response."
            }), 502


    except Exception:

        return jsonify({
            "error": "Could not read Gemini response."
        }), 502


    # ========================================================
    # RETURN TO FRONTEND
    # ========================================================

    return jsonify({

        "content": [

            {
                "type": "text",

                "text": text
            }

        ]

    })


# ============================================================
# OPEN BROWSER WHEN RUN LOCALLY
# ============================================================

def open_browser():

    time.sleep(1.2)

    try:

        webbrowser.open(
            f"http://127.0.0.1:{PORT}/"
        )

    except Exception:
        pass


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("        IRONBODY - GYM PERFORMANCE HUB")
    print("==============================================")
    print(
        f" Website : http://127.0.0.1:{PORT}/"
    )
    print(
        f" Health  : http://127.0.0.1:{PORT}/health"
    )
    print(" AI      : Gemini")
    print(
        f" Model   : {MODEL}"
    )
    print(
        f" API Key : {'Loaded' if GEMINI_API_KEY else 'Missing'}"
    )
    print(
        " Chat output limit : 2000 tokens"
    )
    print(
        " Keep this window open while using IronBody."
    )
    print("==============================================")
    print()


    # Only open browser for local execution.
    if not os.environ.get("RENDER"):

        threading.Thread(
            target=open_browser,
            daemon=True
        ).start()


    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False
    )
