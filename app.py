from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import recommend_by_barcode
import os

app = Flask(__name__)

# ✅ ENABLE CORS (IMPORTANT)
CORS(app)  # allows all origins (safe for learning/demo)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)

    if not data or "barcode" not in data:
        return jsonify({
            "status": "error",
            "message": "barcode is required"
        }), 400

    barcode = str(data.get("barcode", "")).strip()
    top_n = int(data.get("top_n", 4))

    product, recommendations = recommend_by_barcode(barcode, top_n)

    if product is None:
        return jsonify({
            "status": "error",
            "message": "Product not found",
            "barcode": barcode
        }), 404

    return jsonify({
        "status": "success",
        "input_barcode": barcode,
        "product": product,
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
