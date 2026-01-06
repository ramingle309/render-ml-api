import pandas as pd
import pickle
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "ready.csv")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

# ===============================
# BARCODE NORMALIZATION
# ===============================
def normalize_barcode(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    x = x.replace(".0", "")
    x = re.sub(r"\D", "", x)
    return x

# ===============================
# LOAD DATA (IMPORTANT FIX HERE)
# ===============================
USE_COLS = [
    "code",
    "product_name",
    "product_name_clean",
    "ingredients_text_clean",
    "nutriscore_filled_n",
    "allergens",
    "allergens_en"
]

df = pd.read_csv(
    DATASET_PATH,
    usecols=USE_COLS,
    engine="python"
)

df["code"] = df["code"].apply(normalize_barcode)

df["combined_text"] = (
    df["product_name_clean"].fillna("") + " " +
    df["ingredients_text_clean"].fillna("")
)

df["nutriscore_filled_n"] = pd.to_numeric(
    df["nutriscore_filled_n"], errors="coerce"
)

# ===============================
# ✅ UNIFIED ALLERGEN COLUMN (KEY FIX)
# ===============================
df["final_allergens"] = (
    df["allergens_en"]
    .fillna(df["allergens"])
    .replace("", pd.NA)
)

# ===============================
# LOAD TF-IDF VECTORIZER
# ===============================
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ===============================
# HELPERS
# ===============================
def nutri_grade_from_score(score):
    if pd.isna(score):
        return None
    score = float(score)
    if score <= -1:
        return "A"
    elif score <= 2:
        return "B"
    elif score <= 10:
        return "C"
    elif score <= 18:
        return "D"
    else:
        return "E"

def safe_float(x):
    try:
        return None if pd.isna(x) else float(x)
    except:
        return None

def safe_str(x):
    return None if pd.isna(x) else str(x)

# ===============================
# MAIN RECOMMENDER
# ===============================
def recommend_by_barcode(barcode, top_n=4):
    barcode = normalize_barcode(barcode)

    if not barcode:
        return None, []

    matches = df.index[df["code"] == barcode].tolist()
    if not matches:
        return None, []

    idx = matches[0]

    base_text = df.loc[idx, "combined_text"]
    base_score = df.loc[idx, "nutriscore_filled_n"]

    base_vec = vectorizer.transform([base_text])

    candidates = df[
        (df.index != idx) &
        (df["nutriscore_filled_n"] < base_score)
    ].copy()

    candidates = candidates.head(3000)

    candidate_vecs = vectorizer.transform(candidates["combined_text"])
    similarity = cosine_similarity(base_vec, candidate_vecs).flatten()
    candidates["similarity"] = similarity

    recs = (
        candidates
        .sort_values("similarity", ascending=False)
        .drop_duplicates(subset="code")
        .head(top_n)
    )

    # ===============================
    # OUTPUT
    # ===============================
    product = {
        "name": df.loc[idx, "product_name"],
        "code": df.loc[idx, "code"],
        "nutriscore": safe_float(base_score),
        "nutrigrade": nutri_grade_from_score(base_score),
        "allergens": safe_str(df.loc[idx, "final_allergens"])
    }

    recommendations = [
        {
            "name": row["product_name"],
            "code": row["code"],
            "nutriscore": safe_float(row["nutriscore_filled_n"]),
            "nutrigrade": nutri_grade_from_score(row["nutriscore_filled_n"]),
            "allergens": safe_str(row["final_allergens"]),
            "similarity": round(float(row["similarity"]), 3)
        }
        for _, row in recs.iterrows()
    ]

    return product, recommendations
