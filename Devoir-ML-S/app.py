from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent  # les modèles sont cherchés à côté de app.py


def resolve(path: Path, check=None) -> Path:
    """Trouve un fichier .joblib, même renommé par le navigateur ('uniques (1).joblib').
    `check` permet de ne garder que le fichier au bon contenu."""
    others = sorted(p for p in path.parent.glob(f"{path.stem}*{path.suffix}") if p != path)
    for p in ([path] if path.exists() else []) + others:
        try:
            if check is None or check(joblib.load(p)):
                return p
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(f"{path.name} introuvable (ou invalide) dans {path.parent}")


# ─────────────────────────────────────────────────────────────
# Modèle voiture : chargement + prétraitement (identique au notebook)
# ─────────────────────────────────────────────────────────────
CAR_DIR = BASE_DIR / "models" / "voiture"
CAR_FILES = [str(CAR_DIR / n) for n in ("rf_model.joblib", "scaler.joblib", "encoders.joblib")]
CAR_CATS = ["Fuel_Type", "Seller_Type", "Transmission"]  # même ordre que encoders.joblib
CAR_ORDER = ["Kms_Driven", "Present_Price", "Fuel_Type", "Seller_Type", "Transmission", "Age"]


def load_car():
    return {
        "model": joblib.load(resolve(CAR_DIR / "rf_model.joblib")),
        "scaler": joblib.load(resolve(CAR_DIR / "scaler.joblib")),
        "encoders": joblib.load(resolve(CAR_DIR / "encoders.joblib")),
    }


def predict_car(bundle, X: pd.DataFrame):
    X = X[CAR_ORDER].copy()
    for col, enc in zip(CAR_CATS, bundle["encoders"]):
        values = X[col].astype(str)
        unknown = sorted(set(values) - set(enc.classes_))
        if unknown:
            raise ValueError(
                f"Valeur inconnue pour {col} : {', '.join(unknown)}. "
                f"Valeurs acceptées : {', '.join(enc.classes_)}"
            )
        X[col] = enc.transform(values)
    return bundle["model"].predict(bundle["scaler"].transform(X.values))


# ─────────────────────────────────────────────────────────────
# Modèle téléphone : chargement + prétraitement (identique au notebook)
# ─────────────────────────────────────────────────────────────
PHONE_DIR = BASE_DIR / "models" / "telephone"
PHONE_FILES = [str(PHONE_DIR / n) for n in ("rf_model.joblib", "scale.joblib", "uniques.joblib")]
PHONE_ORDER = ["prix", "adresse", "marque", "dim_ecr", "ram", "stockage"]
PHONE_CATS = ["adresse", "marque"]  # ordre de uniques.joblib : adresse, marque, etat


def load_phone():
    # NB : on encode avec uniques.joblib et non encoder.joblib. Dans le notebook,
    # le même LabelEncoder a été ré-ajusté à chaque colonne : encoder.joblib ne
    # connaît donc que les classes de la dernière colonne (etat).
    uniques = joblib.load(resolve(PHONE_DIR / "uniques.joblib", lambda u: len(u) == 3 and len(u[2]) == 4))
    classes = dict(zip(PHONE_CATS, uniques))
    return {
        "model": joblib.load(resolve(PHONE_DIR / "rf_model.joblib")),
        "scaler": joblib.load(resolve(PHONE_DIR / "scale.joblib")),
        "classes": classes,
        "options": classes,  # alimente les listes déroulantes du formulaire
    }


def predict_phone(bundle, X: pd.DataFrame):
    X = X[PHONE_ORDER].copy()
    for col in PHONE_CATS:
        classes = bundle["classes"][col]
        values = X[col].astype(str).str.strip()
        unknown = sorted(set(values) - set(classes))
        if unknown:
            hint = f" Valeurs acceptées : {', '.join(classes)}" if len(classes) <= 12 else ""
            raise ValueError(f"Valeur inconnue pour {col} : {', '.join(unknown[:5])}.{hint}")
        X[col] = values.map({c: i for i, c in enumerate(classes)})
    scaled = bundle["scaler"].transform(X.values.astype(float))
    model = bundle["model"]
    proba = model.predict_proba(scaled)
    return model.classes_[proba.argmax(axis=1)], proba.max(axis=1)


# ─────────────────────────────────────────────────────────────
# Modèle taxi : chargement + prétraitement (identique au notebook)
# ─────────────────────────────────────────────────────────────
TAXI_DIR = BASE_DIR / "models" / "taxi"
TAXI_FILES = [str(TAXI_DIR / n) for n in ("rf_model.joblib", "scaler.joblib", "encoders.joblib")]
TAXI_CATS = ["Time_of_Day", "Day_of_Week", "Traffic_Conditions", "Weather"]  # ordre de encoders.joblib
TAXI_ORDER = [
    "Trip_Distance_km", "Time_of_Day", "Day_of_Week", "Passenger_Count", "Traffic_Conditions",
    "Weather", "Base_Fare", "Per_Km_Rate", "Per_Minute_Rate", "Trip_Duration_Minutes",
]


def load_taxi():
    encoders = joblib.load(resolve(TAXI_DIR / "encoders.joblib"))
    return {
        "model": joblib.load(resolve(TAXI_DIR / "rf_model.joblib")),
        "scaler": joblib.load(resolve(TAXI_DIR / "scaler.joblib")),
        "encoders": encoders,
        "options": {c: list(e.classes_) for c, e in zip(TAXI_CATS, encoders)},
    }


def predict_taxi(bundle, X: pd.DataFrame):
    X = X[TAXI_ORDER].copy()
    for col, enc in zip(TAXI_CATS, bundle["encoders"]):
        values = X[col].astype(str).str.strip()
        unknown = sorted(set(values) - set(enc.classes_))
        if unknown:
            raise ValueError(
                f"Valeur inconnue pour {col} : {', '.join(unknown)}. "
                f"Valeurs acceptées : {', '.join(enc.classes_)}"
            )
        X[col] = enc.transform(values)
    return bundle["model"].predict(bundle["scaler"].transform(X.values.astype(float)))


# ─────────────────────────────────────────────────────────────
# Modèle état de la voiture (Venant / D'occasion) : prétraitement du notebook
# ─────────────────────────────────────────────────────────────
CARCLS_DIR = BASE_DIR / "models" / "voiture_etat"
CARCLS_FILES = [str(CARCLS_DIR / n) for n in ("rf_model.joblib", "scaler.joblib", "uniques.joblib")]
CARCLS_ORDER = ["Marque", "Année", "Boite_Vitesse", "Adresse", "Prix"]
CARCLS_CATS = ["Marque", "Boite_Vitesse", "Adresse"]  # ordre de uniques.joblib (puis Venant_Occasion)


def load_carcls():
    # NB : encodage via uniques.joblib (encoder.joblib ne connaît que la dernière colonne,
    # le même LabelEncoder ayant été ré-ajusté à chaque colonne dans le notebook).
    uniques = joblib.load(resolve(CARCLS_DIR / "uniques.joblib", lambda u: len(u) == 4 and len(u[3]) == 2))
    classes = dict(zip(CARCLS_CATS, uniques))
    return {
        "model": joblib.load(resolve(CARCLS_DIR / "rf_model.joblib")),
        "scaler": joblib.load(resolve(CARCLS_DIR / "scaler.joblib")),
        "classes": classes,
        "options": classes,
    }


def predict_carcls(bundle, X: pd.DataFrame):
    X = X[CARCLS_ORDER].copy()
    # Même nettoyage que le notebook : "Almadies,Dakar" -> "Almadies"
    X["Adresse"] = X["Adresse"].astype(str).str.split(",").str[0].str.strip()
    for col in CARCLS_CATS:
        classes = bundle["classes"][col]
        values = X[col].astype(str).str.strip()
        unknown = sorted(set(values) - set(classes))
        if unknown:
            hint = f" Valeurs acceptées : {', '.join(classes)}" if len(classes) <= 12 else ""
            raise ValueError(f"Valeur inconnue pour {col} : {', '.join(unknown[:5])}.{hint}")
        X[col] = values.map({c: i for i, c in enumerate(classes)})
    scaled = bundle["scaler"].transform(X.values.astype(float))
    model = bundle["model"]
    proba = model.predict_proba(scaled)
    return model.classes_[proba.argmax(axis=1)], proba.max(axis=1)


# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION DES MODÈLES  ← seule section à adapter
# ─────────────────────────────────────────────────────────────
# Pour chaque modèle :
#   path     : fichier du modèle (joblib / pickle). Il doit accepter un
#              DataFrame dont les colonnes sont celles de "features"
#              (idéalement un Pipeline qui contient le prétraitement).
#   task     : "classification" ou "regression"
#   files    : (optionnel) fichiers requis si le modèle en utilise plusieurs
#   loader   : (optionnel) fonction qui charge le modèle (défaut : joblib.load(path))
#   predict  : (optionnel) fonction (modèle, DataFrame) -> prédictions, pour un prétraitement sur mesure
#   labels   : (optionnel) {classe: libellé lisible}
#   unit     : (optionnel) unité affichée pour une régression
#   features : liste des champs du formulaire
#       {"name": "age", "type": "number", "min": 0, "max": 100, "default": 30, "step": 1}
#       {"name": "sexe", "type": "select", "options": ["F", "H"]}
#     "label" (facultatif) : libellé affiché ; sans "options", la liste vient du loader (bundle["options"])

MODELS = {
    "voiture": {
        "title": "Prix d'une voiture",
        "description": "Estime le prix de vente d'une voiture d'occasion à partir de son âge, de son kilométrage et de ses caractéristiques.",
        "path": str(CAR_DIR / "rf_model.joblib"),
        "files": CAR_FILES,
        "loader": load_car,
        "predict": predict_car,
        "task": "regression",
        "unit": "$",
        "features": [
            {"name": "Age", "label": "Âge (ans)", "type": "number", "min": 0, "max": 30, "default": 8, "step": 1},
            {"name": "Kms_Driven", "label": "Kilométrage (km)", "type": "number", "min": 0, "max": 600000, "default": 70000, "step": 1000},
            {"name": "Present_Price", "label": "Prix du véhicule neuf", "type": "number", "min": 0.1, "max": 150.0, "default": 15.0, "step": 0.1},
            {"name": "Fuel_Type", "label": "Carburant", "type": "select", "options": ["CNG", "Diesel", "Petrol"]},
            {"name": "Seller_Type", "label": "Type de vendeur", "type": "select", "options": ["Dealer", "Individual"]},
            {"name": "Transmission", "label": "Boîte de vitesses", "type": "select", "options": ["Automatic", "Manual"]},
        ],
    },
    "taxi": {
        "title": "Prix d'une course de taxi",
        "description": "Estime le prix d'une course à partir de la distance, de la durée, du trafic, de la météo et des tarifs appliqués.",
        "path": str(TAXI_DIR / "rf_model.joblib"),
        "files": TAXI_FILES,
        "loader": load_taxi,
        "predict": predict_taxi,
        "task": "regression",
        "unit": "$",
        "features": [
            {"name": "Trip_Distance_km", "label": "Distance (km)", "type": "number", "min": 1.0, "max": 150.0, "default": 25.0, "step": 0.5},
            {"name": "Trip_Duration_Minutes", "label": "Durée (minutes)", "type": "number", "min": 5.0, "max": 120.0, "default": 60.0, "step": 1.0},
            {"name": "Passenger_Count", "label": "Passagers", "type": "number", "min": 1, "max": 4, "default": 2, "step": 1},
            {"name": "Time_of_Day", "label": "Moment de la journée", "type": "select"},
            {"name": "Day_of_Week", "label": "Type de jour", "type": "select"},
            {"name": "Traffic_Conditions", "label": "Trafic", "type": "select"},
            {"name": "Weather", "label": "Météo", "type": "select"},
            {"name": "Base_Fare", "label": "Tarif de prise en charge", "type": "number", "min": 2.0, "max": 5.0, "default": 3.5, "step": 0.05},
            {"name": "Per_Km_Rate", "label": "Tarif par km", "type": "number", "min": 0.5, "max": 2.0, "default": 1.2, "step": 0.01},
            {"name": "Per_Minute_Rate", "label": "Tarif par minute", "type": "number", "min": 0.1, "max": 0.5, "default": 0.3, "step": 0.01},
        ],
    },
    "telephone": {
        "title": "État d'un téléphone",
        "description": "Détermine si un téléphone est d'occasion, neuf, réconditionné ou venant, à partir de son prix et de ses caractéristiques.",
        "path": str(PHONE_DIR / "rf_model.joblib"),
        "files": PHONE_FILES,
        "loader": load_phone,
        "predict": predict_phone,
        "task": "classification",
        "labels": {0: "D'occasion", 1: "Neuf", 2: "Réconditionné", 3: "Venant"},
        "features": [
            {"name": "prix", "label": "Prix (FCFA)", "type": "number", "min": 0, "max": 4000000, "default": 250000, "step": 5000},
            {"name": "marque", "label": "Marque", "type": "select"},
            {"name": "adresse", "label": "Adresse", "type": "select"},
            {"name": "dim_ecr", "label": "Taille de l'écran (pouces)", "type": "number", "min": 3.0, "max": 10.0, "default": 6.0, "step": 0.1},
            {"name": "ram", "label": "RAM (Go)", "type": "number", "min": 1, "max": 64, "default": 8, "step": 1},
            {"name": "stockage", "label": "Stockage (Go)", "type": "number", "min": 1, "max": 2048, "default": 128, "step": 32},
        ],
    },
    "voiture_etat": {
        "title": "État d'une voiture",
        "description": "Détermine si une voiture est « Venant » ou d'occasion, à partir de sa marque, de son année, de sa boîte, de son adresse et de son prix.",
        "path": str(CARCLS_DIR / "rf_model.joblib"),
        "files": CARCLS_FILES,
        "loader": load_carcls,
        "predict": predict_carcls,
        "task": "classification",
        "labels": {0: "D'occasion", 1: "Venant"},
        "features": [
            {"name": "Marque", "label": "Marque", "type": "select"},
            {"name": "Année", "label": "Année", "type": "number", "min": 1990, "max": 2026, "default": 2015, "step": 1},
            {"name": "Boite_Vitesse", "label": "Boîte de vitesses", "type": "select"},
            {"name": "Adresse", "label": "Quartier", "type": "select"},
            {"name": "Prix", "label": "Prix (FCFA)", "type": "number", "min": 0, "max": 100000000, "default": 8000000, "step": 100000},
        ],
    },
}

# ─────────────────────────────────────────────────────────────
# 2. STYLE
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Prédictions", page_icon="◐", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #F2E9DA;
  --surface: #FBF6ED;
  --ink: #2E2118;
  --muted: #8C7763;
  --line: #E2D3BC;
  --accent: #A4622D;
  --accent-dark: #7A4421;
  --espresso: #2B1E16;
  --cream: #F6EDE0;
  --caramel: #E0B07A;
}

html, body, [class*="css"], .stApp {
  font-family: 'Manrope', sans-serif;
  color: var(--ink);
}
.stApp {
  background: radial-gradient(1100px 420px at 85% -8%, #E8D6B8 0%, rgba(232,214,184,0) 65%), var(--bg);
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
/* Sur ordinateur : on retire le bouton qui replie la barre des modèles, pour qu'elle reste toujours visible */
@media (min-width: 992px) {
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
  section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button,
  section[data-testid="stSidebar"] button[kind="header"] { display: none; }
}
.block-container { max-width: 900px; padding-top: 3.2rem; padding-bottom: 4rem; }

/* Barre latérale */
section[data-testid="stSidebar"] { background: var(--espresso); border-right: none; }
section[data-testid="stSidebar"] .block-container { padding-top: 2.6rem; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * { color: var(--cream); }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: .3rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  padding: .6rem .8rem; border-radius: 12px; width: 100%; transition: background .15s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(246,237,224,.08); }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: rgba(224,176,122,.18);
  box-shadow: inset 3px 0 0 var(--caramel);
}
.brand { font-family: 'Fraunces', Georgia, serif !important; font-weight: 600; font-size: 1.5rem; letter-spacing: -0.01em; margin-bottom: 1.6rem; }
.brand span { color: var(--caramel); font-style: italic; }
.side-note { color: #B9A58C !important; font-size: .78rem; margin-top: 2.2rem; line-height: 1.5; }

/* En-tête de page */
.tag {
  display: inline-block; padding: .28rem .75rem; border-radius: 99px; background: #EADBC2; color: var(--accent-dark);
  font-size: .7rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 1rem;
}
h1.page-title {
  font-family: 'Fraunces', Georgia, serif !important; font-size: 2.8rem; font-weight: 600;
  letter-spacing: -0.02em; line-height: 1.1; margin: 0 0 .6rem 0; color: var(--ink);
}
p.page-sub { color: var(--muted); margin: 0 0 2.2rem 0; font-size: 1.02rem; max-width: 640px; line-height: 1.6; }

/* Onglets */
.stTabs [data-baseweb="tab-list"] { gap: 1.8rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: .7rem 0; font-weight: 600; color: var(--muted); background: transparent; }
.stTabs [data-baseweb="tab"] p { color: inherit !important; }
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 3px; border-radius: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Champs */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--ink) !important; font-weight: 600 !important; font-size: .82rem !important; }
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
  border-radius: 12px !important; background: var(--surface) !important; border-color: var(--line) !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within { border-color: var(--accent) !important; }
div[data-baseweb="input"] input, div[data-baseweb="select"] * { color: var(--ink) !important; }
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* Boutons */
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; font-weight: 700; padding: .7rem 1.6rem;
  border: 1px solid var(--line); background: var(--surface); color: var(--ink);
  transition: all .15s;
}
.stButton > button p, .stDownloadButton > button p { color: inherit !important; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent-dark); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: var(--espresso); border-color: var(--espresso); color: var(--cream);
  box-shadow: 0 10px 22px -14px rgba(43,30,22,.8);
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
  background: var(--accent-dark); border-color: var(--accent-dark); color: #fff;
}

/* Résultat */
.result {
  background: linear-gradient(135deg, #2B1E16 0%, #4A3123 100%); color: var(--cream);
  border-radius: 22px; padding: 1.9rem 2.1rem; margin-top: 1.8rem;
  box-shadow: 0 22px 44px -26px rgba(43,30,22,.7);
}
.result .k { color: #CDB79B; font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 700; }
.result .v { font-family: 'Fraunces', Georgia, serif; font-size: 3.1rem; font-weight: 600; line-height: 1.1; color: var(--cream); margin-top: .5rem; }
.result .c { color: var(--caramel); font-weight: 600; margin-top: 1rem; font-size: .9rem; }
.bar { height: 6px; border-radius: 99px; background: rgba(246,237,224,.16); margin-top: .55rem; overflow: hidden; }
.bar span { display: block; height: 100%; background: linear-gradient(90deg, #D9A066, #EBCB98); border-radius: 99px; }

/* Import & tableau */
[data-testid="stFileUploader"] section { border-radius: 16px; border: 1.5px dashed #CDB79B; background: var(--surface); }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 14px; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
# Pas de ligne vide : Markdown couperait le bloc <style> et afficherait le CSS en texte
st.markdown("\n".join(l for l in CSS.splitlines() if l.strip()), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 3. LOGIQUE
# ─────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    cfg = MODELS[model_key]
    loader = cfg.get("loader")
    return loader() if loader else joblib.load(cfg["path"])


def feature_names(cfg: dict) -> list[str]:
    return [f["name"] for f in cfg["features"]]


def predict(model, X: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Retourne un DataFrame avec 'prediction' (et 'confiance' si disponible)."""
    predictor = cfg.get("predict")
    conf = None
    if predictor:
        result = predictor(model, X)
        raw, conf = result if isinstance(result, tuple) else (result, None)
    else:
        raw = model.predict(X)
        if cfg["task"] == "classification" and hasattr(model, "predict_proba"):
            conf = model.predict_proba(X).max(axis=1)
    out = pd.DataFrame(index=X.index)

    if cfg["task"] == "classification":
        labels = cfg.get("labels", {})
        out["prediction"] = [labels.get(p, p) for p in raw]
        if conf is not None:
            out["confiance"] = conf
    else:
        out["prediction"] = pd.Series(raw, index=X.index).astype(float).round(3)
    return out


def render_field(f: dict, key: str, model=None):
    label = f.get("label", f["name"])
    if f["type"] == "select":
        options = f.get("options")
        if options is None and isinstance(model, dict):
            options = model.get("options", {}).get(f["name"])
        return st.selectbox(label, options or [], key=key)
    step = f.get("step", 1.0)
    return st.number_input(
        label,
        min_value=f.get("min"),
        max_value=f.get("max"),
        value=f.get("default", f.get("min", 0.0)),
        step=step,
        key=key,
    )


def single_prediction(model_key: str, cfg: dict, model):
    cols = st.columns(2)
    values = {}
    for i, f in enumerate(cfg["features"]):
        with cols[i % 2]:
            values[f["name"]] = render_field(f, f"{model_key}_{f['name']}", model)

    st.write("")
    if st.button("Prédire", type="primary", key=f"{model_key}_go"):
        X = pd.DataFrame([values])
        try:
            res = predict(model, X, cfg).iloc[0]
        except Exception as e:  # noqa: BLE001
            st.error(f"La prédiction a échoué : {e}")
            return

        value = res["prediction"]
        if cfg["task"] == "regression":
            value = f"{value:,.2f} {cfg.get('unit', '')}".strip().replace(",", " ")
        conf = ""
        if "confiance" in res:
            conf = (
                f"<div class='c'>Confiance : {res['confiance']:.1%}</div>"
                f"<div class='bar'><span style='width:{res['confiance'] * 100:.0f}%'></span></div>"
            )

        st.markdown(
            f"<div class='result'><div class='k'>Résultat</div>"
            f"<div class='v'>{value}</div>{conf}</div>",
            unsafe_allow_html=True,
        )


def multiple_prediction(model_key: str, cfg: dict, model):
    names = feature_names(cfg)

    st.markdown(
        f"Importez un fichier CSV avec les colonnes : **{', '.join(names)}**"
    )
    template = pd.DataFrame(columns=names).to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger le modèle de fichier", template,
                       file_name=f"{model_key}_modele.csv", mime="text/csv",
                       key=f"{model_key}_tpl")

    file = st.file_uploader("Fichier CSV", type=["csv"], key=f"{model_key}_file",
                            label_visibility="collapsed")
    if file is None:
        return

    try:
        df = pd.read_csv(file)
    except Exception as e:  # noqa: BLE001
        st.error(f"Lecture impossible : {e}")
        return

    missing = [c for c in names if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes : {', '.join(missing)}")
        return

    st.caption(f"{len(df)} lignes importées")
    if st.button("Lancer les prédictions", type="primary", key=f"{model_key}_batch"):
        try:
            res = predict(model, df[names], cfg)
        except Exception as e:  # noqa: BLE001
            st.error(f"La prédiction a échoué : {e}")
            return

        result = pd.concat([df, res], axis=1)
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button("Télécharger les résultats",
                           result.to_csv(index=False).encode("utf-8"),
                           file_name=f"{model_key}_resultats.csv", mime="text/csv",
                           key=f"{model_key}_dl")


# ─────────────────────────────────────────────────────────────
# 4. INTERFACE
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<div class='brand'>Prédic<span>tions</span></div>", unsafe_allow_html=True)
    model_key = st.radio(
        "Modèle",
        list(MODELS.keys()),
        format_func=lambda k: MODELS[k]["title"],
        label_visibility="collapsed",
    )
    st.markdown(
        "<div class='side-note'>4 modèles · prédiction simple<br>ou sur un fichier CSV</div>",
        unsafe_allow_html=True,
    )

cfg = MODELS[model_key]
tag = "Régression" if cfg["task"] == "regression" else "Classification"
st.markdown(f"<span class='tag'>{tag}</span>", unsafe_allow_html=True)

st.markdown(f"<h1 class='page-title'>{cfg['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='page-sub'>{cfg['description']}</p>", unsafe_allow_html=True)

missing_files = [
    Path(p) for p in cfg.get("files", [cfg["path"]])
    if not any(Path(p).parent.glob(f"{Path(p).stem}*{Path(p).suffix}"))
]
if missing_files:
    st.error("Fichiers introuvables : " + ", ".join(f"`{p.relative_to(BASE_DIR)}`" for p in missing_files))
    st.stop()

try:
    model = load_model(model_key)
except Exception as e:  # noqa: BLE001
    st.error(f"Chargement du modèle impossible : {type(e).__name__} — {e}")
    st.stop()

tab_single, tab_multi = st.tabs(["Prédiction simple", "Prédiction multiple"])
with tab_single:
    st.write("")
    single_prediction(model_key, cfg, model)
with tab_multi:
    st.write("")
    multiple_prediction(model_key, cfg, model)
