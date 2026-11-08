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

st.set_page_config(page_title="Prédictions", page_icon="◐", layout="wide")

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap">
<style>

:root {
  --bg: #F6F7F9;
  --surface: #FFFFFF;
  --ink: #161A23;
  --muted: #6B7280;
  --line: #E4E7EC;
  --accent: #3B4EF0;
  --accent-soft: #EEF0FE;
}

html, body, [class*="css"], .stApp {
  font-family: 'Manrope', sans-serif;
  color: var(--ink);
}
.stApp { background: var(--bg); }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 920px; padding-top: 3rem; padding-bottom: 4rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 2.5rem; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * { color: var(--ink); }
h1.page-title, p.page-sub, .brand, .result, label, .stTabs button, .stButton button,
.stDownloadButton button, input, section[data-testid="stSidebar"] p { font-family: 'Manrope', sans-serif !important; }
.brand { font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; margin-bottom: 1.5rem; }
.brand span { color: var(--accent); }

/* Titres */
h1.page-title { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 .25rem 0; }
p.page-sub { color: var(--muted); margin: 0 0 2rem 0; font-size: 1rem; }

/* Onglets */
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: .6rem 0; font-weight: 600; color: var(--muted); background: transparent; }
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Champs */
label, .stNumberInput label, .stSelectbox label { font-weight: 600 !important; font-size: .85rem !important; }
div[data-baseweb="input"], div[data-baseweb="select"] > div {
  border-radius: 10px !important; background: var(--surface) !important; border-color: var(--line) !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
  border-color: var(--accent) !important;
}

/* Boutons */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 700; padding: .65rem 1.4rem;
  border: 1px solid var(--line); background: var(--surface); color: var(--ink);
  transition: border-color .15s, background .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; }
.stButton > button[kind="primary"]:hover { background: #2B3CD4; border-color: #2B3CD4; color: #fff; }

/* Résultat */
.result {
  background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.6rem 1.8rem; margin-top: 1.5rem;
}
.result .k { color: var(--muted); font-size: .85rem; font-weight: 600; }
.result .v { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15; }
.result .c { color: var(--accent); font-weight: 700; margin-top: .3rem; }

/* Upload & tableau */
[data-testid="stFileUploader"] section {
  border-radius: 14px; border: 1.5px dashed var(--line); background: var(--surface);
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

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
            conf = f"<div class='c'>Confiance : {res['confiance']:.1%}</div>"

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

cfg = MODELS[model_key]

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
