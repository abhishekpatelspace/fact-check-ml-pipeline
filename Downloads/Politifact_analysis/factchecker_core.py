"""
factchecker_core.py
====================
Backend logic for the FactChecker AI Platform.
Covers:
  - Data scraping (Politifact)
  - Google Fact Check API integration
  - NLP feature extraction (5 phases)
  - Model training & evaluation (K-Fold + SMOTE)
  - Demo data
  - Humour / critique generation
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urljoin
import time
import random
import io
import os
from typing import Any

# --- Imbalanced-learn ---
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- NLP & ML ---
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

# ============================================================
# CONSTANTS
# ============================================================

SCRAPED_DATA_PATH = "politifact_data.csv"
N_SPLITS = 5

# Binary classification rating groups
GOOGLE_TRUE_RATINGS  = ["True", "Mostly True", "Accurate", "Correct"]
GOOGLE_FALSE_RATINGS = [
    "False", "Mostly False", "Pants on Fire", "Pants on Fire!",
    "Fake", "Incorrect", "Baseless", "Misleading",
]

REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]

PRAGMATIC_WORDS = ["must", "should", "might", "could", "will", "?", "!"]

# ============================================================
# SPACY MODEL LOADER
# ============================================================

_NLP_MODEL = None   # module-level cache


def load_spacy_model():
    """
    Load the spaCy 'en_core_web_sm' model.
    Returns the model or raises OSError if unavailable.
    """
    global _NLP_MODEL
    if _NLP_MODEL is not None:
        return _NLP_MODEL

    try:
        _NLP_MODEL = spacy.load("en_core_web_sm")
        return _NLP_MODEL
    except OSError:
        # Attempt auto-download as a fallback
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        _NLP_MODEL = spacy.load("en_core_web_sm")
        return _NLP_MODEL


# ============================================================
# DEMO DATA
# ============================================================

def get_demo_google_claims():
    """Return a list of hard-coded fact-check claims for offline testing."""
    return [
        {"claim_text": "The earth is flat and NASA is hiding the truth from us.", "rating": "False"},
        {"claim_text": "Vaccines are completely safe and effective for 95% of the population.", "rating": "Mostly True"},
        {"claim_text": "The moon landing was filmed in a Hollywood studio in 1969.", "rating": "False"},
        {"claim_text": "Climate change is primarily caused by human activities and carbon emissions.", "rating": "True"},
        {"claim_text": "You can cure COVID-19 by drinking bleach and taking horse medication.", "rating": "False"},
        {"claim_text": "Regular exercise and balanced diet improve overall health and longevity.", "rating": "True"},
        {"claim_text": "5G towers spread coronavirus and should be taken down immediately.", "rating": "False"},
        {"claim_text": "The Great Wall of China is visible from space with the naked eye.", "rating": "Mostly False"},
        {"claim_text": "Solar energy has become more affordable and efficient in the last decade.", "rating": "True"},
        {"claim_text": "Bill Gates is using vaccines to implant microchips in people.", "rating": "Pants on Fire"},
        {"claim_text": "Drinking 8 glasses of water daily is essential for human health.", "rating": "Mostly True"},
        {"claim_text": "Sharks don't get cancer and their cartilage can cure it in humans.", "rating": "False"},
        {"claim_text": "Electric vehicles produce zero emissions and are completely eco-friendly.", "rating": "Mostly True"},
        {"claim_text": "Humans only use 10% of their brain capacity.", "rating": "False"},
        {"claim_text": "Antibiotics are effective against viral infections like flu and colds.", "rating": "False"},
        # Additional claims to reach 50+
        {"claim_text": "The Bermuda Triangle has supernatural powers that cause ships to disappear.", "rating": "False"},
        {"claim_text": "Eating carrots significantly improves your night vision.", "rating": "Mostly False"},
        {"claim_text": "Lightning never strikes the same place twice.", "rating": "False"},
        {"claim_text": "The average person swallows eight spiders per year while sleeping.", "rating": "False"},
        {"claim_text": "Goldfish have a memory span of only three seconds.", "rating": "False"},
        {"claim_text": "Cracking your knuckles causes arthritis.", "rating": "False"},
        {"claim_text": "Sugar causes hyperactivity in children.", "rating": "Mostly False"},
        {"claim_text": "Reading in dim light damages your eyesight permanently.", "rating": "False"},
        {"claim_text": "Hair and fingernails continue to grow after death.", "rating": "False"},
        {"claim_text": "Shaving makes hair grow back thicker and darker.", "rating": "False"},
        {"claim_text": "We lose most of our body heat through our heads.", "rating": "Mostly False"},
        {"claim_text": "Touching a baby bird will cause its mother to abandon it.", "rating": "False"},
        {"claim_text": "Bats are completely blind and navigate only by echolocation.", "rating": "False"},
        {"claim_text": "Bulls are enraged by the color red.", "rating": "False"},
        {"claim_text": "Dogs see the world entirely in black and white.", "rating": "False"},
        {"claim_text": "The tongue has specific taste zones for different flavors.", "rating": "False"},
        {"claim_text": "Chameleons change color primarily for camouflage.", "rating": "Mostly False"},
        {"claim_text": "Bananas grow on trees.", "rating": "False"},
        {"claim_text": "Fortune cookies originated in China.", "rating": "False"},
        {"claim_text": "Napoleon Bonaparte was unusually short for his time.", "rating": "False"},
        {"claim_text": "Albert Einstein failed mathematics in school.", "rating": "Pants on Fire"},
        {"claim_text": "The Great Pyramid was built by slaves.", "rating": "False"},
        {"claim_text": "Vikings wore horned helmets in battle.", "rating": "False"},
        {"claim_text": "Christopher Columbus discovered America.", "rating": "Mostly False"},
        {"claim_text": "The iron maiden was a medieval torture device.", "rating": "Mostly False"},
        {"claim_text": "Medieval people thought the Earth was flat.", "rating": "False"},
        {"claim_text": "Renewable energy cannot power modern economies reliably.", "rating": "False"},
        {"claim_text": "Organic food is always healthier than conventional food.", "rating": "Mostly False"},
        {"claim_text": "GMO foods are dangerous to human health.", "rating": "False"},
        {"claim_text": "Microwaving food destroys its nutritional value.", "rating": "Mostly False"},
        {"claim_text": "Eating at night causes more weight gain than eating during the day.", "rating": "Mostly False"},
        {"claim_text": "Detox diets remove toxins from your body.", "rating": "False"},
        {"claim_text": "You need to drink milk for strong bones.", "rating": "Mostly False"},
        {"claim_text": "Breakfast is the most important meal of the day.", "rating": "Mostly False"},
        {"claim_text": "Cryptocurrency is completely anonymous and untraceable.", "rating": "False"},
        {"claim_text": "Artificial intelligence will take over all human jobs by 2030.", "rating": "False"},
        {"claim_text": "Wind turbines cause cancer.", "rating": "Pants on Fire"},
        {"claim_text": "Fluoride in water is a government mind control scheme.", "rating": "Pants on Fire"},
        {"claim_text": "Chemtrails are chemicals sprayed by planes for population control.", "rating": "Pants on Fire"},
        {"claim_text": "The Loch Ness Monster has been scientifically proven to exist.", "rating": "False"},
        {"claim_text": "Bigfoot sightings have been verified by credible scientists.", "rating": "False"},
        {"claim_text": "Essential oils can cure serious diseases like cancer.", "rating": "False"},
        {"claim_text": "Homeopathy is proven to be effective by scientific studies.", "rating": "False"},
        {"claim_text": "Acupuncture has no measurable health benefits.", "rating": "Mostly False"},
        {"claim_text": "Meditation has been shown to reduce stress and anxiety.", "rating": "True"},
        {"claim_text": "Sleep deprivation has serious negative health consequences.", "rating": "True"},
        {"claim_text": "Regular handwashing prevents the spread of infectious diseases.", "rating": "True"},
        {"claim_text": "Smoking causes lung cancer and other serious health problems.", "rating": "True"},
        {"claim_text": "Seat belts significantly reduce fatalities in car accidents.", "rating": "True"},
        {"claim_text": "The ozone layer has been recovering since the Montreal Protocol.", "rating": "True"},
        {"claim_text": "Recycling reduces the amount of waste sent to landfills.", "rating": "True"},
    ]


# ============================================================
# GOOGLE FACT CHECK API
# ============================================================

def fetch_google_claims(
    api_key: str,
    num_claims: int = 100,
    progress_callback=None,
    query: str = "politics",
):
    """
    Fetch claims from the Google Fact Check Tools API with pagination.

    Parameters
    ----------
    api_key : str
    num_claims : int
    progress_callback : callable(str) | None
        Optional function called with status messages (e.g. st.empty().text).

    Returns
    -------
    list[dict]  Each dict has 'claim_text' and 'rating'.
    """
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    collected = []
    page_token = None

    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    try:
        while len(collected) < num_claims:
            params = {
                "key": api_key,
                "languageCode": "en",
                "pageSize": min(100, num_claims - len(collected)),
                "query": query,
            }
            if page_token:
                params["pageToken"] = page_token

            _log(f"Fetching Google claims… {len(collected)} collected so far")

            # Retry transient failures with exponential backoff.
            last_exc = None
            response = None
            max_attempts = 4
            for attempt in range(1, max_attempts + 1):
                try:
                    response = requests.get(base_url, params=params, timeout=15)

                    # Permanent auth/permission failures should stop immediately.
                    if response.status_code == 401:
                        raise PermissionError("Invalid API key.")
                    if response.status_code == 403:
                        raise PermissionError("API access forbidden – check Cloud Console permissions.")

                    # Retry on throttling and server-side/transient failures.
                    if response.status_code == 429 or response.status_code >= 500:
                        raise requests.exceptions.HTTPError(
                            f"Transient API error {response.status_code}",
                            response=response,
                        )

                    # 4xx (except 401/403/429 above) is usually a permanent request issue.
                    if 400 <= response.status_code < 500:
                        detail = response.text[:400] if response.text else "No response detail"
                        raise RuntimeError(
                            "Google API request rejected with status "
                            f"{response.status_code}: {detail}"
                        )

                    response.raise_for_status()
                    break
                except RuntimeError:
                    raise
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        raise RuntimeError(
                            f"Google API request failed after {max_attempts} attempts: {exc}"
                        )
                    sleep_s = (2 ** (attempt - 1)) + random.uniform(0, 0.4)
                    _log(
                        f"Transient API error ({exc}). Retrying in {sleep_s:.1f}s "
                        f"[{attempt}/{max_attempts}]"
                    )
                    time.sleep(sleep_s)

            if response is None:
                raise RuntimeError(f"Google API request failed: {last_exc}")

            data = response.json()
            if "claims" not in data or not data["claims"]:
                _log(f"Fetched {len(collected)} claims (no more available).")
                break

            for claim_obj in data["claims"]:
                if len(collected) >= num_claims:
                    break
                claim_text = claim_obj.get("text", "")
                reviews = claim_obj.get("claimReview", [])
                if not reviews:
                    continue
                rating = reviews[0].get("textualRating", "")
                if claim_text and rating:
                    collected.append({"claim_text": claim_text, "rating": rating})

            page_token = data.get("nextPageToken")
            if not page_token:
                _log(f"Fetched {len(collected)} claims (all pages processed).")
                break

        _log(f"Successfully fetched {len(collected)} claims.")
        return collected

    except requests.exceptions.RequestException as exc:
        _log(f"Network error: {exc}")
        return collected


def process_and_map_google_claims(api_results: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Map granular Google ratings → binary (1 = True, 0 = False).
    Ambiguous ratings are discarded.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        DataFrame has columns ['claim_text', 'ground_truth'].
        Dict contains counts for total, true, false, and discarded.
    """
    if not api_results:
        return pd.DataFrame(columns=["claim_text", "ground_truth"]), {}

    processed, true_count, false_count, discarded = [], 0, 0, 0

    for item in api_results:
        text   = item.get("claim_text", "").strip()
        rating = item.get("rating", "").strip()

        if not text or len(text) < 10 or not rating:
            discarded += 1
            continue

        norm = rating.lower().strip().rstrip("!").rstrip("?")
        is_true  = any(norm == r.lower() for r in GOOGLE_TRUE_RATINGS)
        is_false = any(norm == r.lower() for r in GOOGLE_FALSE_RATINGS)

        if is_true:
            processed.append({"claim_text": text, "ground_truth": 1})
            true_count += 1
        elif is_false:
            processed.append({"claim_text": text, "ground_truth": 0})
            false_count += 1
        else:
            discarded += 1

    df = pd.DataFrame(processed)
    if not df.empty:
        df = df.drop_duplicates(subset=["claim_text"], keep="first")

    stats = {
        "total": len(api_results),
        "true": true_count,
        "false": false_count,
        "discarded": discarded,
    }
    return df, stats


# ============================================================
# WEB SCRAPING — POLITIFACT
# ============================================================

def scrape_data_by_date_range(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Scrape Politifact fact-check listings for a given date range.

    Parameters
    ----------
    start_date, end_date : pd.Timestamp
    progress_callback : callable(str) | None

    Returns
    -------
    pd.DataFrame with columns: author, statement, source, date, label
    """
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["author", "statement", "source", "date", "label"])
    scraped_count = 0
    page_count = 0

    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    _log(f"Starting scrape: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}")

    while current_url and page_count < 100:
        page_count += 1
        _log(f"Fetching page {page_count}… {scraped_count} claims so far.")

        try:
            resp = requests.get(current_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except requests.exceptions.RequestException as exc:
            _log(f"Network error: {exc}. Stopping.")
            break

        rows_to_add = []

        for card in soup.find_all("li", class_="o-listicle__item"):
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(strip=True) if date_div else None
            claim_date = None

            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        claim_date = pd.to_datetime(match.group(1), format="%B %d, %Y")
                    except ValueError:
                        continue

            if claim_date is None:
                continue

            if start_date <= claim_date <= end_date:
                stmt_block = card.find("div", class_="m-statement__quote")
                statement = (
                    stmt_block.find("a", href=True).get_text(strip=True)
                    if stmt_block and stmt_block.find("a", href=True)
                    else None
                )
                source_a = card.find("a", class_="m-statement__name")
                source   = source_a.get_text(strip=True) if source_a else None
                footer   = card.find("footer", class_="m-statement__footer")
                author   = None
                if footer:
                    m = re.search(r"By\s+([^•]+)", footer.get_text(strip=True))
                    if m:
                        author = m.group(1).strip()
                label_img = card.find("img", alt=True)
                label = (
                    label_img["alt"].replace("-", " ").title()
                    if label_img and "alt" in label_img.attrs
                    else None
                )
                rows_to_add.append(
                    [author, statement, source, claim_date.strftime("%Y-%m-%d"), label]
                )

            elif claim_date < start_date:
                _log(f"Claim older than start date ({start_date:%Y-%m-%d}). Stopping.")
                current_url = None
                break

        if current_url is None:
            break

        writer.writerows(rows_to_add)
        scraped_count += len(rows_to_add)

        next_link = soup.find(
            "a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I)
        )
        if next_link and "href" in next_link.attrs:
            href = next_link["href"].rstrip("&").rstrip("?")
            current_url = urljoin(base_url, href)
        else:
            _log("No more pages found.")
            current_url = None

    _log(f"Scraping finished! Total claims: {scraped_count}")

    output.seek(0)
    df = pd.read_csv(output, header=0, keep_default_na=False)
    df = df.dropna(subset=["statement", "label"])
    df.to_csv(SCRAPED_DATA_PATH, index=False)
    return df


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def lexical_features(text: str) -> str:
    nlp = load_spacy_model()
    doc = nlp(text.lower())
    tokens = [
        tok.lemma_ for tok in doc
        if tok.text not in STOP_WORDS and tok.is_alpha
    ]
    return " ".join(tokens)


def syntactic_features(text: str) -> str:
    nlp = load_spacy_model()
    doc = nlp(text)
    return " ".join(tok.pos_ for tok in doc)


def semantic_features(text: str) -> list:
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]


def discourse_features(text: str) -> str:
    nlp = load_spacy_model()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    first_words = [s.split()[0].lower() for s in sentences if s.split()]
    return f"{len(sentences)} {' '.join(first_words)}"


def pragmatic_features(text: str) -> list:
    text = text.lower()
    return [text.count(w) for w in PRAGMATIC_WORDS]


def apply_feature_extraction(X: pd.Series, phase: str, vectorizer=None):
    """
    Apply the chosen NLP phase to raw text Series X.

    Returns
    -------
    (X_features, vectorizer)
        X_features : sparse matrix or np.ndarray
        vectorizer : fitted vectorizer (or None for dense phases)
    """
    if phase == "Lexical & Morphological":
        X_proc = X.apply(lexical_features)
        vec = vectorizer or CountVectorizer(binary=True, ngram_range=(1, 2))
        return vec.fit_transform(X_proc), vec

    elif phase == "Syntactic":
        X_proc = X.apply(syntactic_features)
        vec = vectorizer or TfidfVectorizer(max_features=5000)
        return vec.fit_transform(X_proc), vec

    elif phase == "Semantic":
        X_feat = pd.DataFrame(
            X.apply(semantic_features).tolist(),
            columns=["polarity", "subjectivity"],
        )
        return X_feat, None

    elif phase == "Discourse":
        X_proc = X.apply(discourse_features)
        vec = vectorizer or CountVectorizer(ngram_range=(1, 2), max_features=5000)
        return vec.fit_transform(X_proc), vec

    elif phase == "Pragmatic":
        X_feat = pd.DataFrame(
            X.apply(pragmatic_features).tolist(),
            columns=PRAGMATIC_WORDS,
        )
        return X_feat, None

    return None, None


# ============================================================
# CLASSIFIERS
# ============================================================

def get_classifier(name: str):
    """Return a fresh classifier instance by name."""
    if name == "Naive Bayes":
        return MultinomialNB()
    elif name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, class_weight="balanced")
    elif name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=42, class_weight="balanced"
        )
    elif name == "SVM":
        return SVC(kernel="linear", C=0.5, random_state=42, class_weight="balanced")
    raise ValueError(f"Unknown classifier: {name}")


# ============================================================
# MODEL TRAINING & EVALUATION
# ============================================================

def evaluate_models(df: pd.DataFrame, selected_phase: str, log_callback=None):
    """
    Train and evaluate 4 classifiers using Stratified K-Fold CV + SMOTE.

    Parameters
    ----------
    df : pd.DataFrame  (must contain 'statement' and 'label' columns)
    selected_phase : str
    log_callback : callable(str) | None

    Returns
    -------
    (results_df, trained_models, vectorizer)
        results_df : pd.DataFrame of per-model metrics
        trained_models : dict[str, fitted estimator]
        vectorizer : fitted vectorizer (or None)
    """

    def _log(msg):
        if log_callback:
            log_callback(msg)

    # --- Binary label mapping ---
    def _to_binary(label):
        if label in REAL_LABELS:
            return 1
        if label in FAKE_LABELS:
            return 0
        return np.nan

    df = df.copy()
    df["target_label"] = df["label"].apply(_to_binary)
    df = df.dropna(subset=["target_label"])
    df = df[df["statement"].astype(str).str.len() > 10]

    X_raw = df["statement"].astype(str)
    y_raw = df["target_label"].astype(int)

    if len(np.unique(y_raw)) < 2:
        raise ValueError("Only one class after binary mapping – cannot train.")

    # --- Feature extraction (full dataset, used to fit vectorizer) ---
    X_features_full, vectorizer = apply_feature_extraction(X_raw, selected_phase)
    if X_features_full is None:
        raise RuntimeError("Feature extraction returned None.")

    if isinstance(X_features_full, pd.DataFrame):
        X_features_full = X_features_full.values

    y = y_raw.values

    # Helper: get phase-specific transform callable
    def _phase_fn():
        if "Lexical" in selected_phase:
            return lexical_features
        if "Syntactic" in selected_phase:
            return syntactic_features
        if "Discourse" in selected_phase:
            return discourse_features
        return None  # dense phases

    # --- K-Fold setup ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    model_names = ["Naive Bayes", "Decision Tree", "Logistic Regression", "SVM"]
    model_metrics = {}

    X_raw_list = X_raw.tolist()

    for name in model_names:
        _log(f"Training {name} with {N_SPLITS}-Fold CV & SMOTE…")
        fold_metrics = {
            "accuracy": [], "f1": [], "precision": [],
            "recall": [], "train_time": [], "inference_time": [],
        }
        model_instance = get_classifier(name)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_features_full, y)):
            X_train_raw = pd.Series([X_raw_list[i] for i in train_idx])
            X_test_raw  = pd.Series([X_raw_list[i] for i in test_idx])
            y_train = y[train_idx]
            y_test  = y[test_idx]

            phase_fn = _phase_fn()
            if vectorizer is not None and phase_fn is not None:
                X_train = vectorizer.transform(X_train_raw.apply(phase_fn))
                X_test  = vectorizer.transform(X_test_raw.apply(phase_fn))
            else:
                X_train, _ = apply_feature_extraction(X_train_raw, selected_phase)
                X_test,  _ = apply_feature_extraction(X_test_raw,  selected_phase)
                if isinstance(X_train, pd.DataFrame):
                    X_train = X_train.values
                    X_test  = X_test.values

            t0 = time.time()
            try:
                if name == "Naive Bayes":
                    clf = get_classifier(name)
                    clf.fit(np.abs(X_train).astype(float), y_train)
                else:
                    clf = ImbPipeline([
                        ("sampler", SMOTE(random_state=42, k_neighbors=3)),
                        ("classifier", get_classifier(name)),
                    ])
                    clf.fit(X_train, y_train)

                train_time = time.time() - t0

                t1 = time.time()
                y_pred = clf.predict(
                    np.abs(X_test).astype(float) if name == "Naive Bayes" else X_test
                )
                infer_time = (time.time() - t1) * 1000

                fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
                fold_metrics["f1"].append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_metrics["precision"].append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_metrics["recall"].append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_metrics["train_time"].append(train_time)
                fold_metrics["inference_time"].append(infer_time)

            except Exception as exc:
                _log(f"  Fold {fold+1} failed for {name}: {exc}")
                for k in fold_metrics:
                    fold_metrics[k].append(0)

        if fold_metrics["accuracy"]:
            model_metrics[name] = {
                "Model": name,
                "Accuracy": np.mean(fold_metrics["accuracy"]) * 100,
                "F1-Score": np.mean(fold_metrics["f1"]),
                "Precision": np.mean(fold_metrics["precision"]),
                "Recall": np.mean(fold_metrics["recall"]),
                "Training Time (s)": round(np.mean(fold_metrics["train_time"]), 2),
                "Inference Latency (ms)": round(np.mean(fold_metrics["inference_time"]), 2),
            }
        else:
            model_metrics[name] = {
                "Model": name, "Accuracy": 0, "F1-Score": 0,
                "Precision": 0, "Recall": 0,
                "Training Time (s)": 0, "Inference Latency (ms)": 9999,
            }

    # --- Final models trained on full data ---
    _log("Training final models on complete dataset for benchmarking…")
    trained_models_final = {}
    phase_fn = _phase_fn()

    for name in model_names:
        try:
            final_clf = get_classifier(name)
            if vectorizer is not None and phase_fn is not None:
                X_final = vectorizer.transform(X_raw.apply(phase_fn))
            else:
                X_final = X_features_full

            if name == "Naive Bayes":
                final_clf.fit(np.abs(X_final).astype(float), y)
                trained_models_final[name] = final_clf
            else:
                pipe = ImbPipeline([
                    ("sampler", SMOTE(random_state=42, k_neighbors=3)),
                    ("classifier", final_clf),
                ])
                pipe.fit(X_final, y)
                trained_models_final[name] = pipe

        except Exception as exc:
            _log(f"Failed to train final {name}: {exc}")
            trained_models_final[name] = None

    results_df = pd.DataFrame(list(model_metrics.values()))
    return results_df, trained_models_final, vectorizer


# ============================================================
# GOOGLE BENCHMARK
# ============================================================

def run_google_benchmark(
    google_df: pd.DataFrame,
    trained_models: dict,
    vectorizer,
    selected_phase: str,
):
    """
    Test trained models on Google fact-check claims.

    Returns
    -------
    pd.DataFrame with per-model benchmark metrics.
    """
    if google_df.empty:
        raise ValueError("No Google claims provided for benchmarking.")

    X_raw = google_df["claim_text"]
    y_true = google_df["ground_truth"].values

    # Feature extraction (same pipeline as training)
    try:
        if selected_phase in ("Lexical & Morphological", "Syntactic", "Discourse"):
            fn_map = {
                "Lexical & Morphological": lexical_features,
                "Syntactic": syntactic_features,
                "Discourse": discourse_features,
            }
            X_proc = X_raw.apply(fn_map[selected_phase])
            X_features = vectorizer.transform(X_proc)
        elif selected_phase == "Semantic":
            X_features = pd.DataFrame(
                X_raw.apply(semantic_features).tolist(),
                columns=["polarity", "subjectivity"],
            ).values
        elif selected_phase == "Pragmatic":
            X_features = pd.DataFrame(
                X_raw.apply(pragmatic_features).tolist(),
                columns=PRAGMATIC_WORDS,
            ).values
        else:
            raise ValueError(f"Unknown phase: {selected_phase}")
    except Exception as exc:
        raise RuntimeError(f"Feature extraction failed: {exc}")

    results = []
    for model_name, model in trained_models.items():
        try:
            X_use = np.abs(X_features).astype(float) if model_name == "Naive Bayes" else X_features
            t0 = time.time()
            y_pred = model.predict(X_use)
            infer_ms = (time.time() - t0) * 1000

            results.append({
                "Model": model_name,
                "Accuracy": accuracy_score(y_true, y_pred) * 100,
                "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "Inference Latency (ms)": round(infer_ms, 2),
            })
        except Exception as exc:
            results.append({
                "Model": model_name,
                "Accuracy": 0, "F1-Score": 0, "Precision": 0, "Recall": 0,
                "Inference Latency (ms)": 9999,
            })

    return pd.DataFrame(results)


# ============================================================
# HUMOUR / CRITIQUE
# ============================================================

def get_phase_critique(best_phase: str) -> str:
    pool = {
        "Lexical & Morphological": [
            "Ah, the Lexical phase. Proving that sometimes, all you need is raw vocabulary and minimal effort. It's the high-school dropout that won the Nobel Prize.",
            "Just words, nothing fancy. This phase decided to ditch the deep thought and focus on counting. Turns out, quantity has a quality all its own.",
            "The Lexical approach: when in doubt, just scream the words louder. It lacks elegance but gets the job done.",
        ],
        "Syntactic": [
            "Syntactic features won? So grammar actually matters! We must immediately inform Congress.",
            "The grammar police have prevailed. This model focused purely on structure, proving that sentence construction is more important than meaning.",
            "It passed the grammar check! This phase is the sensible adult in the room.",
        ],
        "Semantic": [
            "The Semantic phase won by feeling its feelings. It's highly emotional, heavily relying on vibes and tone.",
            "It turns out sentiment polarity is the secret sauce! Zero complex reasoning required.",
            "Semantic victory! The model simply asked, 'Are they being optimistic or negative?'",
        ],
        "Discourse": [
            "Discourse features won! This phase knows the debate structure better than the content.",
            "The long-winded champion! Presentation beats facts.",
            "Discourse is the winner! It successfully mapped the argument's flow.",
        ],
        "Pragmatic": [
            "The Pragmatic phase won by focusing on keywords like 'must' and '?'. It's the Sherlock Holmes of NLP.",
            "It's all about intent! Concise, ruthless, and apparently correct.",
            "Pragmatic features for the win! If someone uses three exclamation marks, they're either lying or selling crypto.",
        ],
    }
    default = ["The results are in, and the system is speechless."]
    return random.choice(pool.get(best_phase, default))


def get_model_critique(best_model: str) -> str:
    pool = {
        "Naive Bayes": [
            "Naive Bayes: fast, simple, assumes every feature is independent. Brilliantly unaware.",
            "The Simpleton Savant has won! It just counts things.",
            "NB pulled off a victory. Less-is-more philosophy.",
        ],
        "Decision Tree": [
            "The Decision Tree won by asking yes/no questions until it got tired.",
            "It built a beautiful set of if/then statements. Most organized model in the office.",
            "Decision Tree victory! Splitting the data until it couldn't be split anymore.",
        ],
        "Logistic Regression": [
            "Logistic Regression: The veteran politician of ML. Boring, reliable, hard to beat.",
            "The Straight-Line Stunner. Predictable, efficient, definitely got tenure.",
            "LogReg prevails! 'Probability is all you need.'",
        ],
        "SVM": [
            "SVM: It found the biggest gap between truth and lies and parked its hyperplane there.",
            "The Maximizing Margin Master! SVM builds a fortress between classes.",
            "SVM crushed it! Hard, clean, dividing line.",
        ],
    }
    default = ["This model broke the simulation, so we have nothing funny to say."]
    return random.choice(pool.get(best_model, default))


def generate_humorous_critique(df_results: pd.DataFrame, selected_phase: str) -> str:
    if df_results.empty:
        return "The system failed to train anything. Our ML models are on strike."

    df_results = df_results.copy()
    df_results["F1-Score"] = pd.to_numeric(df_results["F1-Score"], errors="coerce").fillna(0)
    best_row   = df_results.loc[df_results["F1-Score"].idxmax()]
    best_model = best_row["Model"]
    max_f1     = best_row["F1-Score"]
    max_acc    = best_row["Accuracy"]

    phase_critique = get_phase_critique(selected_phase)
    model_critique = get_model_critique(best_model)

    summary = (
        f"**Accuracy Report Card:** The Golden Snitch Award goes to the **{best_model}**!\n\n"
        f"This absolute unit achieved a **{max_acc:.2f}% Accuracy** (and {max_f1:.2f} F1-Score) "
        f"on the `{selected_phase}` feature set.\n\n"
    )
    roast = (
        f"### The AI Roast:\n"
        f"**Phase Performance:** {phase_critique}\n\n"
        f"**Model Personality:** {model_critique}\n\n"
        f"*(Disclaimer: All models were equally confused by 'Mostly True'.)*"
    )
    return summary + roast
