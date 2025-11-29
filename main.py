from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import trafilatura
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import List, Dict, Any
import math

app = FastAPI(title="BiasLens")

# --- Configuration & Lexicons ---

import json
import os

# --- Configuration & Lexicons ---

def load_config(filename, default):
    try:
        with open(os.path.join("config", filename), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default.")
        return default

ENTITIES = load_config("entities.json", {})
POLITICAL_LEXICON = load_config("political_lexicon.json", {})
LOADED_LEXICON = load_config("loaded_lexicon.json", [])
OPINION_SIGNALS = load_config("opinion_signals.json", [])
FACT_MARKERS = load_config("fact_markers.json", [])
DISCOURSE_MARKERS = load_config("discourse_markers.json", [])

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(POLITICAL_LEXICON)

# --- Models ---

import requests

# ... (imports)

def extract_text(url: str) -> str:
    # Use requests with a User-Agent to avoid being blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        downloaded = response.text
    except Exception:
        # Fallback to trafilatura fetcher
        downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        raise HTTPException(status_code=400, detail="Could not fetch URL")
    
    text = trafilatura.extract(downloaded)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text")
    return text

def split_sentences(text: str) -> List[str]:
    # Simple split by punctuation, can be improved with nltk/spacy if needed
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def detect_entities(sentence: str) -> List[str]:
    found = []
    lower_s = sentence.lower()
    for entity, aliases in ENTITIES.items():
        if any(alias in lower_s for alias in aliases):
            found.append(entity)
    return list(set(found))

def analyze_stance(sentences: List[str]) -> Dict[str, Any]:
    entity_scores = {e: [] for e in ENTITIES}
    key_sentences = {e: [] for e in ENTITIES}
    
    for sent in sentences:
        ents = detect_entities(sent)
        if not ents:
            continue
        
        score = analyzer.polarity_scores(sent)['compound']
        
        for ent in ents:
            entity_scores[ent].append(score)
            # Store interesting sentences (strong sentiment)
            if abs(score) > 0.3:
                key_sentences[ent].append({"text": sent, "score": score})

    results = []
    for ent, scores in entity_scores.items():
        if not scores:
            continue
        avg_score = sum(scores) / len(scores)
        # Map [-1, 1] to [0, 100]
        meter = int((avg_score + 1) * 50)
        
        label = "Neutral"
        if avg_score > 0.5: label = "Strongly Pro"
        elif avg_score > 0.1: label = "Slightly Pro"
        elif avg_score < -0.5: label = "Strongly Anti"
        elif avg_score < -0.1: label = "Slightly Anti"
        
        results.append({
            "name": ent,
            "avg_sentiment": round(avg_score, 2),
            "stance_meter": meter,
            "stance_label": f"{label}-{ent}",
            "key_sentences": sorted(key_sentences[ent], key=lambda x: abs(x['score']), reverse=True)[:3]
        })
    
    return results

def analyze_subjectivity(sentences: List[str]) -> float:
    opinion_count = 0
    fact_count = 0
    for sent in sentences:
        lower_s = sent.lower()
        if any(w in lower_s for w in OPINION_SIGNALS):
            opinion_count += 1
        if any(w in lower_s for w in FACT_MARKERS) or re.search(r'\d+', sent): # Numbers often indicate facts
            fact_count += 1
            
    total = opinion_count + fact_count
    if total == 0: return 0.0
    return opinion_count / total

def analyze_loaded_language(text: str) -> float:
    lower_text = text.lower()
    words = lower_text.split()
    count = sum(1 for w in words if w in LOADED_LEXICON)
    # Normalize: say 5 loaded words per 1000 is "high" (1.0)
    per_1000 = (count / len(words)) * 1000 if words else 0
    return min(1.0, per_1000 / 5.0)

def analyze_balance(sentences: List[str]) -> float:
    balance_count = 0
    for sent in sentences:
        lower_s = sent.lower()
        if any(w in lower_s for w in DISCOURSE_MARKERS):
            balance_count += 1
    
    # Normalize: say 1 marker every 10 sentences is "good balance"
    ratio = balance_count / len(sentences) if sentences else 0
    normalized_balance = min(1.0, ratio * 10) 
    return 1.0 - normalized_balance # High balance score -> Low bias

# --- API Endpoints ---

class AnalyzeRequest(BaseModel):
    url: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# ... (previous code)

# Load Historical Bias Metrics
HISTORICAL_METRICS = {}
try:
    # Load Indian Express metrics as an example
    ie_path = os.path.join("historical_data_analyis_result", "indian-express", "bias_metrics.json")
    with open(ie_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        # Store by domain
        HISTORICAL_METRICS["indianexpress.com"] = metrics
        HISTORICAL_METRICS["www.indianexpress.com"] = metrics
        print("Loaded historical metrics for Indian Express")
except FileNotFoundError:
    print("Warning: Historical metrics for Indian Express not found.")

def get_source_metrics(url: str) -> Dict[str, Any]:
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower()
    return HISTORICAL_METRICS.get(domain)

@app.post("/analyze")
async def analyze_article(request: AnalyzeRequest):
    try:
        text = extract_text(request.url)
        sentences = split_sentences(text)
        
        # Run Models
        stance_results = analyze_stance(sentences)
        subjectivity_score = analyze_subjectivity(sentences)
        loaded_bias = analyze_loaded_language(text)
        balance_bias = analyze_balance(sentences)
        
        # Model 5: Source History
        source_metrics = get_source_metrics(request.url)
        source_history = 0.0
        
        if source_metrics and 'comparative' in source_metrics:
             diff = abs(source_metrics['comparative']['bias_score_difference'])
             source_history = min(1.0, diff * 5)
        
        max_stance_dev = 0
        for res in stance_results:
            dev = abs(res['avg_sentiment'])
            if dev > max_stance_dev:
                max_stance_dev = dev
        
        overall_bias = (
            0.3 * max_stance_dev +
            0.2 * subjectivity_score +
            0.2 * loaded_bias +
            0.2 * balance_bias +
            0.1 * source_history
        )
        
        bias_label = "Low Bias Risk"
        if overall_bias > 0.6: bias_label = "High Bias Risk"
        elif overall_bias > 0.3: bias_label = "Medium Bias Risk"
        
        explanation = [
            f"Subjectivity Score: {subjectivity_score:.2f} (High opinion/fact ratio)",
            f"Loaded Language Score: {loaded_bias:.2f} (Use of inflammatory words)",
            f"Lack of Balance: {balance_bias:.2f} (Few counter-perspectives detected)"
        ]
        
        if source_history > 0.1:
            explanation.append(f"Source History: {source_history:.2f} (Domain has known historical bias)")

        return {
            "entities": stance_results,
            "bias_risk": round(overall_bias, 2),
            "bias_label": bias_label,
            "explanation": explanation,
            "historical_data": source_metrics
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug-analyze")
async def debug_analyze_article(request: AnalyzeRequest):
    try:
        text = extract_text(request.url)
        sentences = split_sentences(text)
        
        # Run Models
        stance_results = analyze_stance(sentences)
        subjectivity_score = analyze_subjectivity(sentences)
        loaded_bias = analyze_loaded_language(text)
        balance_bias = analyze_balance(sentences)
        source_history = 0.0 
        
        max_stance_dev = 0
        for res in stance_results:
            dev = abs(res['avg_sentiment'])
            if dev > max_stance_dev:
                max_stance_dev = dev
        
        overall_bias = (
            0.3 * max_stance_dev +
            0.2 * subjectivity_score +
            0.2 * loaded_bias +
            0.2 * balance_bias +
            0.1 * source_history
        )
        
        return {
            "stance_results": stance_results,
            "subjectivity_score": subjectivity_score,
            "loaded_bias": loaded_bias,
            "balance_bias": balance_bias,
            "overall_bias": overall_bias
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
