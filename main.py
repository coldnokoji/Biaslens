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

ENTITIES = {
    "BJP": ["bjp", "modi", "narendra modi", "bharatiya janata party", "saffron party", "lotus"],
    "Congress": ["congress", "inc", "rahul gandhi", "sonia gandhi", "kharge", "grand old party"],
    "AAP": ["aap", "aam aadmi party", "arvind kejriwal", "kejriwal", "sisodia"]
}

# Extended Political Lexicon (Inject into VADER)
POLITICAL_LEXICON = {
    # Negative
    "scam": -3.0, "corruption": -3.0, "communal": -2.5, "dynastic": -2.0, 
    "anti-national": -3.5, "dictator": -3.0, "fascist": -3.0, "appeasement": -2.0,
    "jumla": -2.0, "puppet": -2.0, "incompetent": -2.0, "failure": -2.0,
    
    # Positive
    "vikas": 2.5, "development": 2.0, "welfare": 2.0, "inclusive": 2.0,
    "visionary": 2.5, "historic": 2.0, "masterstroke": 2.5, "reform": 1.5,
    "growth": 1.5, "empowerment": 2.0
}

# Propaganda/Loaded Language Lexicon
LOADED_LEXICON = [
    "anti-national", "urban naxal", "tukde", "sickular", "bhakt",
    "godi media", "corrupt", "traitor", "shameless", "evil", "draconian",
    "bizarre", "shocking", "brutal", "massacre", "genocide", "propaganda"
]

# Subjectivity Markers
OPINION_SIGNALS = ["i think", "clearly", "everyone knows", "undoubtedly", "obviously", "believe", "feel"]
FACT_MARKERS = ["report said", "according to", "stated", "official", "data", "study", "survey"]

# Balance Markers
DISCOURSE_MARKERS = ["however", "but", "on the other hand", "although", "conversely", "despite", "while"]

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(POLITICAL_LEXICON)

# --- Models ---

def extract_text(url: str) -> str:
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
    with open("index.html", "r") as f:
        return f.read()

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
        source_history = 0.0 # Placeholder
        
        # Council of Models Aggregation
        # Weights: Stance(0.3), Subj(0.2), Loaded(0.2), Balance(0.2), History(0.1)
        
        # Stance contribution: Deviation from neutral (0.5 in 0-1 scale, or 0 in -1 to 1)
        # We'll take the max absolute stance of any entity as the stance bias risk
        max_stance_dev = 0
        for res in stance_results:
            dev = abs(res['avg_sentiment']) # 0 to 1
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

        return {
            "entities": stance_results,
            "bias_risk": round(overall_bias, 2),
            "bias_label": bias_label,
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
