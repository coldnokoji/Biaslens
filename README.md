# BiasLens 🔍
**AI-Powered Political Bias & Propaganda Detector (No LLMs)**

BiasLens is a full-stack Python web application designed to detect political bias, stance, and propaganda techniques in news articles. Unlike modern generative AI approaches, BiasLens relies entirely on **classical NLP techniques**, lexicon-based analysis, and statistical heuristics to ensure deterministic and explainable results.

## 🚀 How It Works

1.  **Input**: The user provides a URL of a news article.
2.  **Extraction**: The app uses `trafilatura` to scrape and extract the main body text, removing ads and boilerplate.
3.  **Analysis**: The text is processed by a "Council of Models"—five independent analytical modules that assess different dimensions of bias.
4.  **Aggregation**: Scores from all models are weighted and combined to produce a final "Bias Risk Score".
5.  **Visualization**: The frontend displays per-entity stance meters, a risk bar, and highlights evidence in the text.

## 🧠 The "Council of Models" Approach

BiasLens avoids a single "black box" model. Instead, it aggregates insights from five distinct sub-models:

### 1. Stance Detection Model
*   **Goal**: Determine if the article is Pro, Anti, or Neutral towards specific political entities (BJP, Congress, AAP).
*   **Method**:
    *   **Entity Recognition**: Keyword-based matching using a dictionary of aliases (e.g., "Modi" -> BJP, "RaGa" -> Congress).
    *   **Sentiment Analysis**: Uses **VADER** (Valence Aware Dictionary and sEntiment Reasoner), optimized for social media and short texts.
    *   **Lexicon Injection**: We inject a custom Indian Political Lexicon into VADER to catch domain-specific nuances (e.g., "vikas" (+), "scam" (-), "dynastic" (-)).
    *   **Scoring**: Average sentiment of sentences mentioning an entity.

### 2. Subjectivity Model
*   **Goal**: Distinguish between factual reporting and opinionated writing.
*   **Method**: Calculates the ratio of "Opinion Signals" to "Fact Markers".
    *   *Opinion Signals*: "I think", "clearly", "undoubtedly", "shocking".
    *   *Fact Markers*: "According to", "data", "report said", numbers, dates.
*   **Logic**: Higher ratio = Higher Subjectivity = Higher Bias Risk.

### 3. Loaded Language Model
*   **Goal**: Detect inflammatory or emotionally charged vocabulary (Propaganda).
*   **Method**: Frequency analysis using a curated **Propaganda Lexicon**.
    *   *Terms*: "anti-national", "urban naxal", "dictator", "puppet", "draconian".
*   **Scoring**: Normalized frequency per 1000 tokens.

### 4. Balance / Fairness Model
*   **Goal**: Check if the article presents multiple viewpoints.
*   **Method**: Searches for **Discourse Markers** that indicate counter-arguments.
    *   *Markers*: "however", "on the other hand", "conversely", "despite".
*   **Logic**: Lack of these markers suggests a one-sided narrative.

### 5. Source History Model (Placeholder)
*   **Goal**: Account for the historical reputation of the domain.
*   **Method**: (Currently a placeholder) Intended to look up the domain in a database of known partisan outlets.

---

## ⚙️ Configuration

BiasLens uses external JSON configuration files located in the `config/` directory. You can customize these files to tune the model:

*   `entities.json`: Define political entities and their aliases.
*   `political_lexicon.json`: Sentiment scores for political terms.
*   `loaded_lexicon.json`: List of propaganda/inflammatory words.
*   `opinion_signals.json` & `fact_markers.json`: For subjectivity analysis.
*   `discourse_markers.json`: For balance detection.

## 📜 Historical Data Analysis

BiasLens includes a module to analyze historical bias of news sources.
The script `historical_analysis.py` (derived from `historical_data_analysis.ipynb`) can process large datasets of articles to compute long-term bias metrics.

To run the analysis:
```bash
python historical_analysis.py path/to/dataset.csv --text_col "News Content"
```

The results (JSON metrics) can be placed in `historical_data_analyis_result/<domain>/bias_metrics.json` to be used by the live application.

## 📊 Evaluation

To evaluate the model against a test set:

1.  Prepare a CSV file (e.g., `test_set.csv`) with columns: `url,main_entity,true_stance,true_bias_risk`.
2.  Run the evaluation script:
    ```bash
    python evaluate.py test_set.csv
    ```

## 🛠️ Tech Stack

*   **Backend**: FastAPI (Python) - High-performance web framework.
*   **NLP**: `vaderSentiment`, `trafilatura` (Scraping), `re` (Regex).
*   **Frontend**: HTML5, CSS3 (Modern Dark Theme), Vanilla JavaScript.
*   **Design**: Responsive UI with CSS Variables and Flexbox.

## 📦 Installation & Running

### Prerequisites
*   Python 3.8+
*   pip

### Steps

1.  **Clone/Download the repository**
2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Server**:
    ```bash
    python main.py
    ```
5.  **Access the App**:
    Open your browser and go to `http://localhost:8000` (or port 3000 if configured).

## 📂 Project Structure

```
BiasLens/
├── config/              # JSON Configuration files
├── main.py              # FastAPI backend & Logic
├── evaluate.py          # Evaluation script
├── test_set.csv         # Sample test data
├── index.html           # Frontend UI
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```
