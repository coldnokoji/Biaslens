# Prompt for Generating BiasLens Test Data

**Goal:** Generate a CSV dataset to evaluate a political bias detection system for Indian news articles.

**Instructions:**
Please generate a CSV file with 20-30 rows containing real, accessible URLs of news articles related to Indian politics. The CSV should have the following headers: `url`, `main_entity`, `true_stance`, `true_bias_risk`.

**Column Definitions:**

1.  **`url`**: A valid, publicly accessible URL of a news article (from sources like The Hindu, NDTV, OpIndia, Swarajya, The Wire, Indian Express, etc.). **Do not use dead links.**
2.  **`main_entity`**: The primary political party discussed in the article. Must be one of: `BJP`, `Congress`, `AAP`.
3.  **`true_stance`**: An integer representing the article's stance towards the `main_entity`.
    *   `-2`: Strongly Anti / Critical
    *   `-1`: Slightly Anti / Critical
    *   `0`: Neutral / Factual
    *   `1`: Slightly Pro / Favorable
    *   `2`: Strongly Pro / Favorable
4.  **`true_bias_risk`**: An integer representing the level of propaganda or non-objective reporting.
    *   `0`: **Low Risk** (Factual, balanced, neutral tone).
    *   `1`: **Medium Risk** (Some emotional language, slight cherry-picking, or mild one-sidedness).
    *   `2`: **High Risk** (Heavy use of loaded language, name-calling, extreme one-sidedness, propaganda techniques).

**Examples:**

```csv
url,main_entity,true_stance,true_bias_risk
https://www.thehindu.com/news/national/some-factual-report-about-modi-visit.ece,BJP,0,0
https://www.opindia.com/2023/08/article-attacking-rahul-gandhi-strongly/,Congress,-2,2
https://thewire.in/politics/article-critical-of-bjp-policy,BJP,-1,1
https://www.ndtv.com/india-news/neutral-report-on-aap-delhi-model,AAP,1,0
```

**Requirements:**
*   Include a mix of Left-leaning (e.g., The Wire, Scroll), Right-leaning (e.g., OpIndia, Swarajya), and Centrist/Mainstream (e.g., The Hindu, Indian Express) sources.
*   Ensure a balanced distribution of entities (BJP, Congress, AAP).
*   Ensure a mix of bias levels (0, 1, 2).
