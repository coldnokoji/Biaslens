import pandas as pd
import spacy
import json
import pickle
import re
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    """Custom Dataset for batch processing"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Truncate text if too long
        if len(text) > 500:
            text = text[:500]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'idx': idx
        }

class PoliticalBiasAnalyzer:
    """
    Optimized Political Bias Analysis System using Batch Processing
    Analyzes bias towards BJP and Congress using NER, sentiment, and framing analysis
    """

    def __init__(self, batch_size=32, device=None):
        print("Initializing Political Bias Analyzer...")

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.batch_size = batch_size

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Load sentiment analysis model and tokenizer
        print("Loading sentiment analysis model...")
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Political entity mappings
        self.BJP_ENTITIES = {
            "bjp", "bharatiya janata party", "narendra modi", "modi", "pm modi",
            "amit shah", "shah", "rajnath singh", "nirmala sitharaman",
            "yogi adityanath", "adityanath", "jp nadda", "nda", "national democratic alliance"
        }

        self.CONGRESS_ENTITIES = {
            "congress", "indian national congress", "inc", "rahul gandhi", "rahul",
            "sonia gandhi", "sonia", "priyanka gandhi", "priyanka", "mallikarjun kharge",
            "kharge", "upa", "united progressive alliance"
        }

        # Opinion and hedge markers
        self.OPINION_MARKERS = {
            "think", "believe", "assume", "claim", "allegedly", "reportedly",
            "critics say", "observers note", "analysts suggest", "appears to be",
            "seems to", "purportedly", "supposedly", "opinion", "viewed as"
        }

        self.HEDGE_WORDS = {
            "perhaps", "possibly", "might", "could", "may", "likely",
            "probably", "seemingly", "apparently", "suggests", "indicates"
        }

        # Framing words
        self.NEGATIVE_FRAMES = {
            "mob", "riot", "attack", "accuse", "blame", "controversy",
            "scandal", "corruption", "failure", "crisis", "chaos", "turmoil",
            "strongman", "authoritarian", "dictator", "suppress", "crackdown"
        }

        self.POSITIVE_FRAMES = {
            "leader", "initiative", "reform", "development", "progress",
            "achievement", "success", "growth", "innovation", "stability"
        }

        # Results storage
        self.results = {
            'bjp': defaultdict(list),
            'congress': defaultdict(list)
        }

        self.articles_analyzed = 0

    def load_data(self, filepath, text_column='text', title_column=None):
        """Load dataset from CSV, JSON, or Excel files"""
        print(f"Loading data from {filepath}...")

        if filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            self.df = pd.read_json(filepath)
        elif filepath.endswith('.xlsx'):
            self.df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV, JSON, or XLSX")

        self.text_column = text_column
        self.title_column = title_column

        print(f"Loaded {len(self.df)} articles")
        return self.df

    def identify_party(self, text):
        """Identify which party/parties are mentioned in text"""
        text_lower = text.lower()
        parties = []

        for entity in self.BJP_ENTITIES:
            if entity in text_lower:
                parties.append('bjp')
                break

        for entity in self.CONGRESS_ENTITIES:
            if entity in text_lower:
                parties.append('congress')
                break

        return parties

    def extract_context_window(self, text, entity_pos, window_size=2):
        """Extract sentences around entity mention"""
        doc = self.nlp(text)
        sentences = list(doc.sents)

        # Find sentence containing entity
        target_sent_idx = None
        for idx, sent in enumerate(sentences):
            if entity_pos >= sent.start_char and entity_pos <= sent.end_char:
                target_sent_idx = idx
                break

        if target_sent_idx is None:
            return text[:500]

        # Extract window
        start_idx = max(0, target_sent_idx - window_size)
        end_idx = min(len(sentences), target_sent_idx + window_size + 1)

        context = " ".join([sent.text for sent in sentences[start_idx:end_idx]])
        return context

    def batch_sentiment_analysis(self, texts):
        """
        Perform sentiment analysis on a batch of texts
        Returns: list of (sentiment_label, score) tuples
        """
        if not texts:
            return []

        # Create dataset and dataloader
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

        results = [None] * len(texts)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                indices = batch['idx'].numpy()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Get predictions
                scores, labels = torch.max(predictions, dim=1)

                for idx, label, score in zip(indices, labels.cpu().numpy(), scores.cpu().numpy()):
                    # Map labels (0: negative, 1: neutral, 2: positive)
                    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                    sentiment = label_map.get(label, 'neutral')
                    results[idx] = (sentiment, float(score))

        return results

    def detect_opinion_markers(self, text):
        """Detect opinion and hedge words in text"""
        text_lower = text.lower()

        opinion_count = sum(1 for marker in self.OPINION_MARKERS if marker in text_lower)
        hedge_count = sum(1 for hedge in self.HEDGE_WORDS if hedge in text_lower)

        has_opinion = opinion_count > 0 or hedge_count > 0

        return {
            'has_opinion': has_opinion,
            'opinion_count': opinion_count,
            'hedge_count': hedge_count
        }

    def analyze_framing(self, text):
        """Analyze framing bias using loaded language"""
        text_lower = text.lower()

        negative_count = sum(1 for word in self.NEGATIVE_FRAMES if word in text_lower)
        positive_count = sum(1 for word in self.POSITIVE_FRAMES if word in text_lower)

        return {
            'negative_frames': negative_count,
            'positive_frames': positive_count,
            'framing_score': positive_count - negative_count
        }

    def preprocess_articles(self):
        """
        Preprocess all articles to extract contexts and party mentions
        Returns: dictionary with contexts organized by party
        """
        print("\nPreprocessing articles to extract party contexts...")

        contexts_data = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            text = row[self.text_column]

            if pd.isna(text) or len(str(text).strip()) < 50:
                continue

            text = str(text)
            parties_mentioned = self.identify_party(text)

            if not parties_mentioned:
                continue

            for party in parties_mentioned:
                entity_keywords = self.BJP_ENTITIES if party == 'bjp' else self.CONGRESS_ENTITIES

                # Find first mention
                text_lower = text.lower()
                first_pos = len(text)
                for keyword in entity_keywords:
                    pos = text_lower.find(keyword)
                    if pos != -1 and pos < first_pos:
                        first_pos = pos

                if first_pos == len(text):
                    context = text[:500]
                else:
                    context = self.extract_context_window(text, first_pos)

                contexts_data.append({
                    'article_id': idx,
                    'party': party,
                    'context': context,
                    'full_text': text
                })

        return contexts_data

    def analyze_dataset(self):
        """Analyze entire dataset using batch processing"""
        print(f"\nAnalyzing {len(self.df)} articles...")

        # Preprocess to get all contexts
        contexts_data = self.preprocess_articles()

        if not contexts_data:
            print("No articles with party mentions found!")
            return []

        print(f"Found {len(contexts_data)} party mentions across articles")

        # Extract contexts for batch sentiment analysis
        contexts = [item['context'] for item in contexts_data]

        # Perform batch sentiment analysis
        print("\nPerforming batch sentiment analysis...")
        sentiment_results = self.batch_sentiment_analysis(contexts)

        # Process results
        print("\nProcessing framing and opinion analysis...")
        results_list = []
        article_results_dict = {}

        for item, (sentiment, score) in tqdm(zip(contexts_data, sentiment_results), total=len(contexts_data)):
            party = item['party']
            context = item['context']
            article_id = item['article_id']

            # Opinion detection
            opinion_data = self.detect_opinion_markers(context)

            # Framing analysis
            framing_data = self.analyze_framing(context)

            # Store results
            self.results[party]['sentiments'].append(sentiment)
            self.results[party]['sentiment_scores'].append(score)
            self.results[party]['opinions'].append(opinion_data['has_opinion'])
            self.results[party]['opinion_counts'].append(opinion_data['opinion_count'])
            self.results[party]['framing_scores'].append(framing_data['framing_score'])
            self.results[party]['negative_frames'].append(framing_data['negative_frames'])
            self.results[party]['positive_frames'].append(framing_data['positive_frames'])

            # Track per-article results
            if article_id not in article_results_dict:
                article_results_dict[article_id] = {}

            article_results_dict[article_id][party] = {
                'sentiment': sentiment,
                'sentiment_score': score,
                'opinion_detected': opinion_data['has_opinion'],
                'framing_score': framing_data['framing_score']
            }

        # Convert to list format
        for article_id, analysis in article_results_dict.items():
            results_list.append({
                'article_id': article_id,
                'analysis': analysis
            })

        self.articles_analyzed = len(article_results_dict)
        print(f"\nAnalyzed {self.articles_analyzed} articles with party mentions")

        return results_list

    def calculate_bias_metrics(self):
        """Calculate comprehensive bias metrics"""
        metrics = {}

        for party in ['bjp', 'congress']:
            if not self.results[party]['sentiments']:
                metrics[party] = None
                continue

            sentiments = self.results[party]['sentiments']
            sentiment_counter = Counter(sentiments)
            total = len(sentiments)

            # Sentiment distribution
            positive_pct = (sentiment_counter['positive'] / total) * 100
            negative_pct = (sentiment_counter['negative'] / total) * 100
            neutral_pct = (sentiment_counter['neutral'] / total) * 100

            # Average sentiment score
            avg_sentiment_score = np.mean(self.results[party]['sentiment_scores'])

            # Opinion metrics
            opinion_pct = (sum(self.results[party]['opinions']) / total) * 100
            avg_opinion_count = np.mean(self.results[party]['opinion_counts'])

            # Framing metrics
            avg_framing_score = np.mean(self.results[party]['framing_scores'])
            avg_negative_frames = np.mean(self.results[party]['negative_frames'])
            avg_positive_frames = np.mean(self.results[party]['positive_frames'])

            # Bias score (combined metric)
            bias_score = (positive_pct - negative_pct) / 100
            bias_score -= (opinion_pct / 100) * 0.3
            bias_score += avg_framing_score * 0.2

            metrics[party] = {
                'total_mentions': total,
                'sentiment_distribution': {
                    'positive': positive_pct,
                    'negative': negative_pct,
                    'neutral': neutral_pct
                },
                'avg_sentiment_score': avg_sentiment_score,
                'opinion_percentage': opinion_pct,
                'avg_opinion_markers_per_article': avg_opinion_count,
                'framing': {
                    'avg_framing_score': avg_framing_score,
                    'avg_negative_frames': avg_negative_frames,
                    'avg_positive_frames': avg_positive_frames
                },
                'overall_bias_score': bias_score
            }

        # Comparative metrics
        if metrics['bjp'] and metrics['congress']:
            metrics['comparative'] = {
                'coverage_ratio_bjp_to_congress': metrics['bjp']['total_mentions'] / metrics['congress']['total_mentions'],
                'sentiment_gap': metrics['bjp']['sentiment_distribution']['positive'] - metrics['congress']['sentiment_distribution']['positive'],
                'bias_score_difference': metrics['bjp']['overall_bias_score'] - metrics['congress']['overall_bias_score']
            }

        self.metrics = metrics
        return metrics

    def generate_report(self):
        """Generate human-readable report"""
        print("\n" + "="*70)
        print("POLITICAL BIAS ANALYSIS REPORT")
        print("="*70)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Articles Analyzed: {self.articles_analyzed}")
        print(f"Batch Size Used: {self.batch_size}")
        print(f"Device Used: {self.device}")
        print("="*70)

        for party in ['bjp', 'congress']:
            party_name = party.upper()
            print(f"\n{party_name} ANALYSIS")
            print("-"*70)

            if not self.metrics.get(party):
                print(f"No mentions of {party_name} found in dataset")
                continue

            m = self.metrics[party]

            print(f"Total Mentions: {m['total_mentions']}")
            print(f"\nSentiment Distribution:")
            print(f"  Positive: {m['sentiment_distribution']['positive']:.2f}%")
            print(f"  Negative: {m['sentiment_distribution']['negative']:.2f}%")
            print(f"  Neutral:  {m['sentiment_distribution']['neutral']:.2f}%")

            print(f"\nOpinion Analysis:")
            print(f"  Articles with Opinion Markers: {m['opinion_percentage']:.2f}%")
            print(f"  Avg Opinion Markers per Article: {m['avg_opinion_markers_per_article']:.2f}")

            print(f"\nFraming Analysis:")
            print(f"  Avg Framing Score: {m['framing']['avg_framing_score']:.2f}")
            print(f"  Avg Negative Frames: {m['framing']['avg_negative_frames']:.2f}")
            print(f"  Avg Positive Frames: {m['framing']['avg_positive_frames']:.2f}")

            print(f"\nOverall Bias Score: {m['overall_bias_score']:.3f}")
            print(f"  (Range: -1 to +1, where +1 is most positive)")

        if 'comparative' in self.metrics:
            print(f"\n{'='*70}")
            print("COMPARATIVE ANALYSIS")
            print("-"*70)
            comp = self.metrics['comparative']
            print(f"Coverage Ratio (BJP:Congress): {comp['coverage_ratio_bjp_to_congress']:.2f}:1")
            print(f"Sentiment Gap (BJP - Congress): {comp['sentiment_gap']:.2f}%")
            print(f"Bias Score Difference: {comp['bias_score_difference']:.3f}")

            # Interpretation
            print(f"\nInterpretation:")
            if abs(comp['bias_score_difference']) < 0.1:
                print("  → Relatively balanced coverage")
            elif comp['bias_score_difference'] > 0.1:
                print("  → Slight bias towards BJP")
            else:
                print("  → Slight bias towards Congress")

        print("\n" + "="*70)

    def save_model(self, output_path='bias_analysis_results'):
        """Save all results, metrics, and model state"""
        print(f"\nSaving results to {output_path}...")

        import os
        os.makedirs(output_path, exist_ok=True)

        # Save metrics as JSON
        with open(f'{output_path}/bias_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)

        # Save raw results
        with open(f'{output_path}/raw_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)

        # Save configuration
        config = {
            'bjp_entities': list(self.BJP_ENTITIES),
            'congress_entities': list(self.CONGRESS_ENTITIES),
            'opinion_markers': list(self.OPINION_MARKERS),
            'articles_analyzed': self.articles_analyzed,
            'batch_size': self.batch_size,
            'device': str(self.device),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(f'{output_path}/config.json', 'w') as f:
            json.dump(config, f, indent=4)

        # Save detailed CSV report
        report_data = []
        for party in ['bjp', 'congress']:
            if self.metrics.get(party):
                m = self.metrics[party]
                report_data.append({
                    'Party': party.upper(),
                    'Total_Mentions': m['total_mentions'],
                    'Positive_%': m['sentiment_distribution']['positive'],
                    'Negative_%': m['sentiment_distribution']['negative'],
                    'Neutral_%': m['sentiment_distribution']['neutral'],
                    'Opinion_%': m['opinion_percentage'],
                    'Framing_Score': m['framing']['avg_framing_score'],
                    'Bias_Score': m['overall_bias_score']
                })

        pd.DataFrame(report_data).to_csv(f'{output_path}/summary_report.csv', index=False)

        print(f"✓ Saved bias_metrics.json")
        print(f"✓ Saved raw_results.pkl")
        print(f"✓ Saved config.json")
        print(f"✓ Saved summary_report.csv")
        print(f"\nAll results saved to: {output_path}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze political bias in a dataset")
    parser.add_argument("input_file", help="Path to input CSV/JSON/XLSX file")
    parser.add_argument("--text_col", default="text", help="Column name containing article text")
    parser.add_argument("--output_dir", default="bias_analysis_results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = PoliticalBiasAnalyzer(batch_size=args.batch_size)

    # Load data
    analyzer.load_data(args.input_file, text_column=args.text_col)

    # Analyze
    analyzer.analyze_dataset()

    # Calculate metrics
    analyzer.calculate_bias_metrics()

    # Generate report
    analyzer.generate_report()

    # Save results
    analyzer.save_model(args.output_dir)

    print("\n✓ Analysis complete!")
