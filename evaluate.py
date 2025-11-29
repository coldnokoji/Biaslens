import csv
import sys
import argparse
from main import extract_text, split_sentences, analyze_stance, analyze_subjectivity, analyze_loaded_language, analyze_balance, ENTITIES

def get_stance_label(score):
    # Map [-1, 1] to -2, -1, 0, 1, 2
    # Logic from main.py:
    # > 0.5: Strongly Pro (2)
    # > 0.1: Slightly Pro (1)
    # < -0.5: Strongly Anti (-2)
    # < -0.1: Slightly Anti (-1)
    # Else: Neutral (0)
    if score > 0.5: return 2
    elif score > 0.1: return 1
    elif score < -0.5: return -2
    elif score < -0.1: return -1
    return 0

def get_bias_risk_label(score):
    # Logic from main.py:
    # > 0.6: High (2)
    # > 0.3: Medium (1)
    # Else: Low (0)
    if score > 0.6: return 2
    elif score > 0.3: return 1
    return 0

def evaluate(csv_file):
    print(f"Evaluating using {csv_file}...")
    
    y_true_stance = []
    y_pred_stance = []
    y_true_bias = []
    y_pred_bias = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
        return

    total = len(rows)
    print(f"Found {total} articles.")

    for i, row in enumerate(rows):
        url = row['url']
        entity = row['main_entity']
        true_stance = int(row['true_stance'])
        true_bias = int(row['true_bias_risk'])
        
        print(f"[{i+1}/{total}] Analyzing {url}...")
        
        try:
            text = extract_text(url)
            sentences = split_sentences(text)
            
            # Stance
            stance_results = analyze_stance(sentences)
            # Find score for the main entity
            pred_stance_score = 0.0
            for res in stance_results:
                if res['name'].lower() == entity.lower():
                    pred_stance_score = res['avg_sentiment']
                    break
            
            # Bias Risk
            subjectivity = analyze_subjectivity(sentences)
            loaded = analyze_loaded_language(text)
            balance = analyze_balance(sentences)
            source = 0.0
            
            max_stance_dev = 0
            for res in stance_results:
                dev = abs(res['avg_sentiment'])
                if dev > max_stance_dev: max_stance_dev = dev
            
            overall_bias = (0.3 * max_stance_dev + 0.2 * subjectivity + 
                            0.2 * loaded + 0.2 * balance + 0.1 * source)
            
            # Discretize
            pred_stance_label = get_stance_label(pred_stance_score)
            pred_bias_label = get_bias_risk_label(overall_bias)
            
            y_true_stance.append(true_stance)
            y_pred_stance.append(pred_stance_label)
            y_true_bias.append(true_bias)
            y_pred_bias.append(pred_bias_label)
            
        except Exception as e:
            print(f"  Error processing {url}: {e}")
            continue

    # Metrics
    print("\n--- Results ---")
    
    # Stance Accuracy
    correct_stance = sum(1 for t, p in zip(y_true_stance, y_pred_stance) if t == p)
    close_stance = sum(1 for t, p in zip(y_true_stance, y_pred_stance) if abs(t - p) <= 1)
    total_processed = len(y_true_stance)
    
    if total_processed == 0:
        print("No articles processed successfully.")
        return

    print(f"Stance Accuracy (Exact): {correct_stance}/{total_processed} ({correct_stance/total_processed:.2%})")
    print(f"Stance Accuracy (Within 1): {close_stance}/{total_processed} ({close_stance/total_processed:.2%})")
    
    # Bias Accuracy
    correct_bias = sum(1 for t, p in zip(y_true_bias, y_pred_bias) if t == p)
    print(f"Bias Risk Accuracy: {correct_bias}/{total_processed} ({correct_bias/total_processed:.2%})")
    
    # Confusion Matrix (Stance)
    print("\nStance Confusion Matrix (Rows=True, Cols=Pred):")
    # Labels: -2, -1, 0, 1, 2
    labels = [-2, -1, 0, 1, 2]
    matrix = {l: {l2: 0 for l2 in labels} for l in labels}
    
    for t, p in zip(y_true_stance, y_pred_stance):
        matrix[t][p] += 1
        
    print("    " + " ".join(f"{l:>3}" for l in labels))
    for l in labels:
        row_str = " ".join(f"{matrix[l][l2]:>3}" for l2 in labels)
        print(f"{l:>3} {row_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to test set CSV")
    args = parser.parse_args()
    evaluate(args.csv_file)
