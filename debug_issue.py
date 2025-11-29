import traceback
from main import extract_text, split_sentences, analyze_stance, analyze_subjectivity, analyze_loaded_language, analyze_balance

url = "https://www.bjp.org/pressreleases/bjps-press-note-indian-expresss-factually-incorrect-reporting"

print(f"Testing URL: {url}")

try:
    print("Extracting text...")
    text = extract_text(url)
    print(f"Extracted text length: {len(text)}")
    
    print("Splitting sentences...")
    sentences = split_sentences(text)
    print(f"Number of sentences: {len(sentences)}")
    
    print("Analyzing stance...")
    stance = analyze_stance(sentences)
    print(f"Stance results: {stance}")
    
    print("Analyzing subjectivity...")
    subj = analyze_subjectivity(sentences)
    print(f"Subjectivity: {subj}")
    
    print("Analyzing loaded language...")
    loaded = analyze_loaded_language(text)
    print(f"Loaded: {loaded}")
    
    print("Analyzing balance...")
    balance = analyze_balance(sentences)
    print(f"Balance: {balance}")
    
    print("Success!")

except Exception:
    traceback.print_exc()
