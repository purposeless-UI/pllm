import os
import re

def clean_sanskrit_line(line):
    """Removes junk dividers and strips unnecessary formatting characters."""
    line = re.sub(r'={2,}', '', line)  # Removes text dividers like ============
    line = re.sub(r'\s+', ' ', line)   # Collapses multiple blank spaces
    return line.strip()

def build_panini_master_dataset(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file '{input_path}' not found!")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Breakdown lines into a clean array
    lines = [clean_sanskrit_line(l) for l in raw_text.splitlines() if clean_sanskrit_line(l)]
    dataset_rows = []

    # State tracking variables for cross-line structural blocks
    last_speaker = None
    last_query = None
    shloka_accumulator = []
    
    i = 0
    while i < len(lines):
        line = lines[i]

        # ---------------------------------------------------------------------
        # 1. CONVERSATIONAL PATTERNS (व्यवहारिकसंवादाः)
        # ---------------------------------------------------------------------
        # Detects lines tracking dynamic character chat pairs (Ram, Shyam, Guru, Shishya, etc.)
        if any(line.startswith(prefix) for prefix in ["रामः।", "श्यामः।", "गुरुः।", "शिष्याः।", "शिष्यः।", "माता।", "पुत्रः।", "पिता।", "सर्वे।"]):
            speaker_tag = line.split("।")[0].strip()
            
            # Extract content from the immediate next line
            if i + 1 < len(lines):
                dialogue_text = lines[i + 1]
                
                # Check if this speaker tag represents an active question or directive statement
                if speaker_tag in ["रामः", "गुरुः", "माता", "पिता"]:
                    last_query = dialogue_text
                # If it's a response tag, fuse it to create a coherent context pair
                elif speaker_tag in ["श्यामः", "शिष्याः", "शिष्यः", "पुत्रः"] and last_query:
                    dataset_rows.append(f"[BOS] 👤 User: {last_query} 🤖 Panini AI: {dialogue_text} [EOS]")
                    last_query = None  # Reset query tracking field
                    
                i += 2
                continue

        # ---------------------------------------------------------------------
        # 2. SACRED TEXT RECITATIONS & WORD BREAKDOWNS (Upanishad Shlokas)
        # ---------------------------------------------------------------------
        # Checks for multi-line full metric verses ending with standard danda sequences
        elif line.endswith("।") or line.endswith("॥") or any(keyword in line for keyword in ["ॐ", "पूर्णमदः", "ईशावास्य"]):
            if "॥" in line or line == "ॐ" or "शान्तिः" in line:
                # Terminal danda found, package the verse out into explicit training pairs
                dataset_rows.append(f"[BOS] 👤 User: मन्त्रंRecite: {line} 🤖 Panini AI: {line} [EOS]")
                shloka_accumulator = []
            else:
                # Accumulate multi-line strings safely
                shloka_accumulator.append(line)
            i += 1
            continue

        # ---------------------------------------------------------------------
        # 3. GRAMMAR MATRICES & TENSE CONJUGATIONS (लट्, लोट्, लङ्, लिङ्, लृट्, लुट्, लिट्, लुङ्, लृङ्, लेट्)
        # ---------------------------------------------------------------------
        elif any(pronoun in line for pronoun in ["अहं", "त्वं", "सः", "सा", "वयं", "यूयं", "ते", "अहम्", "त्वम्", "वयम्", "यूयम्"]):
            # Split pronouns and actions to convert structural conjugation charts into dynamic language logic
            dataset_rows.append(f"[BOS] 👤 User: धातुप्रयोगः कः: '{line}'? 🤖 Panini AI: {line} [EOS]")
            i += 1
            continue

        # ---------------------------------------------------------------------
        # 4. STANDALONE SENTENCE PROCESSING
        # ---------------------------------------------------------------------
        elif "।" in line:
            clean_phrase = line.replace("।", "").strip()
            # Generate common educational language queries automatically based on syntax markers
            if "कुत्र" in line or "विद्यालय" in line:
                dataset_rows.append(f"[BOS] 👤 User: भवान् कुत्र गच्छति? 🤖 Panini AI: {line} [EOS]")
            elif "किं" in line or "पठति" in line or "संस्कृत" in line:
                dataset_rows.append(f"[BOS] 👤 User: भवान् किं करोति? 🤖 Panini AI: {line} [EOS]")
            else:
                dataset_rows.append(f"[BOS] 👤 User: वाक्यप्रयोगं कुरु: {clean_phrase} 🤖 Panini AI: {line} [EOS]")
            i += 1
            continue

        else:
            # Catch isolated words/vocabulary elements and handle as lookup tokens
            dataset_rows.append(f"[BOS] 👤 User: पदच्छेदः / शब्दः: {line} 🤖 Panini AI: {line} [EOS]")
            i += 1

    # Write out structural compilation
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dataset_rows))

    print(f"🚀 Master Dataset compilation complete!")
    print(f"📊 Processed and outputted {len(dataset_rows)} fully interactive conversational training vectors.")
    print(f"💾 File compiled securely at: {output_path}")

if __name__ == "__main__":
    # 1. Locate the main project folder where prepare_data.py lives
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Point directly to your raw data inside the data/raw/ folder
    INPUT_FILE = os.path.join(SCRIPT_DIR, "data", "raw", "sample_sanskrit_scripture.txt") 
    
    # 3. Save the clean conversational outputs into the data/processed/ folder
    OUTPUT_FILE = os.path.join(SCRIPT_DIR, "data", "processed", "formatted_panini_train.txt")
    
    print(f"🔍 Searching for raw text data at: {INPUT_FILE}")
    print(f"🎯 Output target mapped safely to: {OUTPUT_FILE}")
    
    # 4. Execute the compilation logic
    build_panini_master_dataset(INPUT_FILE, OUTPUT_FILE)
