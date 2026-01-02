# Quick Reference Guide - Medical NLP Pipeline

## 🚀 Quick Start

### Installation (5 minutes)
```bash
# Linux/Mac
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

### Run Pipeline (1 command)
```bash
python pipeline.py
```

---

## 📝 Common Tasks

### Task 1: Analyze a Single Conversation
```python
from pipeline import MedicalNLPPipeline

pipeline = MedicalNLPPipeline()
results = pipeline.process_conversation(your_text, generate_soap=True)
pipeline.print_summary(results)
```

### Task 2: Get Only Medical Entities
```python
from preprocessor import MedicalConversationPreprocessor
from medical_ner import MedicalNER

preprocessor = MedicalConversationPreprocessor()
preprocessed = preprocessor.preprocess(text)

ner = MedicalNER()
summary = ner.generate_summary(preprocessed)
print(summary['symptoms'], summary['diagnosis'])
```

### Task 3: Analyze Sentiment Only
```python
from sentiment_intent import SentimentIntentAnalyzer

analyzer = SentimentIntentAnalyzer()
result = analyzer.analyze_sentiment("I'm worried about my pain")
print(result['sentiment'])  # Output: Anxious, Neutral, or Reassured
```

### Task 4: Generate SOAP Note
```python
from pipeline import MedicalNLPPipeline

pipeline = MedicalNLPPipeline(verbose=False)
results = pipeline.process_conversation(text, generate_soap=True)
print(pipeline.soap_generator.format_soap_note_text(results['soap_note']))
```

### Task 5: Batch Process Multiple Files
```python
from pipeline import MedicalNLPPipeline
import os

pipeline = MedicalNLPPipeline(verbose=False)

for filename in os.listdir('conversations/'):
    with open(f'conversations/{filename}', 'r') as f:
        text = f.read()
    results = pipeline.process_conversation(text)
    pipeline.save_results(results, f'output/{filename}.json')
```

---

## 🔧 Configuration

### Change Models (in code or config.py)
```python
# Use different sentiment model
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

# Use different intent model
INTENT_MODEL = "typeform/distilbert-base-uncased-mnli"
```

### Adjust Confidence Thresholds
```python
SENTIMENT_CONFIDENCE_THRESHOLD = 0.6
INTENT_CONFIDENCE_THRESHOLD = 0.3
```

### Add Custom Intent Categories
```python
INTENT_LABELS = [
    "Seeking reassurance",
    "Reporting symptoms",
    "Your custom intent here"
]
```

---

## 📊 Output Structure

### JSON Output Format
```json
{
  "medical_summary": {
    "patient_name": "Ms. Jones",
    "symptoms": ["neck pain", "back pain"],
    "diagnosis": ["whiplash injury"],
    "treatment": ["physiotherapy"],
    "prognosis": "Full recovery expected"
  },
  "sentiment_intent": {
    "aggregated": {
      "overall_sentiment": "Reassured",
      "dominant_intents": ["Reporting symptoms"]
    }
  },
  "soap_note": { ... }
}
```

### Accessing Results
```python
# Get patient name
patient = results['medical_summary']['patient_name']

# Get symptoms list
symptoms = results['medical_summary']['symptoms']

# Get overall sentiment
sentiment = results['sentiment_intent']['aggregated']['overall_sentiment']

# Get SOAP note sections
subjective = results['soap_note']['subjective']
objective = results['soap_note']['objective']
```

---

## ⚡ Performance Tips

### Speed Up Processing
1. **Disable verbose mode**: `MedicalNLPPipeline(verbose=False)`
2. **Skip SOAP generation**: `generate_soap=False`
3. **Use CPU for small texts**: Set `USE_GPU = False` in config.py
4. **Use lighter models**: Change to DistilBERT variants

### Improve Accuracy
1. **Install medical spaCy model**: `pip install en_core_sci_md`
2. **Use GPU**: Ensure PyTorch with CUDA is installed
3. **Adjust confidence thresholds**: Lower for more results, higher for precision
4. **Add domain keywords**: Update keyword sets in config.py

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Model not found"** | Run `python -m spacy download en_core_web_sm` |
| **Out of memory** | Set `USE_GPU = False` in config.py |
| **Slow processing** | Reduce `MAX_SEQUENCE_LENGTH` or use CPU |
| **No symptoms extracted** | Add keywords to `SYMPTOM_KEYWORDS` in config.py |
| **Wrong sentiment** | Adjust `SENTIMENT_CONFIDENCE_THRESHOLD` |

---

## 📋 File Structure Reference

```
medical-nlp-pipeline/
├── pipeline.py              # Main orchestrator - START HERE
├── preprocessor.py          # Text cleaning & speaker separation
├── medical_ner.py          # Extract symptoms, diagnosis, treatment
├── sentiment_intent.py     # Analyze sentiment & detect intent
├── soap_generator.py       # Generate SOAP notes
├── config.py               # Configuration settings
├── example_usage.py        # Example code
├── requirements.txt        # Dependencies
└── README.md              # Full documentation
```

---

## 🎯 Key Functions Reference

### Preprocessor
```python
preprocessor = MedicalConversationPreprocessor()
result = preprocessor.preprocess(text)
# Returns: utterances, patient_text, doctor_text, temporal_info
```

### Medical NER
```python
ner = MedicalNER()
summary = ner.generate_summary(preprocessed_data)
keywords = ner.extract_keywords(text, top_n=10)
```

### Sentiment & Intent
```python
analyzer = SentimentIntentAnalyzer()
sentiment = analyzer.analyze_sentiment(text)
intent = analyzer.detect_intent(text, top_k=2)
analyses = analyzer.analyze_patient_utterances(utterance_list)
aggregated = analyzer.aggregate_analysis(analyses)
```

### SOAP Generator
```python
generator = SOAPGenerator()
soap = generator.generate_soap_note(preprocessed, summary, sentiment, intent)
text = generator.format_soap_note_text(soap)
```

### Complete Pipeline
```python
pipeline = MedicalNLPPipeline(verbose=True)
results = pipeline.process_conversation(text, generate_soap=True)
pipeline.print_summary(results)
pipeline.save_results(results, "output.json")
```

---

## 💡 Pro Tips

1. **Test with sample data first**: Run `python example_usage.py` before your data
2. **Save intermediate results**: Enable in config.py for debugging
3. **Use batch processing**: Process multiple files efficiently
4. **Customize keywords**: Add domain-specific terms to config.py
5. **Check output folder**: All results saved to `output/` directory

---

## 🔗 Important Links

- **Full Documentation**: See README.md
- **Examples**: Run `python example_usage.py`
- **Configuration**: Edit config.py
- **Support**: Check README.md troubleshooting section

---

## ⌨️ Command Cheat Sheet

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run
python pipeline.py                    # Run with default sample
python example_usage.py              # Interactive examples
python config.py                     # View configuration

# Test modules individually
python preprocessor.py
python medical_ner.py
python sentiment_intent.py
python soap_generator.py
```

