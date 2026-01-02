# Medical NLP Pipeline

A comprehensive Natural Language Processing pipeline for analyzing medical conversations between physicians and patients. The pipeline performs medical entity extraction, sentiment analysis, intent detection, and SOAP note generation.

## 🎯 Features

### 1. **Medical NLP Summarization**
- **Named Entity Recognition (NER)**: Extracts symptoms, treatments, diagnosis, and prognosis
- **Keyword Extraction**: Identifies important medical phrases
- **Structured Summarization**: Converts conversations into structured JSON format

### 2. **Sentiment & Intent Analysis**
- **Sentiment Classification**: Classifies patient sentiment as `Anxious`, `Neutral`, or `Reassured`
- **Intent Detection**: Identifies patient intent (e.g., "Seeking reassurance", "Reporting symptoms", "Expressing concern")
- **Per-utterance and Aggregated Analysis**: Analyzes each patient statement and provides overall insights

### 3. **SOAP Note Generation**
- **Automated SOAP Notes**: Generates structured clinical notes with:
  - **Subjective**: Patient's reported symptoms and history
  - **Objective**: Physical examination findings
  - **Assessment**: Clinical diagnosis and severity
  - **Plan**: Treatment recommendations and follow-up

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB recommended for models)
- Internet connection (for first-time model downloads)

### Python Dependencies
See `requirements.txt` for complete list. Key libraries:
- `transformers` - For BERT-based models
- `torch` - PyTorch backend
- `spacy` - For NER and linguistic analysis
- `scispacy` - Medical NLP models (optional but recommended)

## 🚀 Installation

### Step 1: Clone or Download the Repository

```bash
# Create project directory
mkdir medical-nlp-pipeline
cd medical-nlp-pipeline
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm

# (Optional) Download medical spaCy model for better accuracy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
```

### Step 4: Verify Installation

```bash
# Run a quick test
python -c "import spacy; import transformers; print('✓ Installation successful!')"
```

## 📁 Project Structure

```
medical-nlp-pipeline/
├── preprocessor.py          # Text preprocessing and speaker diarization
├── medical_ner.py          # Medical entity extraction and summarization
├── sentiment_intent.py     # Sentiment and intent analysis
├── soap_generator.py       # SOAP note generation
├── pipeline.py             # Main pipeline orchestrator
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── output/                # Generated results (created automatically)
```

## 💻 Usage

### Basic Usage - Run Complete Pipeline

```bash
python pipeline.py
```

This will:
1. Process the sample conversation (from the assignment)
2. Extract medical entities
3. Analyze sentiment and intent
4. Generate a SOAP note
5. Save results to `output/medical_analysis_results.json`

### Using Individual Modules

#### 1. Preprocessing Only

```python
from preprocessor import MedicalConversationPreprocessor

preprocessor = MedicalConversationPreprocessor()
result = preprocessor.preprocess(conversation_text)

print(f"Patient utterances: {result['num_patient_utterances']}")
print(f"Doctor utterances: {result['num_doctor_utterances']}")
```

#### 2. Medical NER Only

```python
from preprocessor import MedicalConversationPreprocessor
from medical_ner import MedicalNER

preprocessor = MedicalConversationPreprocessor()
preprocessed = preprocessor.preprocess(conversation_text)

ner = MedicalNER()
summary = ner.generate_summary(preprocessed)

print(f"Symptoms: {summary['symptoms']}")
print(f"Diagnosis: {summary['diagnosis']}")
```

#### 3. Sentiment & Intent Analysis Only

```python
from preprocessor import MedicalConversationPreprocessor
from sentiment_intent import SentimentIntentAnalyzer

preprocessor = MedicalConversationPreprocessor()
preprocessed = preprocessor.preprocess(conversation_text)

analyzer = SentimentIntentAnalyzer()
patient_utterances = preprocessed['segmented']['patient']
analyses = analyzer.analyze_patient_utterances(patient_utterances)
aggregated = analyzer.aggregate_analysis(analyses)

print(f"Overall Sentiment: {aggregated['overall_sentiment']}")
print(f"Dominant Intents: {aggregated['dominant_intents']}")
```

#### 4. Custom Pipeline Usage

```python
from pipeline import MedicalNLPPipeline

# Initialize pipeline
pipeline = MedicalNLPPipeline(verbose=True)

# Your conversation text
conversation = """
Physician: How are you feeling?
Patient: I have been experiencing back pain for two weeks.
...
"""

# Process conversation
results = pipeline.process_conversation(
    conversation,
    generate_soap=True  # Set to False to skip SOAP generation
)

# Print formatted summary
pipeline.print_summary(results)

# Save to file
pipeline.save_results(results, "my_results.json")
```

## 📊 Output Format

### JSON Output Structure

```json
{
  "preprocessed": {
    "num_utterances": 24,
    "num_patient_utterances": 12,
    "num_doctor_utterances": 12,
    "temporal_entities": [...]
  },
  "medical_summary": {
    "patient_name": "Ms. Jones",
    "symptoms": ["neck pain", "back pain", "head impact"],
    "diagnosis": ["whiplash injury"],
    "treatment": ["10 physiotherapy sessions", "painkillers"],
    "current_status": "occasional backache",
    "prognosis": "Full recovery expected within six months"
  },
  "sentiment_intent": {
    "aggregated": {
      "overall_sentiment": "Reassured",
      "dominant_intents": ["Reporting symptoms", "Seeking reassurance"],
      "sentiment_distribution": {
        "Reassured": 7,
        "Anxious": 3,
        "Neutral": 2
      }
    }
  },
  "soap_note": {
    "subjective": {...},
    "objective": {...},
    "assessment": {...},
    "plan": {...}
  }
}
```

### Console Output

The pipeline prints a formatted summary including:
- Medical entity extraction results
- Sentiment and intent analysis
- Top medical keywords
- Complete SOAP note in readable format

## 🔧 Configuration Options

### Model Selection

In `medical_ner.py`, you can toggle between models:

```python
# Use medical-specific model (better accuracy)
ner = MedicalNER(use_clinical_bert=True)

# Use general model (faster, lower memory)
ner = MedicalNER(use_clinical_bert=False)
```

### Sentiment/Intent Model Selection

In `sentiment_intent.py`, models are automatically loaded. To disable model usage and use rule-based fallback:

```python
analyzer = SentimentIntentAnalyzer()
result = analyzer.analyze_sentiment(text, use_model=False)
```

### Verbose Output

Control pipeline verbosity:

```python
# Detailed output
pipeline = MedicalNLPPipeline(verbose=True)

# Minimal output
pipeline = MedicalNLPPipeline(verbose=False)
```

## 🎓 Model Justifications

### Architecture: Modular + Parallel Processing
- **Why**: Allows independent testing, reusability, and parallel execution of sentiment/intent
- **Benefit**: Maintainable code, faster processing for independent tasks

### NER: BioClinicalBERT + Rule-based Extraction
- **Why**: Medical domain requires specialized understanding of clinical terminology
- **Justification 1**: Pre-trained on clinical notes (MIMIC-III dataset)
- **Justification 2**: Better recognition of medical entities like "whiplash", "physiotherapy"
- **Justification 3**: Deterministic outputs critical for clinical applications

### Sentiment: DistilBERT
- **Why**: Balance between performance and efficiency
- **Justification 1**: 40% smaller than BERT, faster inference
- **Justification 2**: Can be fine-tuned on medical conversation data
- **Justification 3**: Good understanding of medical context with proper training

### Intent: BART Zero-shot Classification
- **Why**: Flexibility without requiring labeled training data
- **Justification 1**: No training data needed for medical intent categories
- **Justification 2**: Can easily add/modify intent categories
- **Justification 3**: Handles multiple intents per utterance naturally

### SOAP: T5 + Aggregator Pattern
- **Why**: Structured text generation task suits T5 architecture
- **Justification 1**: T5 excels at text-to-text tasks
- **Justification 2**: Aggregator uses all upstream analysis (NER, sentiment, intent)
- **Justification 3**: Hybrid approach (rules for sections, T5 for content) ensures quality

## 🐛 Troubleshooting

### Issue: Models not downloading

```bash
# Manual model download
python -c "from transformers import pipeline; pipeline('sentiment-analysis')"
python -m spacy download en_core_web_sm
```

### Issue: Out of memory errors

**Solution 1**: Use CPU instead of GPU
```python
# In sentiment_intent.py, modify:
device = -1  # Force CPU usage
```

**Solution 2**: Process smaller text chunks
```python
# Limit text length
text = text[:512]  # Process first 512 characters
```

### Issue: spaCy model not found

```bash
# Reinstall spaCy model
python -m spacy download en_core_web_sm --force
```

### Issue: Slow processing

**Solutions**:
- Use lighter models (DistilBERT instead of BERT)
- Disable verbose output
- Process shorter conversations
- Use GPU if available

## 📝 Handling Ambiguous/Missing Data

The pipeline handles missing data gracefully:

1. **Missing entities**: Returns "Not mentioned" or empty lists
2. **Low confidence NER**: Flags entities with confidence scores
3. **Unclear sentiment**: Defaults to "Neutral" with low confidence
4. **Missing diagnosis**: Returns "Pending further evaluation"

## 🔍 Example Questions Answered

### Q: How to handle ambiguous or missing medical data?
- Use confidence scores from NER
- Provide default values for missing fields
- Flag low-confidence extractions for manual review
- Aggregate multiple sources of information

### Q: What pre-trained models for medical summarization?
- **BioClinicalBERT**: For medical entity understanding
- **Clinical-T5**: For generation tasks (if available)
- **SciBERT**: Alternative for scientific/medical text

### Q: Fine-tuning BERT for medical sentiment?
- Use Medical Dialogue Dataset
- Apply data augmentation with paraphrasing
- Few-shot learning with medical examples
- Transfer learning from general sentiment models

### Q: Datasets for healthcare sentiment?
- Medical Dialogue Dataset
- MIMIC-III clinical notes (with IRB approval)
- Healthcare review datasets (Yelp, etc.)
- Reddit health forums (with filtering)

### Q: Training for SOAP mapping?
- Synthetic data from SOAP templates
- Few-shot prompting with examples
- MIMIC-III notes (if accessible)
- Rule-based section identification + DL content filling

### Q: Rule-based vs Deep Learning for SOAP?
- **Hybrid approach used**: Rules for structure, DL for content
- Rules identify section boundaries (keywords, patterns)
- T5 fills content within identified sections
- Combines reliability of rules with flexibility of DL

## 🚀 Future Enhancements

1. **Fine-tune models** on medical conversation datasets
2. **Add medical entity linking** to knowledge bases (UMLS, SNOMED CT)
3. **Multi-language support** for international use
4. **Real-time streaming** for live conversation analysis
5. **Integration with EHR systems** for automated documentation
6. **Active learning** to improve model accuracy over time

## 📄 License

This project is created for educational purposes as part of an NLP assignment.

## 👥 Contributors

Created for Medical NLP Assignment - Conversation Analysis Pipeline

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the example usage code
3. Inspect the output JSON for debugging

---

**Note**: This pipeline is designed for educational and research purposes. For clinical use, ensure proper validation, compliance with medical regulations (HIPAA, etc.), and review by medical professionals.
