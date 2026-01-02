"""
Configuration file for Medical NLP Pipeline
Adjust settings here to customize pipeline behavior
"""

# ==========================================
# MODEL CONFIGURATION
# ==========================================

# Sentiment Analysis Model
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# Alternatives:
# - "cardiffnlp/twitter-roberta-base-sentiment"
# - "nlptown/bert-base-multilingual-uncased-sentiment"

# Intent Classification Model (Zero-shot)
INTENT_MODEL = "facebook/bart-large-mnli"
# Alternatives:
# - "typeform/distilbert-base-uncased-mnli"
# - "cross-encoder/nli-deberta-v3-large"

# Medical NER - Use SciBERT/BioBERT models
USE_MEDICAL_SPACY = True  # Use en_core_sci_md if available
USE_CLINICAL_BERT = False  # Set to True when clinical BERT model is available

# ==========================================
# INTENT CATEGORIES
# ==========================================

# Medical conversation intent labels
INTENT_LABELS = [
    "Seeking reassurance",
    "Reporting symptoms",
    "Expressing concern",
    "Asking for information",
    "Describing improvement",
    "Requesting treatment",
    "Confirming understanding",
    "Expressing relief",
    "Reporting side effects",
    "Requesting clarification"
]

# ==========================================
# SENTIMENT MAPPING
# ==========================================

# Map model outputs to medical context
SENTIMENT_MEDICAL_MAPPING = {
    'POSITIVE': 'Reassured',
    'NEGATIVE': 'Anxious',
    'NEUTRAL': 'Neutral',
    # Add more mappings if using different models
}

# Confidence thresholds
SENTIMENT_CONFIDENCE_THRESHOLD = 0.6
INTENT_CONFIDENCE_THRESHOLD = 0.3

# ==========================================
# PREPROCESSING SETTINGS
# ==========================================

# Speaker identification patterns
DOCTOR_PATTERNS = [
    r'^physician\s*:\s*',
    r'^doctor\s*:\s*',
    r'^dr\.?\s*:\s*',
    r'^\[?physical examination',
]

PATIENT_PATTERNS = [
    r'^patient\s*:\s*',
    r'^pt\.?\s*:\s*',
]

# Text cleaning options
REMOVE_ASTERISKS = True
NORMALIZE_WHITESPACE = True
NORMALIZE_QUOTES = True

# ==========================================
# MEDICAL ENTITY KEYWORDS
# ==========================================

# Symptom-related keywords for rule-based extraction
SYMPTOM_KEYWORDS = {
    'pain', 'ache', 'discomfort', 'hurt', 'sore', 'tender', 'stiff',
    'swelling', 'inflammation', 'headache', 'nausea', 'fever', 'dizzy',
    'fatigue', 'weak', 'shock', 'anxiety', 'nervous', 'trouble sleeping',
    'bleeding', 'bruising', 'numbness', 'tingling', 'burning'
}

# Treatment-related keywords
TREATMENT_KEYWORDS = {
    'therapy', 'physiotherapy', 'treatment', 'medication', 'painkiller',
    'analgesic', 'surgery', 'prescription', 'exercise', 'rest', 'ice',
    'heat', 'massage', 'session', 'dose', 'pill', 'tablet', 'injection',
    'physical therapy', 'occupational therapy', 'counseling'
}

# Diagnosis-related keywords
DIAGNOSIS_KEYWORDS = {
    'injury', 'strain', 'sprain', 'fracture', 'whiplash', 'concussion',
    'trauma', 'damage', 'condition', 'syndrome', 'disease', 'disorder',
    'infection', 'inflammation', 'tear', 'rupture'
}

# Prognosis-related keywords
PROGNOSIS_KEYWORDS = {
    'recovery', 'heal', 'improve', 'progress', 'expectation', 'outlook',
    'prognosis', 'better', 'worse', 'stable', 'chronic', 'acute',
    'long-term', 'short-term', 'permanent', 'temporary'
}

# ==========================================
# SOAP NOTE SETTINGS
# ==========================================

# Section identification indicators
SUBJECTIVE_INDICATORS = [
    'patient', 'reports', 'complains', 'states', 'describes',
    'feels', 'experiencing', 'noticed', 'mentioned', 'history'
]

OBJECTIVE_INDICATORS = [
    'examination', 'physical exam', 'observed', 'found',
    'appears', 'shows', 'demonstrates', 'range of motion',
    'vital signs', 'measurement', 'palpation', 'inspection'
]

ASSESSMENT_INDICATORS = [
    'diagnosis', 'diagnosed', 'assessed', 'impression',
    'condition', 'determined', 'concluded', 'evaluation'
]

PLAN_INDICATORS = [
    'plan', 'treatment', 'recommend', 'prescribe', 'follow-up',
    'continue', 'advise', 'schedule', 'return', 'monitor'
]

# ==========================================
# OUTPUT SETTINGS
# ==========================================

# Output directory
OUTPUT_DIR = "output"

# JSON formatting
JSON_INDENT = 2
JSON_ENSURE_ASCII = False

# Verbose output
VERBOSE = True  # Set to False for minimal output

# Save intermediate results
SAVE_PREPROCESSED = True
SAVE_NER_RESULTS = True
SAVE_SENTIMENT_RESULTS = True
SAVE_SOAP_NOTE = True

# ==========================================
# PERFORMANCE SETTINGS
# ==========================================

# Device selection
USE_GPU = True  # Set to False to force CPU usage
GPU_DEVICE = 0   # GPU device ID (if multiple GPUs available)

# Text truncation limits (for model inputs)
MAX_SEQUENCE_LENGTH = 512
MAX_UTTERANCE_LENGTH = 256

# Batch processing
BATCH_SIZE = 8  # For batch processing multiple conversations

# ==========================================
# ADVANCED SETTINGS
# ==========================================

# Multi-label intent detection
ALLOW_MULTIPLE_INTENTS = True
MAX_INTENTS_PER_UTTERANCE = 2

# Keyword extraction
TOP_N_KEYWORDS = 10
MIN_KEYWORD_LENGTH = 3
MAX_KEYWORD_PHRASE_LENGTH = 3  # Max words in a phrase

# Temporal entity extraction
EXTRACT_DATES = True
EXTRACT_DURATIONS = True

# ==========================================
# VALIDATION SETTINGS
# ==========================================

# Minimum data requirements for processing
MIN_UTTERANCES = 2
MIN_PATIENT_UTTERANCES = 1
MIN_DOCTOR_UTTERANCES = 1

# Warning thresholds
WARN_IF_NO_SYMPTOMS = True
WARN_IF_NO_DIAGNOSIS = True
WARN_IF_LOW_CONFIDENCE = True
LOW_CONFIDENCE_THRESHOLD = 0.5

# ==========================================
# ERROR HANDLING
# ==========================================

# Fallback behavior when models fail
USE_RULE_BASED_FALLBACK = True
CONTINUE_ON_MODEL_ERROR = True

# Default values for missing data
DEFAULT_PATIENT_NAME = "Unknown Patient"
DEFAULT_DIAGNOSIS = "Pending further evaluation"
DEFAULT_PROGNOSIS = "To be determined"
DEFAULT_SENTIMENT = "Neutral"

# ==========================================
# LOGGING
# ==========================================

# Logging configuration
ENABLE_LOGGING = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "output/pipeline.log"
LOG_TO_CONSOLE = True
LOG_TO_FILE = False

# ==========================================
# FEATURE FLAGS
# ==========================================

# Enable/disable specific features
ENABLE_SENTIMENT_ANALYSIS = True
ENABLE_INTENT_DETECTION = True
ENABLE_SOAP_GENERATION = True
ENABLE_KEYWORD_EXTRACTION = True
ENABLE_TEMPORAL_EXTRACTION = True

# Experimental features
ENABLE_ENTITY_LINKING = False  # Link entities to medical databases
ENABLE_MEDICAL_CODE_EXTRACTION = False  # Extract ICD-10 codes


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_device():
    """Get device for model inference"""
    import torch
    if USE_GPU and torch.cuda.is_available():
        return GPU_DEVICE
    return -1  # CPU


def validate_config():
    """Validate configuration settings"""
    warnings = []
    
    if MIN_UTTERANCES < 2:
        warnings.append("MIN_UTTERANCES should be at least 2 for meaningful analysis")
    
    if SENTIMENT_CONFIDENCE_THRESHOLD > 0.9:
        warnings.append("SENTIMENT_CONFIDENCE_THRESHOLD is very high, may reject valid results")
    
    if MAX_SEQUENCE_LENGTH > 512 and SENTIMENT_MODEL.startswith("distilbert"):
        warnings.append("MAX_SEQUENCE_LENGTH > 512 may cause issues with DistilBERT")
    
    if warnings:
        print("⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    return len(warnings) == 0


if __name__ == "__main__":
    print("=" * 60)
    print("Medical NLP Pipeline - Configuration")
    print("=" * 60)
    print(f"\nModels:")
    print(f"  Sentiment: {SENTIMENT_MODEL}")
    print(f"  Intent: {INTENT_MODEL}")
    print(f"  Medical spaCy: {'Enabled' if USE_MEDICAL_SPACY else 'Disabled'}")
    print(f"\nDevice: {'GPU' if USE_GPU else 'CPU'}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Verbose Mode: {'Enabled' if VERBOSE else 'Disabled'}")
    print(f"\nIntent Categories: {len(INTENT_LABELS)}")
    print(f"Symptom Keywords: {len(SYMPTOM_KEYWORDS)}")
    print(f"Treatment Keywords: {len(TREATMENT_KEYWORDS)}")
    print("\n" + "=" * 60)
    
    # Validate configuration
    if validate_config():
        print("\n✓ Configuration validated successfully")
    else:
        print("\n⚠️  Configuration has warnings (see above)")
