"""
Sentiment and Intent Analysis Module
Analyzes patient sentiment and detects intent from medical conversations
"""

from typing import List, Dict, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re


class SentimentIntentAnalyzer:
    """
    Analyzes sentiment and intent in medical conversations
    Uses DistilBERT for sentiment and BART for zero-shot intent classification
    """
    
    def __init__(self):
        """Initialize sentiment and intent models"""
        print("Loading sentiment analysis model...")
        
        # Use DistilBERT for sentiment (faster, efficient)
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Could not load sentiment model: {e}")
            self.sentiment_analyzer = None
        
        print("Loading intent classification model...")
        
        # Zero-shot classification for intent detection
        try:
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Could not load intent model: {e}")
            self.intent_classifier = None
        
        # Medical-specific intent categories
        self.intent_labels = [
            "Seeking reassurance",
            "Reporting symptoms",
            "Expressing concern",
            "Asking for information",
            "Describing improvement",
            "Requesting treatment",
            "Confirming understanding",
            "Expressing relief"
        ]
        
        # Sentiment mapping for medical context
        self.medical_sentiment_map = {
            'POSITIVE': 'Reassured',
            'NEGATIVE': 'Anxious',
            'NEUTRAL': 'Neutral'
        }
    
    def analyze_sentiment_basic(self, text: str) -> str:
        """
        Basic sentiment analysis using keyword matching (fallback)
        
        Args:
            text: Input text
            
        Returns:
            Sentiment label (Anxious, Neutral, Reassured)
        """
        text_lower = text.lower()
        
        # Anxious indicators
        anxious_keywords = [
            'worried', 'concern', 'afraid', 'scared', 'nervous', 'anxious',
            'trouble', 'difficult', 'hard', 'struggle', 'hope', 'wish'
        ]
        
        # Reassured indicators
        reassured_keywords = [
            'better', 'relief', 'good', 'great', 'improved', 'happy',
            'glad', 'thankful', 'appreciate', 'comfortable'
        ]
        
        anxious_count = sum(1 for word in anxious_keywords if word in text_lower)
        reassured_count = sum(1 for word in reassured_keywords if word in text_lower)
        
        if anxious_count > reassured_count:
            return 'Anxious'
        elif reassured_count > anxious_count:
            return 'Reassured'
        else:
            return 'Neutral'
    
    def analyze_sentiment(self, text: str, use_model: bool = True) -> Dict:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            use_model: Whether to use ML model (fallback to rules if False)
            
        Returns:
            Dictionary with sentiment label and confidence score
        """
        if not text or not text.strip():
            return {'sentiment': 'Neutral', 'confidence': 0.0}
        
        # Use model if available
        if use_model and self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text[:512])[0]  # Limit token length
                
                # Map to medical context
                raw_sentiment = result['label']
                confidence = result['score']
                
                # Convert to medical sentiment
                if raw_sentiment == 'POSITIVE':
                    sentiment = 'Reassured'
                elif raw_sentiment == 'NEGATIVE':
                    sentiment = 'Anxious'
                else:
                    sentiment = 'Neutral'
                
                # Adjust based on medical keywords
                if confidence < 0.7:  # Low confidence, use rules
                    sentiment = self.analyze_sentiment_basic(text)
                    confidence = 0.6
                
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'raw_label': raw_sentiment
                }
            
            except Exception as e:
                print(f"Sentiment analysis error: {e}")
        
        # Fallback to rule-based
        sentiment = self.analyze_sentiment_basic(text)
        return {
            'sentiment': sentiment,
            'confidence': 0.6,
            'raw_label': sentiment
        }
    
    def detect_intent_basic(self, text: str) -> List[str]:
        """
        Basic intent detection using keyword matching (fallback)
        
        Args:
            text: Input text
            
        Returns:
            List of detected intents
        """
        text_lower = text.lower()
        detected_intents = []
        
        # Intent patterns
        intent_patterns = {
            'Seeking reassurance': [
                r'will (i|it)',
                r'should i',
                r'am i',
                r'is (it|this) normal',
                r'do (i|you) think'
            ],
            'Reporting symptoms': [
                r'(i have|i feel|i\'m experiencing)',
                r'(pain|hurt|ache|discomfort)',
                r'(it|that) (hurts|aches)'
            ],
            'Expressing concern': [
                r'(worried|concerned|afraid)',
                r'what if',
                r'(i|i\'m) not sure'
            ],
            'Describing improvement': [
                r'(better|improved|improving)',
                r'not as (bad|painful)',
                r'(less|reduced) (pain|discomfort)'
            ],
            'Confirming understanding': [
                r'so (you\'re saying|that means)',
                r'(i understand|got it|okay)',
                r'(that makes sense)'
            ],
            'Expressing relief': [
                r'(relief|relieved|glad)',
                r'(good|great) to (hear|know)',
                r'(thank|appreciate)'
            ]
        }
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_intents.append(intent)
                    break
        
        if not detected_intents:
            detected_intents = ['Reporting symptoms']  # Default
        
        return detected_intents
    
    def detect_intent(self, text: str, use_model: bool = True, top_k: int = 2) -> Dict:
        """
        Detect intent(s) in text using zero-shot classification
        
        Args:
            text: Input text
            use_model: Whether to use ML model
            top_k: Number of top intents to return
            
        Returns:
            Dictionary with detected intents and confidence scores
        """
        if not text or not text.strip():
            return {'intents': ['Unknown'], 'scores': [0.0]}
        
        # Use model if available
        if use_model and self.intent_classifier:
            try:
                result = self.intent_classifier(
                    text[:512],  # Limit token length
                    candidate_labels=self.intent_labels,
                    multi_label=True  # Allow multiple intents
                )
                
                # Get top-k intents with scores above threshold
                threshold = 0.3
                intents = []
                scores = []
                
                for label, score in zip(result['labels'][:top_k], result['scores'][:top_k]):
                    if score >= threshold:
                        intents.append(label)
                        scores.append(float(score))
                
                if not intents:  # If no intent above threshold
                    intents = [result['labels'][0]]
                    scores = [float(result['scores'][0])]
                
                return {
                    'intents': intents,
                    'scores': scores,
                    'all_results': list(zip(result['labels'], result['scores']))
                }
            
            except Exception as e:
                print(f"Intent detection error: {e}")
        
        # Fallback to rule-based
        intents = self.detect_intent_basic(text)
        return {
            'intents': intents,
            'scores': [0.6] * len(intents),
            'all_results': []
        }
    
    def analyze_patient_utterances(self, patient_utterances: List[str]) -> List[Dict]:
        """
        Analyze sentiment and intent for each patient utterance
        
        Args:
            patient_utterances: List of patient text segments
            
        Returns:
            List of analysis results for each utterance
        """
        results = []
        
        for i, utterance in enumerate(patient_utterances):
            sentiment = self.analyze_sentiment(utterance)
            intent = self.detect_intent(utterance)
            
            results.append({
                'utterance_id': i,
                'text': utterance[:100] + '...' if len(utterance) > 100 else utterance,
                'sentiment': sentiment['sentiment'],
                'sentiment_confidence': sentiment['confidence'],
                'intents': intent['intents'],
                'intent_scores': intent['scores']
            })
        
        return results
    
    def aggregate_analysis(self, utterance_analyses: List[Dict]) -> Dict:
        """
        Aggregate sentiment and intent across all utterances
        
        Args:
            utterance_analyses: List of per-utterance analyses
            
        Returns:
            Aggregated analysis
        """
        if not utterance_analyses:
            return {
                'overall_sentiment': 'Neutral',
                'dominant_intents': [],
                'sentiment_distribution': {}
            }
        
        # Count sentiments
        sentiment_counts = {}
        for analysis in utterance_analyses:
            sent = analysis['sentiment']
            sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
        
        # Get overall sentiment (most common)
        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        # Count intents
        intent_counts = {}
        for analysis in utterance_analyses:
            for intent in analysis['intents']:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Get top 3 intents
        sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_intents = [intent for intent, _ in sorted_intents[:3]]
        
        return {
            'overall_sentiment': overall_sentiment,
            'dominant_intents': dominant_intents,
            'sentiment_distribution': sentiment_counts,
            'intent_distribution': dict(sorted_intents),
            'num_utterances_analyzed': len(utterance_analyses)
        }


# Example usage
if __name__ == "__main__":
    from preprocessor import MedicalConversationPreprocessor
    
    sample_text = """
    Patient: I'm a bit worried about my back pain, but I hope it gets better soon.
    Patient: The first four weeks were really rough with the pain.
    Patient: It's much better now, thankfully. I don't feel as anxious anymore.
    Patient: That's a relief to hear! Thank you, doctor.
    """
    
    # Preprocess
    preprocessor = MedicalConversationPreprocessor()
    preprocessed = preprocessor.preprocess(sample_text)
    
    # Analyze sentiment and intent
    analyzer = SentimentIntentAnalyzer()
    
    print("=== Sentiment & Intent Analysis ===\n")
    
    # Analyze each patient utterance
    patient_utterances = preprocessed['segmented']['patient']
    analyses = analyzer.analyze_patient_utterances(patient_utterances)
    
    for analysis in analyses:
        print(f"Utterance {analysis['utterance_id'] + 1}: {analysis['text']}")
        print(f"  Sentiment: {analysis['sentiment']} (confidence: {analysis['sentiment_confidence']:.2f})")
        print(f"  Intents: {', '.join(analysis['intents'])}")
        print()
    
    # Aggregate analysis
    aggregated = analyzer.aggregate_analysis(analyses)
    print("=== Aggregated Analysis ===")
    print(f"Overall Sentiment: {aggregated['overall_sentiment']}")
    print(f"Dominant Intents: {', '.join(aggregated['dominant_intents'])}")
    print(f"Sentiment Distribution: {aggregated['sentiment_distribution']}")
