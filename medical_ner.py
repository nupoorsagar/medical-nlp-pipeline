"""
Medical NER and Summarization Module
Extracts medical entities (symptoms, treatments, diagnosis, prognosis) and generates summaries
"""

import re
from typing import List, Dict, Set
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch


class MedicalNER:
    """
    Medical Named Entity Recognition and Summarization
    Uses BioClinicalBERT for medical entity extraction
    """
    
    def __init__(self, use_clinical_bert: bool = True):
        """
        Initialize NER models
        
        Args:
            use_clinical_bert: Whether to use clinical BERT (requires download)
        """
        # Load spaCy for general entities
        try:
            self.nlp = spacy.load("en_core_sci_md")
            print("Loaded scispacy medical model")
        except OSError:
            print("Medical model not found, using general model...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
        
        # Medical entity patterns (rule-based fallback)
        self.symptom_keywords = {
            'pain', 'ache', 'discomfort', 'hurt', 'sore', 'tender', 'stiff',
            'swelling', 'inflammation', 'headache', 'nausea', 'fever', 'dizzy',
            'fatigue', 'weak', 'shock', 'anxiety', 'nervous', 'trouble sleeping'
        }
        
        self.treatment_keywords = {
            'therapy', 'physiotherapy', 'treatment', 'medication', 'painkiller',
            'analgesic', 'surgery', 'prescription', 'exercise', 'rest', 'ice',
            'heat', 'massage', 'session', 'dose', 'pill', 'tablet', 'injection'
        }
        
        self.diagnosis_keywords = {
            'injury', 'strain', 'sprain', 'fracture', 'whiplash', 'concussion',
            'trauma', 'damage', 'condition', 'syndrome', 'disease', 'disorder'
        }
        
        self.prognosis_keywords = {
            'recovery', 'heal', 'improve', 'progress', 'expectation', 'outlook',
            'prognosis', 'better', 'worse', 'stable', 'chronic', 'acute'
        }
    
    def extract_entities_spacy(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using spaCy
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        doc = self.nlp(text)
        
        entities = {
            'persons': [],
            'dates': [],
            'locations': [],
            'organizations': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
        
        return entities
    
    def extract_medical_entities_rules(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities using rule-based approach
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with symptoms, treatments, diagnosis, prognosis
        """
        text_lower = text.lower()
        doc = self.nlp(text)
        
        entities = {
            'symptoms': [],
            'treatments': [],
            'diagnosis': [],
            'prognosis': []
        }
        
        # Extract symptoms
        for token in doc:
            # Check if token or its lemma matches symptom keywords
            if token.lemma_ in self.symptom_keywords or token.text.lower() in self.symptom_keywords:
                # Get noun chunk context
                for chunk in doc.noun_chunks:
                    if token in chunk:
                        entities['symptoms'].append(chunk.text)
                        break
        
        # Pattern-based extraction for body parts + pain
        pain_pattern = r'(\w+)\s+(pain|ache|discomfort|hurt|injury)'
        for match in re.finditer(pain_pattern, text_lower):
            symptom = match.group(0)
            if symptom not in [s.lower() for s in entities['symptoms']]:
                entities['symptoms'].append(symptom.title())
        
        # Extract treatments
        for token in doc:
            if token.lemma_ in self.treatment_keywords or token.text.lower() in self.treatment_keywords:
                for chunk in doc.noun_chunks:
                    if token in chunk:
                        entities['treatments'].append(chunk.text)
                        break
        
        # Pattern for treatment sessions
        session_pattern = r'(\d+)\s+(?:sessions?|treatments?|visits?)\s+(?:of\s+)?(\w+)'
        for match in re.finditer(session_pattern, text_lower):
            treatment = f"{match.group(1)} {match.group(2)} sessions"
            entities['treatments'].append(treatment.title())
        
        # Extract diagnosis
        for token in doc:
            if token.lemma_ in self.diagnosis_keywords or token.text.lower() in self.diagnosis_keywords:
                for chunk in doc.noun_chunks:
                    if token in chunk:
                        entities['diagnosis'].append(chunk.text)
                        break
        
        # Extract prognosis
        prognosis_phrases = []
        for sent in doc.sents:
            sent_lower = sent.text.lower()
            if any(keyword in sent_lower for keyword in self.prognosis_keywords):
                prognosis_phrases.append(sent.text.strip())
        
        entities['prognosis'] = prognosis_phrases
        
        # Remove duplicates and clean
        for key in entities:
            if key != 'prognosis':
                entities[key] = list(set([e.strip() for e in entities[key] if e.strip()]))
        
        return entities
    
    def extract_patient_info(self, text: str) -> Dict[str, str]:
        """
        Extract patient information from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with patient details
        """
        general_entities = self.extract_entities_spacy(text)
        
        # Extract patient name (usually first person mentioned)
        patient_name = "Unknown"
        if general_entities['persons']:
            patient_name = general_entities['persons'][0]
        
        # Try to find name patterns like "Ms. Jones", "Mr. Smith"
        name_pattern = r'(Ms\.|Mrs\.|Mr\.|Dr\.)\s+([A-Z][a-z]+)'
        name_match = re.search(name_pattern, text)
        if name_match:
            patient_name = f"{name_match.group(1)} {name_match.group(2)}"
        
        return {
            'patient_name': patient_name,
            'mentioned_dates': general_entities['dates'],
            'locations': general_entities['locations']
        }
    
    def extract_current_status(self, patient_text: str) -> str:
        """
        Extract current symptom status from patient utterances
        
        Args:
            patient_text: Combined patient text
            
        Returns:
            Current status description
        """
        # Look for phrases indicating current state
        current_patterns = [
            r'(now|currently|still|these days)\s+.*?(pain|discomfort|ache|better|worse)',
            r'(occasional|sometimes|rarely)\s+.*?(pain|discomfort|ache)',
            r'(I|i)\s+(have|get|feel|experience)\s+.*?(pain|discomfort)',
        ]
        
        status_mentions = []
        for pattern in current_patterns:
            matches = re.finditer(pattern, patient_text.lower())
            for match in matches:
                status_mentions.append(match.group(0))
        
        if status_mentions:
            return "; ".join(set(status_mentions))
        
        return "Status not explicitly mentioned"
    
    def generate_summary(self, preprocessed_data: Dict) -> Dict:
        """
        Generate comprehensive medical summary
        
        Args:
            preprocessed_data: Output from preprocessor
            
        Returns:
            Structured medical summary
        """
        full_text = preprocessed_data['full_text']
        patient_text = preprocessed_data['patient_text']
        
        # Extract all entities
        medical_entities = self.extract_medical_entities_rules(full_text)
        patient_info = self.extract_patient_info(full_text)
        current_status = self.extract_current_status(patient_text)
        
        # Build structured summary
        summary = {
            'patient_name': patient_info['patient_name'],
            'symptoms': medical_entities['symptoms'],
            'diagnosis': medical_entities['diagnosis'],
            'treatment': medical_entities['treatments'],
            'current_status': current_status,
            'prognosis': medical_entities['prognosis'][0] if medical_entities['prognosis'] else "Not mentioned",
            'temporal_info': {
                'dates_mentioned': patient_info['mentioned_dates'],
                'locations': patient_info['locations']
            }
        }
        
        return summary
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract important medical keywords/phrases
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        doc = self.nlp(text)
        
        # Collect noun chunks and named entities
        candidates = []
        
        # Add noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to 3-word phrases
                candidates.append(chunk.text.lower())
        
        # Add named entities
        for ent in doc.ents:
            candidates.append(ent.text.lower())
        
        # Add medical keywords found
        for word in self.symptom_keywords.union(self.treatment_keywords, 
                                                 self.diagnosis_keywords):
            if word in text.lower():
                candidates.append(word)
        
        # Count frequency and return top N
        from collections import Counter
        keyword_counts = Counter(candidates)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(top_n)]
        
        return top_keywords


# Example usage
if __name__ == "__main__":
    from preprocessor import MedicalConversationPreprocessor
    
    sample_text = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September. Can you walk me through what happened?
    Patient: Yes, it was on September 1st. I had neck and back pain for four weeks.
    Physician: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions. The doctor said it was whiplash injury.
    Patient: Now I only get occasional backaches.
    Physician: Your recovery looks good. I expect full recovery within six months.
    """
    
    # Preprocess
    preprocessor = MedicalConversationPreprocessor()
    preprocessed = preprocessor.preprocess(sample_text)
    
    # Extract medical entities
    ner = MedicalNER()
    summary = ner.generate_summary(preprocessed)
    
    print("=== Medical NER Summary ===")
    print(f"Patient: {summary['patient_name']}")
    print(f"\nSymptoms: {summary['symptoms']}")
    print(f"Diagnosis: {summary['diagnosis']}")
    print(f"Treatment: {summary['treatment']}")
    print(f"Current Status: {summary['current_status']}")
    print(f"Prognosis: {summary['prognosis']}")
    
    keywords = ner.extract_keywords(preprocessed['full_text'])
    print(f"\nTop Keywords: {keywords[:5]}")
