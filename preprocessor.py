"""
Preprocessor Module for Medical Conversation NLP Pipeline
Handles text cleaning, speaker diarization, and sentence segmentation
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import spacy

@dataclass
class Utterance:
    """Represents a single utterance in the conversation"""
    speaker: str  # 'doctor' or 'patient'
    text: str
    original_text: str
    utterance_id: int

class MedicalConversationPreprocessor:
    """
    Preprocesses medical conversations for downstream NLP tasks
    """
    
    def __init__(self):
        """Initialize preprocessor with spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Patterns for speaker identification
        self.doctor_patterns = [
            r'^physician\s*:\s*',
            r'^doctor\s*:\s*',
            r'^dr\.?\s*:\s*',
            r'^\[?physical examination',
        ]
        
        self.patient_patterns = [
            r'^patient\s*:\s*',
            r'^pt\.?\s*:\s*',
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove asterisks and formatting markers
        text = re.sub(r'\*+', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def identify_speaker(self, line: str) -> Tuple[str, str]:
        """
        Identify speaker and extract text from a line
        
        Args:
            line: Single line from conversation
            
        Returns:
            Tuple of (speaker, text)
        """
        line_lower = line.lower()
        
        # Check for doctor patterns
        for pattern in self.doctor_patterns:
            if re.match(pattern, line_lower):
                text = re.sub(pattern, '', line_lower, flags=re.IGNORECASE)
                return 'doctor', self.clean_text(text)
        
        # Check for patient patterns
        for pattern in self.patient_patterns:
            if re.match(pattern, line_lower):
                text = re.sub(pattern, '', line_lower, flags=re.IGNORECASE)
                return 'patient', self.clean_text(text)
        
        # If no pattern matches, try to infer from context
        if any(word in line_lower for word in ['physician', 'doctor', 'dr.']):
            return 'doctor', self.clean_text(line)
        elif 'patient' in line_lower:
            return 'patient', self.clean_text(line)
        
        # Default to unknown
        return 'unknown', self.clean_text(line)
    
    def parse_conversation(self, text: str) -> List[Utterance]:
        """
        Parse conversation text into structured utterances
        
        Args:
            text: Raw conversation text
            
        Returns:
            List of Utterance objects
        """
        utterances = []
        lines = text.split('\n')
        
        current_speaker = None
        current_text = []
        utterance_id = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            speaker, text = self.identify_speaker(line)
            
            # If we found a valid speaker
            if speaker in ['doctor', 'patient']:
                # Save previous utterance if exists
                if current_speaker and current_text:
                    combined_text = ' '.join(current_text)
                    utterances.append(Utterance(
                        speaker=current_speaker,
                        text=combined_text,
                        original_text=combined_text,
                        utterance_id=utterance_id
                    ))
                    utterance_id += 1
                
                # Start new utterance
                current_speaker = speaker
                current_text = [text] if text else []
            
            elif current_speaker:
                # Continue current utterance
                if text:
                    current_text.append(text)
        
        # Add final utterance
        if current_speaker and current_text:
            combined_text = ' '.join(current_text)
            utterances.append(Utterance(
                speaker=current_speaker,
                text=combined_text,
                original_text=combined_text,
                utterance_id=utterance_id
            ))
        
        return utterances
    
    def extract_temporal_info(self, text: str) -> List[Dict]:
        """
        Extract temporal information (dates, durations) from text
        
        Args:
            text: Input text
            
        Returns:
            List of temporal entities
        """
        doc = self.nlp(text)
        temporal_entities = []
        
        # spaCy DATE entities
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                temporal_entities.append({
                    'text': ent.text,
                    'type': 'date',
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Custom patterns for durations
        duration_patterns = [
            r'(\d+)\s*(week|month|year|day|hour)s?',
            r'(first|second|third)\s*(week|month|year)',
        ]
        
        for pattern in duration_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                temporal_entities.append({
                    'text': match.group(0),
                    'type': 'duration',
                    'start': match.start(),
                    'end': match.end()
                })
        
        return temporal_entities
    
    def segment_by_speaker(self, utterances: List[Utterance]) -> Dict[str, List[str]]:
        """
        Segment utterances by speaker
        
        Args:
            utterances: List of Utterance objects
            
        Returns:
            Dictionary with 'doctor' and 'patient' keys containing their utterances
        """
        segmented = {
            'doctor': [],
            'patient': []
        }
        
        for utt in utterances:
            if utt.speaker in segmented:
                segmented[utt.speaker].append(utt.text)
        
        return segmented
    
    def preprocess(self, conversation_text: str) -> Dict:
        """
        Main preprocessing pipeline
        
        Args:
            conversation_text: Raw conversation text
            
        Returns:
            Dictionary containing:
                - utterances: List of Utterance objects
                - patient_text: Combined patient utterances
                - doctor_text: Combined doctor utterances
                - full_text: Combined all text
                - temporal_info: Extracted temporal entities
        """
        # Parse conversation
        utterances = self.parse_conversation(conversation_text)
        
        # Segment by speaker
        segmented = self.segment_by_speaker(utterances)
        
        # Combine texts
        patient_text = ' '.join(segmented['patient'])
        doctor_text = ' '.join(segmented['doctor'])
        full_text = ' '.join([utt.text for utt in utterances])
        
        # Extract temporal information
        temporal_info = self.extract_temporal_info(full_text)
        
        return {
            'utterances': utterances,
            'patient_text': patient_text,
            'doctor_text': doctor_text,
            'full_text': full_text,
            'segmented': segmented,
            'temporal_info': temporal_info,
            'num_patient_utterances': len(segmented['patient']),
            'num_doctor_utterances': len(segmented['doctor'])
        }


# Example usage
if __name__ == "__main__":
    sample_text = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September. Can you walk me through what happened?
    Patient: Yes, it was on September 1st, around 12:30 in the afternoon.
    """
    
    preprocessor = MedicalConversationPreprocessor()
    result = preprocessor.preprocess(sample_text)
    
    print("=== Preprocessing Results ===")
    print(f"Total utterances: {len(result['utterances'])}")
    print(f"Patient utterances: {result['num_patient_utterances']}")
    print(f"Doctor utterances: {result['num_doctor_utterances']}")
    print(f"\nTemporal entities found: {len(result['temporal_info'])}")
    for temp in result['temporal_info']:
        print(f"  - {temp['text']} ({temp['type']})")
