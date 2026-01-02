"""
SOAP Note Generation Module
Generates structured SOAP (Subjective, Objective, Assessment, Plan) notes from medical conversations
"""

from typing import Dict, List
import re
from dataclasses import dataclass


@dataclass
class SOAPNote:
    """Structured SOAP note"""
    subjective: Dict
    objective: Dict
    assessment: Dict
    plan: Dict


class SOAPGenerator:
    """
    Generates SOAP notes from medical conversations
    Uses rule-based extraction + aggregation of NER/Sentiment/Intent outputs
    """
    
    def __init__(self):
        """Initialize SOAP generator with extraction patterns"""
        
        # Patterns for identifying SOAP sections
        self.subjective_indicators = [
            'patient', 'reports', 'complains', 'states', 'describes',
            'feels', 'experiencing', 'noticed', 'mentioned', 'history'
        ]
        
        self.objective_indicators = [
            'examination', 'physical exam', 'observed', 'found',
            'appears', 'shows', 'demonstrates', 'range of motion',
            'vital signs', 'measurement', 'palpation', 'inspection'
        ]
        
        self.assessment_indicators = [
            'diagnosis', 'diagnosed', 'assessed', 'impression',
            'condition', 'determined', 'concluded', 'evaluation'
        ]
        
        self.plan_indicators = [
            'plan', 'treatment', 'recommend', 'prescribe', 'follow-up',
            'continue', 'advise', 'schedule', 'return', 'monitor'
        ]
    
    def extract_chief_complaint(self, patient_utterances: List[str]) -> str:
        """
        Extract chief complaint from patient's first few utterances
        
        Args:
            patient_utterances: List of patient statements
            
        Returns:
            Chief complaint string
        """
        if not patient_utterances:
            return "Not specified"
        
        # Look in first 2-3 utterances for main complaint
        early_utterances = ' '.join(patient_utterances[:3])
        
        # Common complaint patterns
        complaint_patterns = [
            r'(pain|ache|discomfort|hurt|issue|problem|trouble)\s+(?:in|with|from)?\s*(\w+)',
            r'(had|have|experiencing)\s+([\w\s]+(?:pain|injury|accident|issue))',
        ]
        
        complaints = []
        for pattern in complaint_patterns:
            matches = re.finditer(pattern, early_utterances.lower())
            for match in matches:
                complaints.append(match.group(0))
        
        if complaints:
            # Return most specific complaint
            return max(complaints, key=len).strip().capitalize()
        
        # Fallback: return first meaningful utterance
        return patient_utterances[0][:100] if patient_utterances else "Not specified"
    
    def extract_history_of_present_illness(self, 
                                           patient_utterances: List[str],
                                           temporal_info: List[Dict]) -> str:
        """
        Extract history of present illness (HPI) from patient statements
        
        Args:
            patient_utterances: List of patient statements
            temporal_info: Temporal entities extracted from text
            
        Returns:
            HPI description
        """
        if not patient_utterances:
            return "No history provided"
        
        # Combine patient utterances into coherent history
        history_parts = []
        
        # Look for accident/injury description
        for utterance in patient_utterances:
            utterance_lower = utterance.lower()
            if any(word in utterance_lower for word in ['accident', 'injury', 'incident', 'happened']):
                history_parts.append(utterance)
        
        # Look for symptom progression
        for utterance in patient_utterances:
            utterance_lower = utterance.lower()
            if any(word in utterance_lower for word in ['weeks', 'months', 'days', 'first', 'started', 'began']):
                history_parts.append(utterance)
        
        # Look for treatment received
        for utterance in patient_utterances:
            utterance_lower = utterance.lower()
            if any(word in utterance_lower for word in ['treatment', 'therapy', 'medication', 'went to', 'visited']):
                history_parts.append(utterance)
        
        if history_parts:
            return ' '.join(history_parts)
        
        # Fallback: use all patient utterances
        return ' '.join(patient_utterances)
    
    def extract_physical_exam_findings(self, doctor_utterances: List[str]) -> str:
        """
        Extract physical examination findings from doctor's statements
        
        Args:
            doctor_utterances: List of doctor statements
            
        Returns:
            Physical exam findings
        """
        exam_findings = []
        
        for utterance in doctor_utterances:
            utterance_lower = utterance.lower()
            
            # Look for examination-related statements
            if any(indicator in utterance_lower for indicator in self.objective_indicators):
                exam_findings.append(utterance)
                continue
            
            # Look for specific findings
            if any(term in utterance_lower for term in [
                'range of motion', 'movement', 'tenderness', 'swelling',
                'normal', 'appears', 'shows', 'no signs', 'positive', 'negative'
            ]):
                exam_findings.append(utterance)
        
        if exam_findings:
            return ' '.join(exam_findings)
        
        return "Physical examination findings not documented in conversation"
    
    def extract_observations(self, doctor_utterances: List[str]) -> str:
        """
        Extract general observations about patient
        
        Args:
            doctor_utterances: List of doctor statements
            
        Returns:
            Observations string
        """
        observations = []
        
        for utterance in doctor_utterances:
            utterance_lower = utterance.lower()
            if any(term in utterance_lower for term in [
                'looks', 'appears', 'seems', 'presents', 'condition', 'gait', 'posture'
            ]):
                observations.append(utterance)
        
        if observations:
            return ' '.join(observations)
        
        return "Patient appears in stated condition"
    
    def build_subjective(self, 
                        patient_utterances: List[str],
                        temporal_info: List[Dict],
                        sentiment_data: Dict) -> Dict:
        """
        Build Subjective section of SOAP note
        
        Args:
            patient_utterances: Patient statements
            temporal_info: Temporal entities
            sentiment_data: Sentiment analysis results
            
        Returns:
            Subjective section dictionary
        """
        chief_complaint = self.extract_chief_complaint(patient_utterances)
        hpi = self.extract_history_of_present_illness(patient_utterances, temporal_info)
        
        # Extract current symptoms from patient statements
        current_symptoms = []
        for utterance in patient_utterances:
            if any(word in utterance.lower() for word in ['now', 'currently', 'still', 'today']):
                current_symptoms.append(utterance)
        
        return {
            'chief_complaint': chief_complaint,
            'history_of_present_illness': hpi,
            'current_symptoms': ' '.join(current_symptoms) if current_symptoms else "See HPI",
            'patient_sentiment': sentiment_data.get('overall_sentiment', 'Neutral')
        }
    
    def build_objective(self, 
                       doctor_utterances: List[str],
                       full_text: str) -> Dict:
        """
        Build Objective section of SOAP note
        
        Args:
            doctor_utterances: Doctor statements
            full_text: Full conversation text
            
        Returns:
            Objective section dictionary
        """
        physical_exam = self.extract_physical_exam_findings(doctor_utterances)
        observations = self.extract_observations(doctor_utterances)
        
        # Look for physical examination header
        exam_conducted = "physical examination" in full_text.lower() or \
                        "examination conducted" in full_text.lower()
        
        return {
            'physical_exam': physical_exam,
            'observations': observations,
            'vital_signs': "Not recorded in conversation",
            'exam_conducted': exam_conducted
        }
    
    def build_assessment(self,
                        medical_summary: Dict,
                        sentiment_data: Dict,
                        doctor_utterances: List[str]) -> Dict:
        """
        Build Assessment section of SOAP note
        
        Args:
            medical_summary: Medical NER summary
            sentiment_data: Sentiment analysis
            doctor_utterances: Doctor statements
            
        Returns:
            Assessment section dictionary
        """
        # Extract diagnosis from medical summary
        diagnosis_list = medical_summary.get('diagnosis', [])
        diagnosis = ', '.join(diagnosis_list) if diagnosis_list else "Pending further evaluation"
        
        # Determine severity from context
        severity = "Mild to moderate"
        for utterance in doctor_utterances:
            if any(word in utterance.lower() for word in ['severe', 'serious', 'critical']):
                severity = "Severe"
                break
            elif any(word in utterance.lower() for word in ['mild', 'minor', 'slight']):
                severity = "Mild"
                break
            elif any(word in utterance.lower() for word in ['improving', 'better', 'recovery']):
                severity = "Mild, improving"
                break
        
        return {
            'diagnosis': diagnosis,
            'severity': severity,
            'clinical_impression': medical_summary.get('prognosis', 'Clinical impression pending'),
            'symptoms_severity': sentiment_data.get('overall_sentiment', 'Neutral')
        }
    
    def build_plan(self,
                  medical_summary: Dict,
                  doctor_utterances: List[str],
                  intent_data: Dict) -> Dict:
        """
        Build Plan section of SOAP note
        
        Args:
            medical_summary: Medical NER summary
            doctor_utterances: Doctor statements
            intent_data: Intent analysis
            
        Returns:
            Plan section dictionary
        """
        # Extract treatments from medical summary
        treatments = medical_summary.get('treatment', [])
        treatment_plan = ', '.join(treatments) if treatments else "Continue current management"
        
        # Look for follow-up recommendations
        follow_up = "Follow up as needed"
        for utterance in doctor_utterances:
            if any(word in utterance.lower() for word in ['return', 'follow-up', 'come back', 'check']):
                follow_up = utterance
                break
        
        # Look for medication/therapy recommendations
        recommendations = []
        for utterance in doctor_utterances:
            if any(word in utterance.lower() for word in ['recommend', 'continue', 'should', 'advise']):
                recommendations.append(utterance)
        
        # Patient education based on intent
        patient_education = "Continue monitoring symptoms"
        if 'Seeking reassurance' in intent_data.get('dominant_intents', []):
            patient_education = "Reassurance provided regarding prognosis and recovery timeline"
        
        return {
            'treatment': treatment_plan,
            'medications': "See treatment plan",
            'follow_up': follow_up,
            'patient_education': patient_education,
            'recommendations': ' '.join(recommendations) if recommendations else "Continue current course"
        }
    
    def generate_soap_note(self,
                          preprocessed_data: Dict,
                          medical_summary: Dict,
                          sentiment_data: Dict,
                          intent_data: Dict) -> Dict:
        """
        Generate complete SOAP note from all pipeline outputs
        
        Args:
            preprocessed_data: Output from preprocessor
            medical_summary: Output from medical NER
            sentiment_data: Output from sentiment analysis
            intent_data: Output from intent detection
            
        Returns:
            Complete SOAP note in dictionary format
        """
        patient_utterances = preprocessed_data['segmented']['patient']
        doctor_utterances = preprocessed_data['segmented']['doctor']
        full_text = preprocessed_data['full_text']
        temporal_info = preprocessed_data['temporal_info']
        
        # Build each section
        subjective = self.build_subjective(patient_utterances, temporal_info, sentiment_data)
        objective = self.build_objective(doctor_utterances, full_text)
        assessment = self.build_assessment(medical_summary, sentiment_data, doctor_utterances)
        plan = self.build_plan(medical_summary, doctor_utterances, intent_data)
        
        soap_note = {
            'patient_name': medical_summary.get('patient_name', 'Unknown'),
            'subjective': subjective,
            'objective': objective,
            'assessment': assessment,
            'plan': plan,
            'metadata': {
                'generated_from': 'conversation_transcript',
                'num_patient_utterances': len(patient_utterances),
                'num_doctor_utterances': len(doctor_utterances),
                'overall_sentiment': sentiment_data.get('overall_sentiment', 'Neutral'),
                'dominant_intents': intent_data.get('dominant_intents', [])
            }
        }
        
        return soap_note
    
    def format_soap_note_text(self, soap_note: Dict) -> str:
        """
        Format SOAP note as readable text
        
        Args:
            soap_note: SOAP note dictionary
            
        Returns:
            Formatted text
        """
        text = f"""
SOAP NOTE - {soap_note['patient_name']}
{'=' * 60}

SUBJECTIVE:
-----------
Chief Complaint: {soap_note['subjective']['chief_complaint']}

History of Present Illness:
{soap_note['subjective']['history_of_present_illness']}

Current Symptoms: {soap_note['subjective']['current_symptoms']}
Patient Sentiment: {soap_note['subjective']['patient_sentiment']}

OBJECTIVE:
----------
Physical Examination:
{soap_note['objective']['physical_exam']}

Observations: {soap_note['objective']['observations']}
Vital Signs: {soap_note['objective']['vital_signs']}

ASSESSMENT:
-----------
Diagnosis: {soap_note['assessment']['diagnosis']}
Severity: {soap_note['assessment']['severity']}
Clinical Impression: {soap_note['assessment']['clinical_impression']}

PLAN:
-----
Treatment: {soap_note['plan']['treatment']}
Medications: {soap_note['plan']['medications']}
Follow-up: {soap_note['plan']['follow_up']}
Patient Education: {soap_note['plan']['patient_education']}
Recommendations: {soap_note['plan']['recommendations']}

{'=' * 60}
        """
        return text.strip()


# Example usage
if __name__ == "__main__":
    # Sample data (would come from previous modules)
    sample_preprocessed = {
        'segmented': {
            'patient': [
                "I'm doing better, but I still have some discomfort now and then.",
                "It was on September 1st. I had to stop in traffic when another car hit me.",
                "I had neck and back pain for four weeks. It was really bad.",
                "Yes, I had ten physiotherapy sessions.",
                "Now I only get occasional backaches."
            ],
            'doctor': [
                "How are you feeling today?",
                "Can you walk me through what happened?",
                "Did you seek medical attention?",
                "Everything looks good. Full range of movement, no tenderness.",
                "I expect full recovery within six months."
            ]
        },
        'full_text': "Combined conversation text...",
        'temporal_info': [{'text': 'September 1st', 'type': 'date'}]
    }
    
    sample_medical_summary = {
        'patient_name': 'Ms. Jones',
        'symptoms': ['Neck pain', 'Back pain'],
        'diagnosis': ['Whiplash injury'],
        'treatment': ['10 physiotherapy sessions', 'Painkillers'],
        'current_status': 'Occasional backache',
        'prognosis': 'Full recovery expected within six months'
    }
    
    sample_sentiment = {
        'overall_sentiment': 'Reassured',
        'sentiment_distribution': {'Reassured': 3, 'Anxious': 1, 'Neutral': 1}
    }
    
    sample_intent = {
        'dominant_intents': ['Reporting symptoms', 'Seeking reassurance', 'Describing improvement']
    }
    
    # Generate SOAP note
    generator = SOAPGenerator()
    soap_note = generator.generate_soap_note(
        sample_preprocessed,
        sample_medical_summary,
        sample_sentiment,
        sample_intent
    )
    
    print(generator.format_soap_note_text(soap_note))
