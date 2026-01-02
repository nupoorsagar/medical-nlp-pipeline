"""
Medical NLP Pipeline Orchestrator
Coordinates all modules: Preprocessing, NER, Sentiment/Intent Analysis, SOAP Generation
"""

import json
import time
from typing import Dict, Optional
from pathlib import Path

from preprocessor import MedicalConversationPreprocessor
from medical_ner import MedicalNER
from sentiment_intent import SentimentIntentAnalyzer
from soap_generator import SOAPGenerator


class MedicalNLPPipeline:
    """
    Complete NLP pipeline for medical conversation analysis
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize pipeline with all components
        
        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        
        if self.verbose:
            print("Initializing Medical NLP Pipeline...")
            print("-" * 60)
        
        # Initialize components
        if self.verbose:
            print("Loading Preprocessor...")
        self.preprocessor = MedicalConversationPreprocessor()
        
        if self.verbose:
            print("Loading Medical NER...")
        self.ner = MedicalNER()
        
        if self.verbose:
            print("Loading Sentiment & Intent Analyzer...")
        self.sentiment_intent = SentimentIntentAnalyzer()
        
        if self.verbose:
            print("Loading SOAP Generator...")
        self.soap_generator = SOAPGenerator()
        
        if self.verbose:
            print("-" * 60)
            print("Pipeline initialized successfully!")
            print()
    
    def process_conversation(self, 
                           conversation_text: str,
                           generate_soap: bool = True) -> Dict:
        """
        Process a medical conversation through the complete pipeline
        
        Args:
            conversation_text: Raw conversation text
            generate_soap: Whether to generate SOAP note (optional)
            
        Returns:
            Dictionary containing all analysis results
        """
        results = {}
        
        # Step 1: Preprocessing
        if self.verbose:
            print("Step 1/4: Preprocessing conversation...")
        start_time = time.time()
        
        preprocessed = self.preprocessor.preprocess(conversation_text)
        results['preprocessed'] = {
            'num_utterances': len(preprocessed['utterances']),
            'num_patient_utterances': preprocessed['num_patient_utterances'],
            'num_doctor_utterances': preprocessed['num_doctor_utterances'],
            'temporal_entities': preprocessed['temporal_info']
        }
        
        if self.verbose:
            print(f"  ✓ Found {preprocessed['num_patient_utterances']} patient utterances")
            print(f"  ✓ Found {preprocessed['num_doctor_utterances']} doctor utterances")
            print(f"  ✓ Completed in {time.time() - start_time:.2f}s")
            print()
        
        # Step 2: Medical NER & Summarization
        if self.verbose:
            print("Step 2/4: Extracting medical entities...")
        start_time = time.time()
        
        medical_summary = self.ner.generate_summary(preprocessed)
        keywords = self.ner.extract_keywords(preprocessed['full_text'])
        
        results['medical_summary'] = medical_summary
        results['keywords'] = keywords
        
        if self.verbose:
            print(f"  ✓ Patient: {medical_summary['patient_name']}")
            print(f"  ✓ Symptoms: {len(medical_summary['symptoms'])} identified")
            print(f"  ✓ Diagnosis: {medical_summary['diagnosis']}")
            print(f"  ✓ Completed in {time.time() - start_time:.2f}s")
            print()
        
        # Step 3: Sentiment & Intent Analysis
        if self.verbose:
            print("Step 3/4: Analyzing sentiment and intent...")
        start_time = time.time()
        
        patient_utterances = preprocessed['segmented']['patient']
        utterance_analyses = self.sentiment_intent.analyze_patient_utterances(patient_utterances)
        aggregated_sentiment = self.sentiment_intent.aggregate_analysis(utterance_analyses)
        
        results['sentiment_intent'] = {
            'per_utterance': utterance_analyses,
            'aggregated': aggregated_sentiment
        }
        
        if self.verbose:
            print(f"  ✓ Overall Sentiment: {aggregated_sentiment['overall_sentiment']}")
            print(f"  ✓ Dominant Intents: {', '.join(aggregated_sentiment['dominant_intents'][:2])}")
            print(f"  ✓ Completed in {time.time() - start_time:.2f}s")
            print()
        
        # Step 4: SOAP Note Generation (optional)
        if generate_soap:
            if self.verbose:
                print("Step 4/4: Generating SOAP note...")
            start_time = time.time()
            
            soap_note = self.soap_generator.generate_soap_note(
                preprocessed,
                medical_summary,
                aggregated_sentiment,
                aggregated_sentiment
            )
            
            results['soap_note'] = soap_note
            
            if self.verbose:
                print(f"  ✓ SOAP note generated successfully")
                print(f"  ✓ Completed in {time.time() - start_time:.2f}s")
                print()
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save pipeline results to JSON file
        
        Args:
            results: Pipeline output dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any non-serializable objects
        serializable_results = json.loads(
            json.dumps(results, default=str)
        )
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict):
        """
        Print a formatted summary of results
        
        Args:
            results: Pipeline output dictionary
        """
        print("\n" + "=" * 70)
        print("MEDICAL NLP PIPELINE - RESULTS SUMMARY")
        print("=" * 70)
        
        # Medical Summary
        print("\n📋 MEDICAL SUMMARY:")
        print("-" * 70)
        ms = results['medical_summary']
        print(f"Patient: {ms['patient_name']}")
        print(f"Symptoms: {', '.join(ms['symptoms'][:5])}")
        print(f"Diagnosis: {', '.join(ms['diagnosis']) if ms['diagnosis'] else 'Not specified'}")
        print(f"Treatment: {', '.join(ms['treatment'][:3])}")
        print(f"Current Status: {ms['current_status']}")
        print(f"Prognosis: {ms['prognosis']}")
        
        # Sentiment & Intent
        print("\n💬 SENTIMENT & INTENT ANALYSIS:")
        print("-" * 70)
        si = results['sentiment_intent']['aggregated']
        print(f"Overall Sentiment: {si['overall_sentiment']}")
        print(f"Dominant Intents: {', '.join(si['dominant_intents'])}")
        print(f"Sentiment Distribution: {si['sentiment_distribution']}")
        
        # Keywords
        print("\n🔑 KEY MEDICAL TERMS:")
        print("-" * 70)
        print(f"{', '.join(results['keywords'][:10])}")
        
        # SOAP Note (if generated)
        if 'soap_note' in results:
            print("\n📄 SOAP NOTE:")
            print("-" * 70)
            soap_text = self.soap_generator.format_soap_note_text(results['soap_note'])
            print(soap_text)
        
        print("\n" + "=" * 70)


def main():
    """
    Main execution function with sample conversation
    """
    # Sample conversation from the document
    sample_conversation = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    
    Physician: I understand you were in a car accident last September. Can you walk me through what happened?
    
    Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
    
    Physician: That sounds like a strong impact. Were you wearing your seatbelt?
    
    Patient: Yes, I always do.
    
    Physician: What did you feel immediately after the accident?
    
    Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
    
    Physician: Did you seek medical attention at that time?
    
    Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.
    
    Physician: How did things progress after that?
    
    Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
    
    Physician: That makes sense. Are you still experiencing pain now?
    
    Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.
    
    Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
    
    Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.
    
    Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
    
    Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.
    
    Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.
    
    [Physical Examination Conducted]
    
    Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
    
    Patient: That's a relief!
    
    Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
    
    Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?
    
    Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.
    
    Patient: Thank you, doctor. I appreciate it.
    
    Physician: You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything.
    """
    
    # Initialize pipeline
    print("=" * 70)
    print("MEDICAL NLP PIPELINE - DEMONSTRATION")
    print("=" * 70)
    print()
    
    pipeline = MedicalNLPPipeline(verbose=True)
    
    # Process conversation
    results = pipeline.process_conversation(
        sample_conversation,
        generate_soap=True
    )
    
    # Print summary
    pipeline.print_summary(results)
    
    # Save results
    pipeline.save_results(results, "output/medical_analysis_results.json")
    
    print("\n✓ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
