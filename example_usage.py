"""
Example usage of the Medical NLP Pipeline
Demonstrates various ways to use the pipeline components
"""

from pipeline import MedicalNLPPipeline
from preprocessor import MedicalConversationPreprocessor
from medical_ner import MedicalNER
from sentiment_intent import SentimentIntentAnalyzer
from soap_generator import SOAPGenerator
import json


# Sample conversations for testing
SAMPLE_CONVERSATIONS = {
    "car_accident": """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September. Can you walk me through what happened?
    Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind.
    Physician: That sounds like a strong impact. Were you wearing your seatbelt?
    Patient: Yes, I always do.
    Physician: What did you feel immediately after the accident?
    Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
    Physician: Did you seek medical attention at that time?
    Patient: Yes, I went to Moss Bank Accident and Emergency. They said it was a whiplash injury, but they didn't do any X-rays.
    Physician: How did things progress after that?
    Patient: The first four weeks were rough. My neck and back pain were really bad. I had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy.
    Physician: Are you still experiencing pain now?
    Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.
    Physician: Let's do a physical examination to check your mobility.
    [Physical Examination Conducted]
    Physician: Everything looks good. Your neck and back have a full range of movement. I expect you to make a full recovery within six months.
    Patient: That's a relief! Thank you, doctor.
    """,
    
    "follow_up": """
    Doctor: Hello Mr. Smith, how have you been since our last visit?
    Patient: Much better, doctor. The medication has helped a lot.
    Doctor: That's great to hear. Any side effects?
    Patient: No, nothing at all. I'm sleeping better too.
    Doctor: Excellent. Let's continue with the current treatment plan.
    Patient: Sounds good. When should I come back?
    Doctor: Let's schedule a follow-up in three months.
    """,
    
    "new_symptoms": """
    Physician: What brings you in today?
    Patient: I've been having these terrible headaches for the past week. They're really bad in the morning.
    Physician: On a scale of 1 to 10, how would you rate the pain?
    Patient: Maybe a 7 or 8. It's pretty intense.
    Physician: Any other symptoms? Nausea, vision changes?
    Patient: Yes, I feel a bit nauseous sometimes, and my vision gets blurry.
    Physician: I'd like to run some tests. This could be migraines, but we need to rule out other causes.
    Patient: I'm a bit worried. Is it serious?
    Physician: We'll get to the bottom of it. Most cases are treatable once we identify the cause.
    """
}


def example_1_complete_pipeline():
    """Example 1: Run complete pipeline on sample conversation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Complete Pipeline Execution")
    print("=" * 80 + "\n")
    
    # Initialize pipeline
    pipeline = MedicalNLPPipeline(verbose=True)
    
    # Process conversation
    results = pipeline.process_conversation(
        SAMPLE_CONVERSATIONS["car_accident"],
        generate_soap=True
    )
    
    # Print summary
    pipeline.print_summary(results)
    
    # Save results
    pipeline.save_results(results, "output/example1_results.json")
    print("\n✓ Results saved to output/example1_results.json")


def example_2_individual_modules():
    """Example 2: Use individual modules separately"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Individual Module Usage")
    print("=" * 80 + "\n")
    
    conversation = SAMPLE_CONVERSATIONS["new_symptoms"]
    
    # Step 1: Preprocessing
    print("Step 1: Preprocessing...")
    preprocessor = MedicalConversationPreprocessor()
    preprocessed = preprocessor.preprocess(conversation)
    print(f"  ✓ Extracted {preprocessed['num_patient_utterances']} patient utterances")
    print(f"  ✓ Extracted {preprocessed['num_doctor_utterances']} doctor utterances")
    
    # Step 2: Medical NER
    print("\nStep 2: Medical Entity Extraction...")
    ner = MedicalNER()
    summary = ner.generate_summary(preprocessed)
    print(f"  ✓ Symptoms found: {summary['symptoms']}")
    print(f"  ✓ Diagnosis: {summary['diagnosis']}")
    
    # Step 3: Sentiment Analysis
    print("\nStep 3: Sentiment & Intent Analysis...")
    analyzer = SentimentIntentAnalyzer()
    patient_utterances = preprocessed['segmented']['patient']
    analyses = analyzer.analyze_patient_utterances(patient_utterances)
    aggregated = analyzer.aggregate_analysis(analyses)
    print(f"  ✓ Overall Sentiment: {aggregated['overall_sentiment']}")
    print(f"  ✓ Dominant Intents: {', '.join(aggregated['dominant_intents'][:2])}")
    
    # Step 4: Keywords
    print("\nStep 4: Keyword Extraction...")
    keywords = ner.extract_keywords(preprocessed['full_text'], top_n=5)
    print(f"  ✓ Top keywords: {', '.join(keywords)}")


def example_3_batch_processing():
    """Example 3: Process multiple conversations"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Batch Processing Multiple Conversations")
    print("=" * 80 + "\n")
    
    pipeline = MedicalNLPPipeline(verbose=False)  # Disable verbose for batch
    
    batch_results = {}
    
    for name, conversation in SAMPLE_CONVERSATIONS.items():
        print(f"Processing: {name}...")
        results = pipeline.process_conversation(conversation, generate_soap=False)
        batch_results[name] = {
            'patient_name': results['medical_summary']['patient_name'],
            'symptoms': results['medical_summary']['symptoms'],
            'sentiment': results['sentiment_intent']['aggregated']['overall_sentiment'],
            'num_utterances': results['preprocessed']['num_utterances']
        }
        print(f"  ✓ Completed")
    
    # Save batch results
    with open('output/batch_results.json', 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print("\n✓ Batch processing complete. Results saved to output/batch_results.json")
    
    # Print summary table
    print("\n" + "-" * 80)
    print(f"{'Conversation':<20} {'Patient':<15} {'Sentiment':<12} {'Symptoms'}")
    print("-" * 80)
    for name, data in batch_results.items():
        symptoms_str = ', '.join(data['symptoms'][:2])
        print(f"{name:<20} {data['patient_name']:<15} {data['sentiment']:<12} {symptoms_str}")
    print("-" * 80)


def example_4_custom_analysis():
    """Example 4: Custom analysis with specific focus"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Analysis - Sentiment Focus")
    print("=" * 80 + "\n")
    
    conversation = SAMPLE_CONVERSATIONS["new_symptoms"]
    
    # Preprocess
    preprocessor = MedicalConversationPreprocessor()
    preprocessed = preprocessor.preprocess(conversation)
    
    # Focus on sentiment analysis
    analyzer = SentimentIntentAnalyzer()
    patient_utterances = preprocessed['segmented']['patient']
    
    print("Per-Utterance Sentiment Analysis:")
    print("-" * 80)
    
    for i, utterance in enumerate(patient_utterances):
        sentiment = analyzer.analyze_sentiment(utterance)
        intent = analyzer.detect_intent(utterance, top_k=1)
        
        print(f"\nUtterance {i+1}:")
        print(f"  Text: {utterance[:60]}...")
        print(f"  Sentiment: {sentiment['sentiment']} (confidence: {sentiment['confidence']:.2f})")
        print(f"  Intent: {intent['intents'][0]}")


def example_5_soap_only():
    """Example 5: Generate SOAP note only (skip NER/Sentiment if already done)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Direct SOAP Note Generation")
    print("=" * 80 + "\n")
    
    conversation = SAMPLE_CONVERSATIONS["follow_up"]
    
    # Run pipeline to get all components
    pipeline = MedicalNLPPipeline(verbose=False)
    results = pipeline.process_conversation(conversation, generate_soap=True)
    
    # Print only SOAP note
    soap_text = pipeline.soap_generator.format_soap_note_text(results['soap_note'])
    print(soap_text)


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("MEDICAL NLP PIPELINE - EXAMPLE USAGE DEMONSTRATIONS")
    print("=" * 80)
    
    examples = [
        ("Complete Pipeline", example_1_complete_pipeline),
        ("Individual Modules", example_2_individual_modules),
        ("Batch Processing", example_3_batch_processing),
        ("Custom Sentiment Analysis", example_4_custom_analysis),
        ("SOAP Note Only", example_5_soap_only)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples) + 1}. Run all examples")
    print("  0. Exit")
    
    try:
        choice = input("\nSelect an example (0-6): ").strip()
        choice = int(choice)
        
        if choice == 0:
            print("Exiting...")
            return
        elif choice == len(examples) + 1:
            # Run all examples
            for name, func in examples:
                func()
        elif 1 <= choice <= len(examples):
            # Run selected example
            examples[choice - 1][1]()
        else:
            print("Invalid choice. Please select a number between 0 and 6.")
    
    except ValueError:
        print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
