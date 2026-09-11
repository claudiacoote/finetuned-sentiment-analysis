from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

model_path = "/Users/claudia/Documents/sentiment-analysis/fine_tuned_sentiment_model"

print("Loading fine-tuned model...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# DistilBERT only needs these two inputs
tokenizer.model_input_names = ["input_ids", "attention_mask"]

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

print("Model loaded successfully!\n")


def analyze_custom_text():
    print("\n" + "=" * 50)
    print("Custom Sentiment Analysis")
    print("=" * 50)
    print("Enter movie reviews to analyze (or 'quit' to exit)\n")

    while True:
        user_input = input("Enter review: ").strip()

        if user_input.lower() == "quit":
            break

        if user_input:
            result = classifier(user_input)[0]

            print(f"Sentiment: {result['label']}")
            print(f"Confidence: {result['score']:.2%}\n")


analyze_custom_text()