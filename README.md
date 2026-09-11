# Movie Review Sentiment Analysis with DistilBERT

An AI-powered sentiment analysis project that classifies movie reviews as **positive** or **negative**.

The project uses DistilBERT and the Hugging Face Transformers library. A pre-trained sentiment analysis model was further fine-tuned using a custom dataset containing informal movie reviews, slang and contemporary language.

The aim of the project is to investigate whether fine-tuning an existing transformer model on domain-specific data can improve its ability to understand informal expressions such as *"meh"*, *"mid"*, *"slaps"* and other modern language that may appear in movie reviews.

---

## Features

- **DistilBERT Transformer Model:** Uses a lightweight version of BERT designed to retain much of BERT's language understanding while requiring fewer computational resources.
- **Pre-trained Sentiment Model:** Uses `distilbert-base-uncased-finetuned-sst-2-english` as the starting model.
- **Custom Fine-tuning:** The model is further trained using a custom dataset of movie reviews containing informal language and slang.
- **Binary Sentiment Classification:** Reviews are classified as either `POSITIVE` or `NEGATIVE`.
- **Confidence Scores:** Each prediction includes a confidence score indicating how confident the model is in its classification.
- **Interactive Testing:** Users can enter their own movie reviews through a command-line interface.
- **Model Evaluation:** Accuracy and other classification metrics can be used to compare model performance before and after fine-tuning.

---

## Project Structure

```text
sentiment-analysis/
├── fine_tuned_sentiment_model/      # Saved fine-tuned model and tokenizer
├── sentiment_analyser.py            # Original sentiment analyser
├── sentiment_analyser-Finetuned.py  # Fine-tuned model interface
├── slang_reviews.csv                 # Custom slang review dataset
├── README.md                         # Project documentation
└── venv39/                           # Local Python virtual environment
```

> **Note:** The virtual environment (`venv39`) should normally be excluded from GitHub using a `.gitignore` file.

---

## Model Information

| Property | Value |
| --- | --- |
| Base Model | `distilbert-base-uncased-finetuned-sst-2-english` |
| Architecture | DistilBERT |
| Task | Binary Sentiment Classification |
| Labels | POSITIVE / NEGATIVE |
| Maximum Token Length | 512 |
| Framework | Hugging Face Transformers |
| Backend | PyTorch |
| Fine-tuning Data | Custom movie review/slang dataset |

---

## Dataset

The project uses a custom dataset containing movie review text and corresponding sentiment labels.

The dataset contains two main fields:

| Column | Description |
| --- | --- |
| `text` | Movie review text |
| `label` | Sentiment classification (`0` = negative, `1` = positive) |

The custom dataset includes informal expressions and modern slang to expose the model to language that may not be represented as strongly in traditional sentiment datasets.

Examples of the type of language considered include:

```text
"The movie was meh"
"This film absolutely slaps"
"That was so mid"
"I loved this movie"
"That film was awful"
```

The dataset is split into **training and test sets**, allowing the fine-tuned model to be evaluated on data that was not used directly during training.

---

## Fine-tuning

The project begins with the Hugging Face model:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

This model has already been trained for sentiment classification using the **SST-2 (Stanford Sentiment Treebank)** dataset.

Rather than training a transformer from scratch, **transfer learning** is used. The existing model is further fine-tuned using the custom movie review dataset.

The fine-tuning process includes:

1. Loading the custom dataset.
2. Splitting the data into training and test datasets.
3. Tokenising reviews using the DistilBERT tokenizer.
4. Training the model on the custom data.
5. Evaluating the model against the test dataset.
6. Saving the fine-tuned model and tokenizer for later use.

The resulting model is stored in:

```text
fine_tuned_sentiment_model/
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd sentiment-analysis
```

Replace `<repository-url>` with the URL of this project's GitHub repository.

### 2. Create a Virtual Environment

For a standard Python installation:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```text
venv\Scripts\activate
```

### Intel Mac Compatibility

This project was also tested on an Intel-based Mac using Python 3.9.

The virtual environment can be created with:

```bash
/usr/bin/python3 -m venv venv39
source venv39/bin/activate
```

### 3. Install Dependencies

```bash
pip install transformers torch pandas numpy scikit-learn
```

Depending on the versions of Python and PyTorch being used, compatible package versions may be required.

---

## Usage

### Original Pre-trained Model

Run:

```bash
python sentiment_analyser.py
```

This uses the original pre-trained DistilBERT sentiment model.

### Fine-tuned Model

Run:

```bash
python sentiment_analyser-Finetuned.py
```

The fine-tuned version loads the locally saved model from:

```text
fine_tuned_sentiment_model/
```

The program then prompts the user to enter a movie review.

For example:

```text
==================================================
Custom Sentiment Analysis
==================================================
Enter movie reviews to analyze (or 'quit' to exit)

Enter review: This film was absolutely brilliant

Sentiment: POSITIVE
Confidence: 99.42%
```

Enter:

```text
quit
```

to exit the application.

---

## Loading the Fine-tuned Model

The fine-tuned model and tokenizer can be loaded using Hugging Face Transformers:

```python
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

model_path = "./fine_tuned_sentiment_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# DistilBERT uses input IDs and attention masks
tokenizer.model_input_names = ["input_ids", "attention_mask"]

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)
```

Using a relative path (`./fine_tuned_sentiment_model`) makes the project more portable between different computers than using a machine-specific absolute path.

---

## Evaluation

The model can be evaluated using metrics such as:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion matrix**

These metrics can be used to compare the original pre-trained model with the fine-tuned model and determine whether training on the custom slang dataset improves performance on informal movie reviews.

A key area of interest is how the model handles ambiguous or slang-based phrases such as:

```text
"The film was meh"
```

Misclassifications are useful during evaluation because they can highlight weaknesses in the training data and identify areas where additional examples may be required.

---

## Handling Long Reviews

DistilBERT supports a maximum sequence length of **512 tokens**.

Reviews longer than this need to be truncated or divided into smaller sections before being processed.

Possible future approaches include:

**Sliding Window**

- Divide long reviews into smaller chunks.
- Analyse each chunk separately.
- Combine the predictions or confidence scores.

**Smart Truncation**

- Preserve important sections from the beginning and end of a review.
- Select the most relevant sentences before classification.

**Alternative Models**

Models designed for longer sequences, such as Longformer or BigBird, could also be investigated.

---

## Limitations

The current project has several limitations:

- Sentiment is limited to two classes: positive and negative.
- Neutral or mixed opinions are not represented as a separate class.
- The custom fine-tuning dataset is relatively small.
- Slang can be highly context-dependent and can change over time.
- Short or ambiguous statements such as *"the film was meh"* may still be misclassified.
- DistilBERT has a maximum sequence length of 512 tokens.
- Model predictions represent statistical classifications and may not always reflect the intended meaning of a review.

---

## Future Improvements

Potential improvements include:

- Expand the custom training dataset.
- Add more examples of slang and contemporary language.
- Introduce a `NEUTRAL` sentiment category.
- Compare the original and fine-tuned models systematically.
- Improve handling of ambiguous expressions.
- Implement a Streamlit web interface.
- Add support for longer movie reviews.
- Investigate multilingual sentiment analysis.
- Add aspect-based sentiment analysis for areas such as acting, plot, dialogue and cinematography.
- Evaluate alternative transformer architectures.

---

## Requirements

Core Python dependencies include:

```text
transformers
torch
datasets
scikit-learn
pandas
numpy
```

If a Streamlit interface is added:

```text
streamlit
```

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Pandas
- NumPy
- Scikit-learn
- Google Colab
- Visual Studio Code

---

## Conclusion

This project demonstrates how **transfer learning and fine-tuning** can be applied to a practical Natural Language Processing task.

A pre-trained DistilBERT sentiment classifier provides the initial language understanding, while additional fine-tuning on a custom movie review dataset allows the project to investigate how domain-specific training data affects the model's understanding of informal and contemporary language.

The project also demonstrates important stages of a machine-learning workflow, including data preparation, tokenisation, model fine-tuning, evaluation, model persistence and interactive inference.

---

## Author

Claudia

---

## Licence

This project was developed for educational purposes.