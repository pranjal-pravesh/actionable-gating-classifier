# MobileBERT Fine-Tuning for Gating Classification

## Project Overview
This project implements a gating classifier using MobileBERT, a lightweight BERT model optimized for mobile devices. The classifier determines whether a given text is "actionable" (requiring a response or action) or "non-actionable" (informational or non-directive).

## Problem Statement
Natural language processing systems often need to determine if a user's input requires an action. This binary classification task helps in building more responsive conversational agents and command systems by filtering actionable requests from general statements.

## Dataset
The dataset consists of labeled text examples in two categories:
- **Actionable (1)**: Texts that require actions or responses (e.g., commands, questions, requests)
- **Non-actionable (0)**: Texts that are merely informational (e.g., statements, observations)

The dataset is balanced with:
- 691 actionable examples
- 670 non-actionable examples

Data sources include multiple Gemini-generated examples stored in the `dataset/` directory.

## Project Structure
- `fine_tuning.ipynb`: Jupyter notebook containing the model training code
- `data_processing.ipynb`: Notebook for preprocessing and preparing the dataset
- `final_dataset.csv`: The processed dataset used for training
- `dataset/`: Directory containing the raw data files
- `mobilebert-finetuned-actionable/`: (Generated after training) Directory containing the trained model

## Implementation
The implementation uses Hugging Face's Transformers library to fine-tune the MobileBERT model. The process includes:

1. Data preprocessing and preparation
2. Loading the pre-trained MobileBERT model
3. Fine-tuning on the actionable/non-actionable classification task
4. Evaluation of model performance
5. Saving the model for future use

## Results
The fine-tuned model achieved excellent performance metrics:

- **Accuracy**: 97.3%
- **F1 Score**: 97.3%  
- **Precision**: 96.1%
- **Recall**: 98.5%

These metrics indicate that the model is highly effective at distinguishing between actionable and non-actionable text.

## Usage
### Requirements
- Python 3.7+
- PyTorch
- Transformers
- Pandas
- NumPy
- Wandb (for tracking experiments)

### Training
The model can be retrained using the `fine_tuning.ipynb` notebook.

### Inference
To use the trained model for predictions:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./mobilebert-finetuned-actionable")
model = AutoModelForSequenceClassification.from_pretrained("./mobilebert-finetuned-actionable")

# Example function for inference
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    
    return {
        "text": text,
        "prediction": "Actionable" if prediction == 1 else "Non-actionable",
        "confidence": probabilities[0][prediction].item()
    }

# Example usage
result = classify_text("Please set an alarm for 7 AM tomorrow")
print(result)
```

## Future Work
- Expand the dataset with more diverse examples
- Experiment with other lightweight models like DistilBERT
- Deploy the model as an API or mobile application
- Implement multi-class classification for different types of actionable requests

## License
This project is available under the MIT License. 