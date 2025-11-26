# NLP-PROJECT-1
🚀 Project Overview

The goal of this project is to fine-tune a pre-trained Transformer model for the NLP task of question classification. Using Hugging Face Transformers, the model learns to categorize questions into predefined classes.

📂 Dataset

Dataset Used: TREC Question Classification Dataset

Contains open-domain questions categorized into 6 coarse classes (e.g., entity, location, number).

Includes thousands of labeled question samples.

Split used:

Train: 80%

Validation: 10%

Test: 10%

🧹 Data Preprocessing

The notebook applies the following preprocessing steps:

Lowercasing text

Basic punctuation removal

Removing extra whitespace

Tokenization using model-specific tokenizer (e.g., BERT tokenizer)

Padding/truncation to a maximum sequence length (128 tokens)

🧠 Model Details

Pre-trained Base Model:

bert-base-uncased (default)

Configurable: can also use distilbert-base-uncased, roberta-base, etc.

Task: Sequence classification Model Components:

Transformer encoder

Classification head with num_labels outputs

⚙️ Training Setup

Training performed using Hugging Face Trainer API:

Parameter Value Learning Rate 2e-5 Batch Size 16 (train), 32 (eval) Epochs 3 Weight Decay 0.01 Evaluation Strategy Per Epoch Best Model Selection By F1-score Mixed Precision Enabled (if GPU supports fp16)

Libraries Used:

Transformers

Datasets

Evaluate

PyTorch

scikit-learn

Matplotlib / Seaborn

📊 Evaluation Metrics

The model is evaluated using:

Accuracy

Precision

Recall

F1-score (macro)

Confusion matrix (visualization included)

A detailed classification report is generated using scikit-learn.

💾 Saved Model

The fine-tuned model and tokenizer are saved to:

/content/finetuned-trec-bert

This can be reloaded for inference or deployment.

🌐 Optional: Streamlit App

A lightweight web app is included for inference.

Features:

Text input box

Preprocessing applied automatically

Real-time label prediction

Probability distribution shown for each label

To run in Colab:

streamlit run app.py

(ngrok tunneling instructions included in the notebook)

📁 Repository Structure (Suggested) |-- FineTuneLLM.ipynb # Google Colab notebook |-- app.py # Streamlit demo (optional) |-- README.md # Project documentation |-- /finetuned-trec-bert # Saved model directory

📝 Report Summary (Short)

Dataset: TREC Question Classification

Model: BERT-base (fine-tuned)

Training: 3 epochs, AdamW optimizer

Evaluation: Macro F1, accuracy, precision, recall

Observations:

Model converges quickly

Larger models (RoBERTa) may improve performance

Preprocessing affects classification performance

Challenges:

Small dataset size

Class imbalance in some question types

🔮 Future Improvements

Hyperparameter tuning

Experiment with RoBERTa / DistilBERT

Add data augmentation

Train longer for better performance

Deploy via Flask or Hugging Face Spaces
