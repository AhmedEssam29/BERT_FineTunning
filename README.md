# Fine-Tuning BERT for Text Classification
## By/Ahmed Essam 

This repository contains the code and methodology for fine-tuning the `bert-base-uncased` model on a custom dataset for text classification. The dataset consists of tweets related to medical and pharmaceutical topics, and the goal is to classify them into two categories.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

## Overview
The project fine-tunes the `bert-base-uncased` model using TensorFlow and the Hugging Face Transformers library. The model is trained on a dataset of tweets, with the objective of classifying them into one of two categories. The fine-tuning process involves:
- Tokenizing the text
- Building a custom BERT-based model
- Evaluating its performance using precision, recall, and F1-score

## Dataset
The dataset consists of **23,514 tweets** related to medical and pharmaceutical topics. Each tweet is labeled with a binary classification (`0` or `1`).

### Dataset Structure

| Column     | Description                          |
|------------|--------------------------------------|
| **Tweet**  | The text of the tweet.              |
| **WordCount** | The number of words in the tweet. |

### Dataset Example

| Tweet | WordCount |
|-----------------------------------------|------------|
| Intravenous azithromycin-induced ototoxicity. | 3 |
| Immobilization, while Paget's bone disease was present, and perhaps enhanced activation of... | 24 |
| Unaccountable severe hypercalcemia in a patient treated for hypoparathyroidism with... | 11 |

![alt text](https://github.com/AhmedEssam29/BERT_FineTunning/blob/main/output.png?raw=true)

## Model Architecture
The model is built using the `bert-base-uncased` model from Hugging Face's Transformers library. The architecture is as follows:

- **Input Layer**: Tokenized input (`input IDs` and `attention masks`) with a max sequence length of **100**.
- **BERT Layer**: Extracts features using the pre-trained BERT model.
- **Pooling Layer**: Applies global max pooling to the BERT output.
- **Dense Layers**: Two fully connected layers with ReLU activation and dropout for regularization.
- **Output Layer**: A softmax layer for binary classification.

### Model Summary
```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_ids (InputLayer)       [(None, 100)]             0         
attention_mask (InputLayer)  [(None, 100)]             0         
bert_layer (BertLayer)       (None, 100, 768)          110M      
_________________________________________________________________
global_max_pooling1d (Global (None, 768)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               196864    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 258       
=================================================================
Total params: 110,230,018
Trainable params: 110,230,018
Non-trainable params: 0
_________________________________________________________________
```

## Training
The model was trained using the following configuration:

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Categorical Accuracy
- **Batch Size**: 32
- **Epochs**: 10
- **Early Stopping**: Enabled to prevent overfitting.

## Results
The model achieved the following performance on the test set:

| Metric      | Class 0 | Class 1 | Weighted Avg |
|------------|--------|--------|--------------|
| **Precision**  | 0.72    | 0.47    | 0.65         |
| **Recall**     | 0.98    | 0.05    | 0.71         |
| **F1-Score**   | 0.83    | 0.09    | 0.62         |

### Confusion Matrix
```
Class 0: Precision = 0.72, Recall = 0.98, F1-Score = 0.83
Class 1: Precision = 0.47, Recall = 0.05, F1-Score = 0.09
```

## Requirements
To run this project, install the following Python libraries:

```
pandas>=1.3.0
seaborn>=0.11.0
matplotlib>=3.4.0
numpy>=1.21.0
wordcloud>=1.8.0
tensorflow>=2.6.0
scikit-learn>=0.24.0
transformers>=4.0.0
keras>=2.6.0
```

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
### Clone the repository:
```bash
git clone https://github.com/AhmedEssam29/BERT_FineTunning.git
cd BERT_FineTunning
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

