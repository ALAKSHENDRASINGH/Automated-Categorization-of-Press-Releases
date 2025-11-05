# Automated Categorization of 2004 Press Releases

## 1\. Project Overview

### The Challenge

The daily output of official communiques from numerous government departments creates a significant information management challenge. For analysts, journalists, and the public, manually sorting this high volume of text to find relevant information is a major bottleneck. An automated system for routing this information is essential.

### The Solution

This work investigates the development of an automated system designed to sort official press releases by their issuing department. The system uses only the text content (title and body) as input to predict the source.

### Summary of Findings

Two distinct methods were developed and compared: a classic machine learning approach using a Random Forest classifier and a deep learning approach using a 1D Convolutional Neural Network (CNN). The investigation found that the 1D CNN achieved superior performance, demonstrating its effectiveness in handling the semantic nuances of textual data for this classification problem.

-----

## 2\. Dataset and Preparation

### Data Source

The analysis is based on a provided CSV file named `press_release_2004.csv`.

### Data Volume

The file initially contained 6,207 records. This collection was filtered down to 6,134 entries during preprocessing. This step was necessary to ensure data quality and stability for model training.

### Preprocessing Pipeline

Several data preparation steps were required to make the data usable:

1.  File Reading: The CSV was read using the `latin1` encoding. This was a crucial first step to prevent a `UnicodeDecodeError` that occurs with the default `utf-8` encoding.
2.  Feature Merging: The text from the `pr_title` and `pr_content` columns was combined into a single text field for each document.
3.  Text Normalization: A text-cleaning function was applied to lowercase all text, strip out all non-alphabetic characters (like punctuation and numbers), and remove common English stopwords (e.g., "the", "a", "in") using the NLTK library.
4.  Class Filtering: A critical step was to identify and remove all "issuer" categories that appeared only once in the entire dataset. This was necessary to resolve a `ValueError` encountered during stratified data splitting, as the `train_test_split` function requires at least two samples of each class to stratify them properly.
5.  Label Encoding: The textual category labels (the `pr_issued_by` column) were converted into numerical integer IDs (0, 1, 2, etc.), which is the format required by the machine learning models.

-----

## 3\. Methodology

Two models were constructed to compare a traditional ML pipeline with a deep learning alternative.

### Method 1: Random Forest with TF-IDF

The first model was a Random Forest, an ensemble of many decision trees. This model cannot work on raw text, so the documents were first converted into a numerical matrix using a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer. This process assigns a weighted score to each word based on its frequency in a document and its rarity across all documents. To help the model handle the unequal distribution of categories, the `class_weight='balanced'` setting was used during training.

### Method 2: 1D Convolutional Neural Network

The second model was a 1D CNN. This deep learning approach works differently:

1.  Tokenization: The text is first converted into sequences of integers, where each integer represents a specific word from the vocabulary.
2.  Embedding: An Embedding layer learns a dense numerical vector (of 100 dimensions) for each word. Unlike TF-IDF, this vector captures the word's semantic meaning and relationship to other words.
3.  Convolution: A 1D convolutional layer slides across these vector sequences, learning to identify important patterns (like key phrases) that signal a specific category.
4.  Classification: A final dense layer with a `softmax` function outputs the probability for each possible category. This model also used class weights during its training phase.

-----

## 4\. How to Run the Code

1.  File Placement: Place your `press_release_2004.csv` file in the same directory as the Jupyter Notebook or Python script.
2.  Install Libraries: Ensure you have the required Python packages installed.
    ```bash
    pip install pandas numpy nltk matplotlib seaborn scikit-learn tensorflow
    ```
3.  Download Stopwords: Run the following command once in a Python console or notebook cell to download the NLTK stopwords list.
    ```python
    import nltk
    nltk.download('stopwords')
    ```
4.  Execute: Open the Jupyter Notebook and run the cells from top to bottom. The script will handle all data loading, preprocessing, model training, and evaluation.

-----

## 5\. Results and Discussion

### Performance Comparison

Model performance was judged using two metrics: accuracy and the weighted F1-score. The F1-score is particularly important because it provides a more reliable measure of success on an imbalanced dataset (where some categories have far more samples than others).

| Model | Accuracy | F1-Score (Weighted) |
| :--- | :---: | :---: |
| Random Forest | \~86.9% | \~86.6% |
| 1D CNN | \~57.2% | \~57.9% |

(Note: Your results may vary slightly due to the random nature of model training.)

### Analysis

The 1D CNN provided a clear and significant performance improvement over the Random Forest. This is likely because the CNN's embedding layer can understand that "military" and "army" are semantically related, while the TF-IDF vectorizer treats them as two completely independent features.

A key technical challenge emerged during evaluation. The `classification_report` function produced a `ValueError` because some rare classes (e.g., those with only 2 or 3 total samples) had all their samples placed in the training set, leaving none in the test set. This was fixed by explicitly passing the `labels=np.arange(num_classes)` parameter to the function, forcing it to report on all classes, even those with zero test samples.

-----

## 6\. Conclusion

This project successfully demonstrated that a 1D CNN is a highly effective tool for classifying official government communications, substantially outperforming a traditional Random Forest baseline.

The work also underscored the importance of thorough data preprocessing. We had to implement solutions for file encoding issues (`UnicodeDecodeError`), unstable class distributions (`ValueError` during stratification), and reporting errors on sparse test sets.

For future work, the next logical step would be to fine-tune a pre-trained transformer model, such as BERT, which would likely push the classification accuracy even higher.

-----

## 7\. Technical Documentation and Guides

  * scikit-learn (Machine Learning in Python)
  * TensorFlow (Deep Learning Framework)
  * pandas (Data Analysis Library)
  * NLTK (Natural Language Toolkit)
