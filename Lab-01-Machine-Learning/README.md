# Lab 01 – Sentiment Analysis Using Traditional Machine Learning

**Module:** PUSL3205 – Large Language Models and Generative AI  
**Practical Topic:** Text Preprocessing, Bag-of-Words, TF-IDF and Machine Learning  
**Programming Language:** Python  
**Development Environment:** Google Colab  
**Dataset:** [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)  
**Main Implementation:** [`notebooks/LabSheet_1.ipynb`](notebooks/LabSheet_1.ipynb)

---

## 1. Introduction

Sentiment analysis is a Natural Language Processing (NLP) task that identifies the polarity expressed in text, commonly represented as positive or negative sentiment. Traditional machine-learning approaches can perform this classification by transforming review sentences into numerical features and learning patterns that distinguish the two classes.

This laboratory develops a binary sentiment classifier using the **UCI Sentiment Labelled Sentences** dataset, comprising product reviews from Amazon, film reviews from IMDb and restaurant reviews from Yelp. The practical follows the full workflow from dataset acquisition and preprocessing to exploratory analysis, feature extraction, training, evaluation and prediction on new sentences.

Two feature representations—**Bag-of-Words (BoW)** and **Term Frequency–Inverse Document Frequency (TF-IDF)**—are compared with two supervised classifiers: **Logistic Regression** and **Linear Support Vector Machine (SVM)**. The final model is selected using training-only cross-validation and evaluated on a held-out test set.

## 2. Objectives

The laboratory aims to develop practical understanding of a traditional NLP sentiment-classification pipeline. Its specific objectives are to:

1. Obtain and inspect a labelled text dataset from a recognised source.
2. Standardise review text using suitable preprocessing operations.
3. Compare original and cleaned text and explore sentiment-related word frequencies.
4. Divide the data into reproducible training and testing subsets.
5. Convert text to numerical vectors using the **required BoW method**, with TF-IDF as an additional comparison.
6. Train and compare Logistic Regression and Linear SVM classifiers.
7. Evaluate models using **accuracy, precision, recall, F1-score and a confusion matrix**.
8. Apply the selected model to previously unseen review sentences and examine errors.



## 3. Tools and Technologies

| Tool / Library | Purpose in this implementation |
|---|---|
| Python | Main programming language |
| Google Colab / Jupyter Notebook | Interactive execution and preservation of code and outputs |
| Pandas | Loading, combining, cleaning and exploring labelled reviews |
| NumPy | Numerical operations and fixed random seed |
| Requests and ZipFile | Downloading and extracting the UCI dataset |
| `re` and `html` | Regular-expression cleaning and HTML entity decoding |
| Scikit-learn | Data splitting, text vectorisation, cross-validation, modelling and metrics |
| Matplotlib and Seaborn | Class distribution, word-frequency and model-comparison visualisations |
| Joblib | Saving and reloading the fitted model pipeline |
| JSON / CSV | Exporting experiment settings, predictions and results |


## 4. Dataset Description

### 4.1 Source and structure

The dataset contains review sentences from three domains. Each review has a binary sentiment label: `0` for **negative** and `1` for **positive**.

| Source | Review domain | Records loaded |
|---|---|---:|
| Amazon | Product reviews | 1,000 |
| IMDb | Movie reviews | 1,000 |
| Yelp | Restaurant reviews | 1,000 |
| **Total** | | **3,000** |

The notebook retains three original fields: `review`, `sentiment` and `domain`. It reads each tab-separated file with quotation processing disabled to avoid interpreting quotation marks inside reviews as file delimiters.

### 4.2 Initial quality checks

The initial inspection produced the following results:

| Check | Recorded result |
|---|---:|
| Initial rows | 3,000 |
| Initial columns | 3 |
| Missing values across the original columns | 0 |
| Duplicate review–label pairs before cleaning | 17 |
| Initially negative reviews | 1,500 |
| Initially positive reviews | 1,500 |

The original dataset was evenly balanced across its two classes.

### 4.3 Dataset after preprocessing and deduplication

The notebook removes unusable entries and duplicate cleaned review texts, and checks for identically cleaned text with conflicting sentiment labels. The recorded run found **zero contradictory cleaned texts**.

| Cleaning result | Count |
|---|---:|
| Original reviews | 3,000 |
| Reviews removed by cleaning/deduplication | 30 |
| **Final reviews** | **2,970** |
| Final negative reviews | 1,484 |
| Final positive reviews | 1,486 |

After cleaning, the classes remained almost perfectly balanced: **49.97% negative and 50.03% positive**.

## 5. Tasks and Implementation

### 5.1 Task 1 — Download, load and inspect the dataset

The official UCI ZIP archive was downloaded, and the Amazon, IMDb and Yelp text files were loaded separately into Pandas. Their records were combined into a single DataFrame while retaining a `domain` column. The notebook checked record counts, missing values, duplicates, class balance and representative reviews.

**Evidence:** The notebook's dataset-loading and inspection outputs confirm 1,000 records per domain and 3,000 records in total.

### 5.2 Task 2 — Preprocess the text

A custom `preprocess_text()` function was applied to every review. The implemented operations include:

1. Converting text to lowercase.
2. Decoding HTML entities and standardising apostrophes.
3. Removing URLs and HTML tags.
4. Normalising selected negative contractions, such as `can't` to `can not` and `don't` to `do not`.
5. Removing punctuation and other unwanted characters while retaining letters, digits and whitespace.
6. Collapsing repeated whitespace and trimming the result.

The processed text was stored in a new **`clean_text`** column, leaving the original `review` column available for comparison and error analysis. Repeated cleaned reviews were removed before the train–test split to reduce exact-duplicate leakage.

**Recorded preprocessing examples:**

| Original sentence | Cleaned sentence |
|---|---|
| `This product is AMAZING!!!` | `this product is amazing` |
| `The movie was not good.` | `the movie was not good` |
| `I don't like this product.` | `i do not like this product` |
| `The food was    excellent!` | `the food was excellent` |


### 5.3 Task 3 — Explore review sentiment and vocabulary

The cleaned dataset was explored using class-distribution plots, representative positive and negative examples, and frequent-word tables. Stopwords were excluded for word-frequency exploration, with `not`, `no`, `nor` and `never` retained because negation matters for sentiment.

The cleaned distribution contained **1,484 negative** and **1,486 positive** reviews. Both classes therefore had almost equal representation, reducing the concern that a classifier could obtain a high accuracy simply by predicting the majority class.

The word-frequency visualisations support a comparison of common language in each class. However, frequency alone cannot reliably establish sentiment: a term can occur in both classes, and the meaning of a phrase depends on context and neighbouring words.

### 5.4 Task 4 — Split the dataset

The cleaned dataset was separated into feature text (`X = clean_text`) and sentiment labels (`y = sentiment`). An **80:20 stratified split** was performed with `random_state=42`.

| Subset | Negative | Positive | Total |
|---|---:|---:|---:|
| Training | 1,187 | 1,189 | **2,376** |
| Testing | 297 | 297 | **594** |
| **Total** | **1,484** | **1,486** | **2,970** |

Stratification preserved class balance. The test subset was reserved for evaluation, while model selection was based on cross-validation performed within the training subset.

### 5.5 Task 5 — Convert text into numerical features

**Bag-of-Words:** `CountVectorizer` was used to represent text through word and phrase frequencies. The configuration specifies `max_features=10000` and `ngram_range=(1, 2)`, allowing unigrams and bigrams. It was fitted on training text only, and the fitted vocabulary was used to transform test text.

**TF-IDF:** `TfidfVectorizer` was used as an additional experiment with the same 10,000-feature limit and unigram/bigram range. TF-IDF reduces the relative weight of terms that appear broadly across documents.

| Representation | Training matrix | Testing matrix |
|---|---|---|
| BoW | `2376 × 10000` | `594 × 10000` |
| TF-IDF | `2376 × 10000` | `594 × 10000` |

The use of bigrams can capture short expressions such as `not good`, although the model is still not designed to understand complete sentence meaning.

### 5.6 Task 6 — Train traditional machine-learning classifiers

Four pipeline configurations were implemented:

| Experiment | Representation | Classifier |
|---|---|---|
| Model 1 | BoW | Logistic Regression |
| Model 2 | BoW | Linear SVM |
| Model 3 | TF-IDF | Logistic Regression |
| Model 4 | TF-IDF | Linear SVM |

Each pipeline contains its vectoriser and classifier, ensuring that vectorisation is fitted only on the relevant training fold during validation. A **five-fold stratified cross-validation** procedure, shuffled with random seed 42, compared the models using **macro F1-score**.

| Model | Mean CV macro F1 | CV F1 standard deviation |
|---|---:|---:|
| **TF-IDF + Linear SVM** | **0.8169** | 0.0168 |
| BoW + Logistic Regression | 0.8080 | 0.0114 |
| BoW + Linear SVM | 0.8051 | 0.0144 |
| TF-IDF + Logistic Regression | 0.7995 | 0.0098 |

**Selected model:** TF-IDF + Linear SVM, because it had the highest *training cross-validation* macro F1-score. This decision was made independently of the held-out test scores.

### 5.7 Task 7 — Evaluate the classifiers

All four trained models were evaluated on the same held-out test set. Accuracy measures overall correctness; macro precision and recall describe class-level performance; macro F1 combines precision and recall while giving equal weight to both classes.

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| **TF-IDF + Linear SVM** | **85.02%** | **85.08%** | **85.02%** | **85.01%** |
| BoW + Logistic Regression | 83.84% | 83.84% | 83.84% | 83.84% |
| TF-IDF + Logistic Regression | 83.16% | 83.17% | 83.16% | 83.16% |
| BoW + Linear SVM | 82.83% | 82.83% | 82.83% | 82.83% |

The selected TF-IDF + Linear SVM model had the highest observed test accuracy among the four experiments. Its test macro F1-score was also the highest. However, these test-set comparisons are descriptive; the choice of the model was already fixed by cross-validation.

### 5.8 Task 8 — Predict sentiment in new sentences

A prediction function cleaned each new sentence with the same preprocessing function used during training, then passed the cleaned text through the fitted TF-IDF + Linear SVM pipeline.

The exact outputs recorded in the notebook were:

| New input sentence | Predicted sentiment |
|---|---|
| This product is absolutely amazing! | Positive |
| The movie was terrible and boring. | Negative |
| I really enjoyed the experience. | Positive |
| The food was not good at all. | Negative |
| The service was excellent. | Positive |
| **I would never recommend this product.** | **Positive — incorrect for the intended meaning** |
| The quality is acceptable but not impressive. | Negative |
| I love this place! | Positive |

**Seven of these eight hand-picked examples produced the expected polarity**, but this small informal demonstration is not an additional accuracy benchmark. Most importantly, the `never recommend` failure reveals a limitation of a linear bag-of-features sentiment model even when simple negation words are preserved.

## 6. Experimental Results and Evidence

### 6.1 Best-performing model

| Measure | Recorded value |
|---|---:|
| Selected method | **TF-IDF + Linear SVM** |
| Cross-validation macro F1 | 0.8169 |
| Held-out test accuracy | **0.850168 (85.02%)** |
| Held-out macro precision | **0.850841 (85.08%)** |
| Held-out macro recall | **0.850168 (85.02%)** |
| Held-out macro F1 | **0.850097 (85.01%)** |
| Correct predictions | **505** |
| Incorrect predictions | **89** |
| Test instances | 594 |

### 6.2 Confusion matrix

The final test confusion matrix was:

| Actual / Predicted | Negative | Positive | Row total |
|---|---:|---:|---:|
| **Actual negative** | **246** | 51 | 297 |
| **Actual positive** | 38 | **259** | 297 |
| **Column total** | 284 | 310 | **594** |

- **True negatives (246):** Negative reviews classified correctly.
- **False positives (51):** Negative reviews incorrectly classified as positive.
- **False negatives (38):** Positive reviews incorrectly classified as negative.
- **True positives (259):** Positive reviews classified correctly.

The classifier correctly predicted **505 of 594** reviews. Positive-class recall was **259/297 ≈ 87.21%**, while negative-class recall was **246/297 ≈ 82.83%**. The larger false-positive count indicates that the model was somewhat more likely to mislabel a negative review as positive than the reverse in this test set.

### 6.3 Classification report and error analysis

The notebook's classification report rounded both class F1-scores to approximately **0.85**. The more detailed confusion matrix reveals the difference in class recall described above.

The notebook saved **89 misclassified reviews** to `results/misclassified-reviews.csv`. Examples shown in the executed output include:

| Review excerpt | Actual | Predicted |
|---|---|---|
| “The soundtrack sucked.” | Negative | Positive |
| “Glad I didn't pay to see it.” | Negative | Positive |
| “The color is even prettier than I thought it was …” | Positive | Negative |
| “Not too screamy not to masculine but just right” | Positive | Negative |

These examples illustrate the difficulty of capturing contextual meaning with surface vocabulary and short n-grams. The reasons for individual errors are plausible interpretations, not established causal explanations.

### 6.4 Saving and reloading the model

The fitted model pipeline was saved as `output/sentiment-model.joblib` using Joblib. The notebook reloaded this artifact and successfully classified **“The product is excellent!”** as **Positive**. This confirms that the fitted model can be reused without retraining, provided the original text preprocessing function is applied first.

An experiment summary was also exported to `results/experiment-summary.json`, along with the model-comparison results, classification report, error cases and unseen-sentence predictions.

## 7. Observations and Discussion

### 7.1 Data preparation and class balance

All three source files contributed 1,000 reviews, and the initial classes were evenly distributed. Cleaning and deduplication reduced the dataset from **3,000 to 2,970** records, after which the classes remained almost equally represented. This made accuracy a useful headline measure, while macro precision, recall and F1 provided a more complete picture.

### 7.2 Comparing representations and classifiers

The TF-IDF + Linear SVM pipeline outperformed the alternative configurations in both training cross-validation and the held-out test results. However, **TF-IDF was not uniformly better than BoW**: BoW + Logistic Regression (83.84%) performed slightly better than TF-IDF + Logistic Regression (83.16%). This demonstrates that the value of a feature representation depends partly on the classifier with which it is paired.

Likewise, a Linear SVM did not automatically outperform Logistic Regression for every representation: BoW + Logistic Regression outperformed BoW + Linear SVM in the recorded run.

### 7.3 Interpretation of class-level errors

The selected classifier was somewhat more successful at recalling positive reviews (**87.21%**) than negative reviews (**82.83%**). Its 51 false positives exceeded its 38 false negatives. For applications in which negative feedback should not be overlooked, these errors may matter more than the overall accuracy figure suggests.

### 7.4 Why the model sometimes fails

The failed prediction for **“I would never recommend this product.”** shows that a sentence may be classified incorrectly despite containing an explicit negation. Some misclassified test reviews also contain indirect criticism or nuanced praise. Linear models using TF-IDF features learn statistical associations between tokens and labels, but do not explicitly model discourse, intent or long-range context.

### 7.5 Cross-validation versus test performance

The mean cross-validation macro F1 was **0.8169**, while the held-out test macro F1 was approximately **0.8501**. These figures are based on different evaluation samples and need not be identical. A single train–test split cannot establish that the model will reliably achieve the same score on another sample or a different domain.

## 8. Challenges, Limitations and Potential Improvements

The following limitations are supported by the design and observed behaviour of the experiment:

1. **Nuanced sentiment and negation:** Basic contraction handling and bigrams do not resolve every negative expression, as shown by the `never recommend` prediction.
2. **Surface-level text representation:** BoW and TF-IDF do not explicitly represent semantic context, sarcasm or implicit sentiment.
3. **Domain generalisation:** Reviews were pooled from three sources, but a separate leave-one-domain-out or per-domain evaluation was **not** performed. Performance on new domains remains uncertain.
4. **Limited model-search scope:** Four fixed configurations were compared without a systematic hyperparameter search. Further experiments may find improvements, but none are claimed here.
5. **Single held-out split:** Reported test performance is based on one stratified split; uncertainty intervals or repeated held-out evaluations were not calculated.
6. **Interpretation of word counts:** Frequent words are informative for exploration but are not sufficient proof of why a model predicts a particular class.
7. **Pipeline packaging:** The saved Scikit-learn pipeline includes vectorisation and classification, but the separate text-cleaning function must still be applied consistently before inference.

Possible follow-up experiments include evaluating by review domain, testing alternative n-gram sizes, applying training-only hyperparameter tuning, examining confidence calibration and comparing with a contextual language model. These are **future improvements**, not completed results.

## 9. Learning Outcomes

Based on the tasks demonstrated in the notebook, this laboratory develops experience in:

- Obtaining and validating a labelled NLP dataset from an external source.
- Cleaning natural-language text while retaining an original-text reference.
- Understanding class distributions, duplicates and potential data leakage.
- Visualising positive and negative review vocabulary.
- Using BoW and TF-IDF to convert text into numerical features.
- Training Logistic Regression and Linear SVM pipelines.
- Applying stratified splitting and training-only cross-validation.
- Selecting a model without using the test set for model selection.
- Evaluating classification using accuracy, macro precision, macro recall, macro F1 and confusion matrices.
- Examining actual prediction failures and acknowledging model limitations.
- Saving and reloading a fitted model and exporting experiment results.

The key practical insight is that strong overall accuracy does not remove the need to inspect class-specific errors and new-sentence behaviour.


## 10. Conclusion

This laboratory demonstrates a complete traditional sentiment-analysis workflow using the UCI Sentiment Labelled Sentences dataset. After preprocessing and deduplication, **2,970 reviews** were available for the experiment. Four combinations of feature representation and classifier were compared using five-fold stratified training cross-validation.

The **TF-IDF + Linear SVM** pipeline was selected by cross-validation and achieved **85.02% test accuracy** and **85.01% macro F1-score**, correctly classifying **505 of 594** held-out reviews. The model also showed unequal class recall and made **89 test errors**, including more false-positive than false-negative predictions.

Testing new sentences demonstrated practical inference, but also exposed a failure on the explicit negative sentence *“I would never recommend this product.”* The overall result is therefore promising for traditional NLP classification, while highlighting the importance of inspecting mistakes, understanding limitations and reporting performance honestly.

