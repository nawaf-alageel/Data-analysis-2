![License](https://img.shields.io/badge/license-MIT-blue.svg)

# 📊 Data Analysis 2 Project

### 🗓️ Deadline: Thursday, 24th October 2024, 11:59 pm

## Overview

This project demonstrates the application of three core data science techniques: 
- **Naive Bayes** for classification 🩺
- **Market Basket Analysis** for association rule learning 🛒
- **Text Analysis** for text classification 📰 

Each technique is applied to a specific dataset, and the results, including code and reports, are hosted in this GitHub repository.

## Repository Structure
<pre>

├── Market_Basket_Analysis/
│   ├── market_basket_analysis.ipynb
│   ├── olist_order_items_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_products_dataset.csv
│   └── product_category_name_translation.csv
│
├── Naive_Bayes_Classifier/
│   ├── naive_bayes_classifier.ipynb
│   └── pima_indians_diabetes.csv
│
├── Text_Analysis/
│   ├── test.csv
│   ├── text_classification.ipynb
│   └── train.csv
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
|</pre>

# 🩺 Naive Bayes Classifier for Diabetes Prediction

## Overview
This project uses the **Pima Indians Diabetes Dataset** to build a Naive Bayes classifier aimed at predicting diabetes. The classifier's performance is evaluated through various data preprocessing techniques, including handling missing values, scaling, and imputation. Advanced analysis is also performed to enhance the model's accuracy. 📈

## Dataset
- **Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** Outcome (0 for non-diabetic, 1 for diabetic)

## Methodology
1. **Data Preprocessing:** 🔧
   - Replaced invalid zero values with NaN in key features.
   - Used KNN imputation to fill missing values.
   - Applied standardization and log transformation to skewed features.
2. **Model Development:** 🏗️
   - Trained a Naive Bayes classifier using all features and a reduced set.
   - Evaluated accuracy, classification report, and ROC-AUC for different versions of the model.
3. **Advanced Techniques:** 🚀
   - Removed outliers based on Z-scores.
   - Created interaction terms for feature engineering.
   - Used ensemble methods (Voting Classifier) to boost model performance.

## Results
- **Accuracy (All Features with Imputation):** 0.74 ⭐
- **Accuracy (Reduced Features):** 0.75 ⭐
- **Accuracy (After Removing Outliers):** 0.76 ⭐
- **Ensemble Model Accuracy:** 0.73 ⭐
- **ROC-AUC:** 0.74 (on test data) 📊

## Deliverables
- **`naive_bayes_classifier.ipynb`:** Jupyter Notebook implementing the Naive Bayes classifier.

## Insights
- Dropping 'Insulin' and 'SkinThickness' improved the model's accuracy slightly. 📈
- Handling outliers positively impacted model performance. 🌟
- Combining models with ensemble techniques did not significantly outperform the individual models.

## How to Run
1. Load the dataset from the provided link.
2. Run the `naive_bayes_classifier.ipynb` in Jupyter Notebook. 🖥️

# 🛒 Market Basket Analysis for Brazilian E-Commerce

## Overview
This project implements **Market Basket Analysis** on the Brazilian E-Commerce dataset using association rule learning techniques. The analysis identifies frequent itemsets and discovers patterns in product purchases to inform decision-making. 📊

## Dataset
- **Source:** [Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Features:** Order details, product information, categories, and customer details.

## Methodology
1. **Data Preprocessing:** 🔍
   - Merged different tables to create a comprehensive dataset.
   - Handled missing values in product categories and other relevant columns.
   - Transformed data into a transaction-based format for analysis.
2. **Market Basket Analysis:** 🛍️
   - Used the Apriori algorithm to identify frequent itemsets.
   - Generated association rules based on metrics like support, confidence, and lift.
3. **Key Insights:** 💡
   - Identified frequently purchased product categories, such as 'furniture_decor' and 'bed_bath_table'.
   - Discovered strong associations, e.g., 'health_beauty' → 'perfumery' with high confidence and lift.

## Results
- **Top 3 Frequent Itemsets:** 🥇
  1. Furniture & Decor (Support: 28.5%)
  2. Bed & Bath Table (Support: 27.8%)
  3. Housewares (Support: 14.3%)

- **Top 3 Association Rules:** 📜
  1. If 'Health & Beauty', then 'Perfumery' (Confidence: 48%, Lift: 4.85)
  2. If 'Home Comfort', then 'Bed & Bath Table' (Confidence: 86%, Lift: 3.09)
  3. If 'Toys', then 'Baby' (Confidence: 38.8%, Lift: 2.98)

## Deliverables
- **`market_basket_analysis.ipynb`:** Jupyter Notebook implementing the analysis.

## How to Run
1. Load the dataset from the provided link.
2. Run the `market_basket_analysis.ipynb` in Jupyter Notebook. 🖥️

# 📰 Fake News Text Classification

## Overview
This project builds a text classification model to identify fake news using various machine learning and deep learning techniques. The dataset is preprocessed to improve model performance, and evaluation metrics are used to assess the model's effectiveness. 🔍

## Dataset
- **Source:** [Fake News Detection Dataset](https://www.kaggle.com/c/fake-news/data)
- **Files:**
  - `train.csv`: Training dataset for model training.
  - `test.csv`: Test dataset for evaluation.

## Methodology
1. **Data Preprocessing:** 🧹
   - Cleaned missing values, tokenized text, and applied lemmatization.
   - Used TF-IDF for classical models and word embeddings for deep learning models.
2. **Models:** 🏗️
   - Classical Models: Naive Bayes, Logistic Regression.
   - Deep Learning Models: Bidirectional LSTM, LSTM with Attention Mechanism.
3. **Performance Metrics:** 📊
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC for model evaluation.

## Results
- **Naive Bayes Accuracy:** 86% 📈
- **Logistic Regression Accuracy:** 92% 📈
- **Bidirectional LSTM Accuracy:** 92% 📈
- **Attention LSTM Accuracy:** 91% 📈

## Deliverables
- **`text_classification.ipynb`:** Jupyter Notebook with model implementation.

## Insights
- Deep learning models, especially the Bidirectional LSTM, demonstrated superior performance with high accuracy. 🌟
- The project highlights the effectiveness of word embeddings and hyperparameter tuning in improving model results. 📈

## 🚀 How to Use

1. **Clone the Repository:**

   <pre><div class="dark bg-gray-950 contain-inline-size rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language">git clone https://github.com/nawaf-alageel/Data-analysis-2.git
   </code></div></div></pre>
2. **Navigate to the Project Directory:**

   <pre><div class="dark bg-gray-950 contain-inline-size rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">cd Data-analysis-2
   </code></div></div></pre>
3. **Open the Jupyter Notebooks:** 📓

   * For Naive Bayes Classifier: `Naive_Bayes_Classifier/naive_bayes_classifier.ipynb`
   * For Market Basket Analysis: `Market_Basket_Analysis/market_basket_analysis.ipynb`
   * For Text Analysis: `Text_Analysis/text_classification.ipynb`

   You can open these notebooks in Jupyter by running:

   <pre><div class="dark bg-gray-950 contain-inline-size rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">jupyter notebook
   </code></div></div></pre>

## Requirements

* Python 3.x 🐍
* Jupyter Notebook 📓
* Pandas 📊
* NumPy 🔢
* Scikit-learn ⚙️
* Matplotlib (optional for visualization) 📈
* Seaborn (optional for visualization) 
* NLTK (for text analysis) 📖
* TensorFlow (for deep learning models) ⚡
* Keras (for deep learning models) 🧠
* TfidfVectorizer (for feature extraction) 🔍
* WordCloud (for word cloud visualization) ☁️
* TextBlob (for sentiment analysis) 💬
* Plotly (for interactive visualizations) 📊
* SHAP (for explainable AI) 🧩
* gensim (for pre-trained word embeddings) 🎓

## 🤝 Collaborators

* Nawaf Alageel 👨‍💻
* Aziz Salah 👨‍💻

## 🙏 Acknowledgements

Special thanks to **Umm Al-Qura University** for providing the datasets and guidance throughout this project. 🎓
