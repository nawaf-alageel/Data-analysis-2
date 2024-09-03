
# Data Analysis 2 Project

### Deadline: Thursday, 24th October 2024, 11:59 pm

## Overview

This project demonstrates the application of three core data science techniques: Naive Bayes for classification, Market Basket Analysis for association rule learning, and Text Analysis for text classification. Each technique is applied to a specific dataset, and the results, including code and reports, are hosted in this GitHub repository.

## Repository Structure

|-- Naive_Bayes_Classifier/
|   |-- pima_indians_diabetes.csv
|   |-- naive_bayes_classifier.ipynb
|   |-- naive_bayes_report.pdf
|
|-- Market_Basket_Analysis/
|   |-- brazilian_ecommerce.csv
|   |-- market_basket_analysis.ipynb
|   |-- market_basket_report.pdf
|
|-- Text_Analysis/
|   |-- train.csv
|   |-- test.csv
|   |-- submit.csv
|   |-- text_classification.ipynb
|   |-- text_analysis_report.pdf
|
|-- README.md

span

### 1. Naive Bayes Classifier

* **Objective:** Build a Naive Bayes classifier to predict diabetes using the Pima Indians Diabetes Dataset.
* **Dataset:** [Pima Indians Diabetes Dataset]()
* **Deliverables:**
  * `naive_bayes_classifier.ipynb`: Jupyter Notebook implementing the Naive Bayes classifier.
  * `naive_bayes_report.pdf`: A report on data processing, model choice, performance evaluation, and insights.

### 2. Market Basket Analysis

* **Objective:** Apply Market Basket Analysis using association rule learning techniques on the Brazilian E-Commerce Dataset.
* **Dataset:** [Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
* **Deliverables:**
  * `market_basket_analysis.ipynb`: Jupyter Notebook for Market Basket Analysis.
  * `market_basket_report.pdf`: A report summarizing discovered rules and their real-world applications.

### 3. Text Analysis (Extra Model)

* **Objective:** Build a text classification model for categorizing fake news.
* **Dataset:** [Fake News Detection Dataset](https://www.kaggle.com/c/fake-news/data)
* **Deliverables:**
  * `text_classification.ipynb`: Jupyter Notebook for text classification.
  * `text_analysis_report.pdf`: A report on text preprocessing, model performance, and insights.
  * `train.csv`: Training dataset for the text classification model.
  * `test.csv`: Test dataset for evaluating the model.
  * `submit.csv`: Submission file containing the model predictions.

## How to Use

1. **Clone the Repository:**

   <pre><div class="dark bg-gray-950 contain-inline-size rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">git clone https://github.com/nawaf-alageel/Data-analysis-2.git
   </code></div></div></pre>
2. **Navigate to the Project Directory:**

   <pre><div class="dark bg-gray-950 contain-inline-size rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">cd Data-analysis-2
   </code></div></div></pre>
3. **Open the Jupyter Notebooks:**

   * For Naive Bayes Classifier: `Naive_Bayes_Classifier/naive_bayes_classifier.ipynb`
   * For Market Basket Analysis: `Market_Basket_Analysis/market_basket_analysis.ipynb`
   * For Text Analysis: `Text_Analysis/text_classification.ipynb`

   You can open these notebooks in Jupyter by running:

   <pre><div class="dark bg-gray-950 contain-inline-size rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">jupyter notebook
   </code></div></div></pre>

## Requirements

* Python 3.x
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-learn
* Matplotlib (optional for visualization)
* NLTK (for text analysis)

## Results

The results of each task, including code and reports, are included in their respective folders. The reports provide detailed explanations of the methodologies used, the findings, and their potential real-world applications.

## Collaborators

* Nawaf Alageel
* Aziz Salah

## **License**

This project is licensed under the MIT License - see the [LICENSE]() file for details.

## Acknowledgements

Special thanks to Umm Al-Qura University for providing the datasets and guidance throughout this project.
