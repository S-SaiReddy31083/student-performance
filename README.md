# 📊 Student Performance Analysis

## 📌 Project Overview

This project analyzes student performance using Python libraries such as **Pandas, NumPy, Matplotlib, and Seaborn**. It focuses on exploring how different factors like study time, absences, and demographics influence student grades, along with visualizing insights and evaluating results using graphical methods.

## 📂 Dataset

The dataset contains student-related information, including:

* Grades (G1, G2, G3)
* Study time
* Absences
* Gender
* Other demographic and academic factors

## 🛠️ Tools Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**

## ⚙️ Workflow

### 1. Data Loading

* Loaded dataset using Pandas
* Inspected using `head()`, `info()`, and `describe()`

### 2. Data Cleaning

* Checked for missing values
* Ensured correct data types
* Prepared columns for analysis

### 3. Exploratory Data Analysis (EDA)

Key questions explored:

* How does gender affect student performance?
* Does study time improve grades?
* What is the impact of absences on performance?
* How strongly are G1, G2, and G3 correlated?

### 4. Data Visualization

Visualizations created using **Matplotlib and Seaborn**:

* **Histograms** to understand grade distribution
* **Bar charts** for categorical comparisons (e.g., gender vs performance)
* **Scatter plots** to analyze relationships (e.g., absences vs grades)
* **Heatmaps** for correlation analysis

### 5. Model Evaluation (Confusion Matrix)

A **confusion matrix** was implemented and visualized using **Seaborn heatmap** to better understand prediction performance.

#### Confusion Matrix Highlights:

* Displays True Positives, True Negatives, False Positives, and False Negatives
* Helps evaluate classification accuracy visually
* Implemented using `seaborn.heatmap()` for clear graphical representation

## 📊 Key Insights

* Students with higher study time tend to achieve better grades
* Increased absences are associated with lower performance
* Early grades (**G1 and G2**) strongly predict final grade (**G3**)
* Performance differences across gender are present but not always significant

## 📈 Project Structure

```bash
student-performance/
│
├── student_data.csv
├── analysis.py
├── confusion_matrix_plot.png
└── README.md
```

## 🚀 Conclusion

This project demonstrates fundamental **Exploratory Data Analysis (EDA)**, **data visualization**, and **basic model evaluation techniques**. The addition of a **confusion matrix visualization using Seaborn** improves interpretability and provides deeper insight into model performance.

---

Feel free to explore and extend this project with more advanced techniques such as machine learning models or feature engineering!
