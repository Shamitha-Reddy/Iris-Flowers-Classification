# Iris-Flowers-Classification
Predict the different species of flowers on the length of there petals and sepals


Certainly! This code is a Python script that performs several data analysis and machine learning tasks using the Iris dataset. The Iris dataset is a commonly used dataset in machine learning and consists of measurements of four features (sepal length, sepal width, petal length, and petal width) for three species of iris flowers (setosa, versicolor, and virginica).

Let's break down the code step by step:

1. **Importing Libraries:** The code begins by importing necessary libraries, including NumPy, Matplotlib, Pandas, Seaborn, and scikit-learn. These libraries are used for data manipulation, visualization, and machine learning.

2. **Loading the Dataset:**
   - `iris = pd.read_csv("iris.csv")`: Reads the Iris dataset from a CSV file into a Pandas DataFrame called `iris`.
   - Various `print` statements are used to display information about the dataset, such as its shape and summary statistics.

3. **Data Exploration and Visualization:**
   - Checking for null values: `iris.isna().sum()` is used to count the number of missing values in each column.
   - Histograms and density plots: These plots help visualize the distribution of each feature.
   - Box plots: Box plots are used to identify outliers in the dataset.
   - Violin plots: Violin plots are used to visualize the distribution of each feature for different iris species.
   - Pairplot: A pairplot is created to visualize pairwise relationships between features, colored by species.

4. **Correlation Analysis:**
   - A heatmap is created to visualize the correlation between features. This helps identify which features are highly correlated and can be important for machine learning.

5. **Data Preparation:**
   - Features (`train_X`) and target variable (`train_y`) are separated for training.
   - Features (`test_X`) and target variable (`test_y`) are separated for testing.
   - Machine learning models will be trained on `train_X` and `train_y` and tested on `test_X` and `test_y`.

6. **Machine Learning Models:**
   - Three different machine learning models are trained and evaluated:
     - Logistic Regression (`LogisticRegression`)
     - Support Vector Machine (`SVC`)
     - K-Nearest Neighbors (`KNeighborsClassifier`)

7. **Model Evaluation:**
   - Accuracy scores are calculated for each model using `metrics.accuracy_score`.
   - Confusion matrices and classification reports are generated for model evaluation. These provide insights into model performance, including precision, recall, and F1-score for each class (iris species).

8. **Summary:**
   - The code provides a comprehensive analysis of the Iris dataset, including data exploration, visualization, and the training and evaluation of machine learning models to classify iris flowers into their respective species.

Overall, this code is a practical example of using Python libraries for data analysis and machine learning on a real-world dataset. It demonstrates how to load, explore, visualize, and model data for classification tasks.
