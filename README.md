Decision Tree Algorithm Project Structure
Project Overview
1. The dataset contains multiple features, including student demographics (e.g., gender, parental education), previous grades, and other indicators like internet access. The target variable is the FinalGrade of each student.
2. The project workflow is designed to handle data preprocessing, feature transformation, model building, hyperparameter tuning, evaluation, and visualization.

Data Preparation
1. The first step is to load the dataset and examine its structure.
2. Features and target are separated: the features include all student information, while the target is the final grade.
3. Missing values in numeric features are handled using median imputation to ensure the model can handle incomplete data.
4. Categorical variables (like gender or parental education) are converted into numerical format using one-hot encoding, which allows the model to process them effectively.

Feature Engineering
1. Optionally, previous grades (like first-term and second-term grades) can be combined to create a new feature representing the average performance before the final grade.
2. This helps capture overall academic trends and can improve model performance.

Model Selection and Pipeline
1. A Decision Tree Classifier is used for prediction. This model is chosen because it is highly interpretable and can handle both numeric and categorical features.
2. The project uses a pipeline that combines preprocessing and model training, which ensures that all transformations are applied consistently during training and testing.

Hyperparameter Tuning
1. To optimize the decision tree, Grid Search with cross-validation is performed.

Key hyperparameters tuned include:
1. Maximum depth of the tree
2. Minimum number of samples required to split a node
3. Minimum number of samples required at a leaf
4. Cost-complexity pruning parameter (ccp_alpha)
5. This allows the project to explore both pre-pruning (limiting tree growth) and post-pruning (simplifying the tree after training), which helps prevent overfitting and improves generalization.

Model Training and Evaluation
1. The dataset is split into a training set (70%) and a testing set (30%).
2. The best model from Grid Search is trained on the training data.

Evaluation metrics include:
1. Accuracy: how often the model predicts the correct final grade
2. Confusion Matrix: shows which grades are being misclassified
3. Classification Report: includes precision, recall, and F1-score for a detailed assessment
4. This evaluation ensures that the model not only predicts well overall but also handles all categories effectively.

Interpretability and Visualization
1. The trained decision tree is visualized, showing how features influence the prediction of student grades. Each node represents a decision based on a feature value.
2. Feature importance is computed to identify the factors that contribute most to predicting final grades. This can reveal which aspects of student performance or demographics are most influential.
3. Additionally, the relationship between cross-validation scores and pruning is plotted to understand how model simplification affects performance.

Results and Insights
1. The model provides a clear view of which features are most critical for student performance.
2. Decision tree visualization allows educators to interpret rules and patterns, e.g., how prior grades or parental education affect the final outcome.
3. Hyperparameter tuning ensures the model is optimized for both accuracy and simplicity, avoiding overly complex trees that may overfit the training data.

Conclusion
1. This project demonstrates a full machine learning workflow for predicting student performance: from data preprocessing and feature encoding, to model building, hyperparameter optimization, and interpretability.
2. The outcome is a reliable, interpretable model that can be used to:
3. Predict student grades based on input features
4. Identify key factors influencing performance
5. Support educational interventions and policy decisions

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Data analysis Project Structure
Part 1: Data Cleaning & Imputation
1. Remove duplicate rows to clean the dataset.
2. Select categorical columns relevant to customer ratings.
3. Convert blank strings to NA for proper handling of missing data.
4. Impute missing values in categorical data:
5. Denoising Autoencoder (Deep Learning) with mode imputation.
6. One-hot encoding for training the autoencoder.
7. Impute missing values in numerical data:
8. Random Forest imputation using missForest for the Total_Purchases variable.

Outcome:
A fully cleaned and imputed dataset ready for machine learning analysis.

Part 2: Predictive Modeling
1. This part focuses on predicting customer ratings (Low or High) based on the cleaned data.

Models Used
1. Binary Logistic Regression
2. Evaluates linear associations between independent variables and customer ratings.

Outputs:
1. Coefficients & odds ratios
2. Confusion matrix
3. ROC Curve and AUC
4. Random Forest (Ranger)
5. Captures complex, non-linear relationships in the data.

Outputs:
1. Variable importance plot
2. Confusion matrix
3. ROC Curve and AUC
4. Model Evaluation

Compare ROC curves of Logistic Regression and Random Forest.

Analyze AUC for each model.

Identify key predictors affecting customer ratings.

Part 3: Partial Dependence Analysis
1. Explore how variables affect the probability of a High rating using Partial Dependence Plots (PDPs):
2. Total_Purchases (numeric)
3. Income (categorical)
4. Interaction: Total_Purchases × Product_Category
5. Visualize non-linear effects and interaction patterns for insights.


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Follow these guidelines to execute the code.

Project 1 - Decision Tree Algorithm Project.zip
1. Download the folder and locate the .png files. The output results are stored there.
2. The source code can be found at Decision Tree Algorithm Project\Decision Tree CART.py.
3. To run the program, download the latest Python interpreter from the official website: http://python.org/downloads/release/python-3143/
5. Install Microsoft Visual Studio 2026 Community or any other IDE to run the program.
6. For best compatibility, efficiency, accessibility, and faster processing, it is recommended to use PyCharm.
7. After installing the Python interpreter, open the settings of your IDE and create a Python virtual environment to manage scripts and packages separately.
8. The IDE will automatically detect available Python interpreters; select the latest version.
9. If you encounter issues with the latest interpreter, use a lower version instead.
10. Install all required packages before running the program. This can be done via the terminal within the virtual environment using pip or pip3.
11. Some IDEs may automatically prompt you to install missing packages when running the program, depending on the software used.
12. Alternatively, you can use the PyPI interface to search for and install required packages directly for faster setup.
13. After installing the required packages, update the CSV file path or directory in the code to match the location of your dataset (Line 16 of the code)
14. Once everything is set up, run the program.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Project 2 - Data Analysis Project.R
1. This file must be executed sequentially, line by line, in RStudio. Avoid running the entire script at once, as this may cause execution errors.
2. Other IDEs such as Microsoft Visual Studio 2026 Community or Visual Studio Code can be used; however, they require additional configuration and may lead to compatibility issues.
3. This code involves Machine Learning and Deep Learning model training, which requires high RAM capacity and significant computational resources. Running the script on low-spec machines may result in slow performance or failures.
4. Ensure that all required libraries are installed. You can install them via the Packages panel on the right-hand side of RStudio by searching for the package names. Alternatively, when executing library() calls, RStudio will prompt you to install any missing packages automatically.
5. For Machine Learning and Deep Learning functionality using Keras and TensorFlow, install Python interpreter version 3.10 or lower to avoid compatibility issues with newer versions.
6. Recommended version: Python 3.10.11: https://www.python.org/downloads/release/python-31011/
7. No manual interpreter configuration is required; RStudio will automatically detect the installed Python version when running the code.
8. For lines 34 and 37, ensure that the file paths are updated to match your own local directory.
9. Once these steps are completed, execute the code sequentially to ensure successful execution.
