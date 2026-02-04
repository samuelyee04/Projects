import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Load and import dataset
df = pd.read_csv("student_performance_realistic.csv")

# Feature Engineering: Combine G1 and G2 columns to calculate the average
# df["AvgGrade"] = (df["G1"] + df["G2"]) / 2

# Define features and target in the dataset
X = df.drop(columns="FinalGrade")
# X = df.drop(columns=["FinalGrade", "G1", "G2"])  # Drop G1, G2 since AvgGrade replaces them
y = df["FinalGrade"]

# Method 1 for encoding categorical columns
# Create label encoder for each categorical column
# from sklearn.preprocessing import LabelEncoder
# le_gender = LabelEncoder()
# le_education = LabelEncoder()
# le_internet = LabelEncoder()

# Create new columns for encoded categorical variables
# X['Gender_Encode'] = le_gender.fit_transform(X['Gender'])
# X['ParentalEducation_Encode'] = le_education.fit_transform(X['ParentalEducation'])
# X['Internet_Encode'] = le_internet.fit_transform(X['Internet'])

# Drop the old columns from the dataset after encoding new columns
# X = X.drop(['Gender', 'ParentalEducation', 'Internet'], axis='columns')

# Method 2 for encoding categorical columns
# # Encode categorical columns
# label_encoders = {}
# for column in data.columns:
#     if data[column].dtype == 'object':
#         le = LabelEncoder()
#         data[column] = le.fit_transform(data[column])
#         label_encoders[column] = le


# Identify categorical and numerical columns in the dataset
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

# Impute missing values with median for numeric columns
# Transform categorical columns into numerical format
preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Full Pipeline with preprocessing and decision tree classifier
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42, criterion="gini"))  # Gini Impurity = CART Algorithm
])

# Split dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Grid Search for Hyperparameter Tuning (pre-pruning and post-pruning)
param_grid = {
    "classifier__max_depth": [3, 5, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__ccp_alpha": [0.0, 0.001, 0.01, 0.05]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, n_jobs=-1, verbose=0
)

# Fit the training data to generate a model
grid_search.fit(X_train, y_train)

# Evaluate the best model for prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
print(f"\nEncoded Feature Names:\n {feature_names}")

print(f"\nBest Parameters: {grid_search.best_params_}")

# Identify the accuracy for train and test dataset
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"\nConfusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n {classification_report(y_test, y_pred)}")

# Tree Visualization
clf = best_model.named_steps["classifier"]
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

# Plot using matplotlib library
plt.figure(figsize=(18, 12))
plot_tree(clf, feature_names=feature_names, class_names=sorted(y.unique()), filled=True)
plt.title("Optimized Decision Tree with Cost-Complexity Pruning")
plt.savefig("optimized_decision_tree.png")
plt.show()

# Visualize the most important feature
importance = clf.feature_importances_
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(len(importance)), importance[indices], align="center")
plt.xticks(range(len(importance)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.savefig("feature_importance_plot.png")
plt.show()

# Create a DataFrame from cross-validation results
results_df = pd.DataFrame(grid_search.cv_results_)

# Sort by mean test score for visualization
results_df = results_df.sort_values(by="mean_test_score", ascending=False)

# Display top configurations
print(results_df[[
    'mean_test_score', 'std_test_score', 'params'
]].head(10))

# Visualize the relationship between cross validation and cost-complexity pruning
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=results_df,
    x="param_classifier__ccp_alpha",
    y="mean_test_score",
    marker="o"
)
plt.title("Cross-Validation Score vs. Cost-Complexity Pruning")
plt.xlabel("Cost-Complexity Alpha")
plt.ylabel("Mean of Cross-Validation Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("cv_score_vs_ccp_alpha.png")
plt.show()

# Export Tree to DOT format for Graphviz
# dot_file = "optimized_decision_tree.dot"
# export_graphviz(
#    clf,
#    out_file=dot_file,
#    feature_names=feature_names,
#    class_names=sorted(y.unique()),
#    filled=True,
#    rounded=True,
#    special_characters=True
# )

# Optional: Display using graphviz (requires graphviz installed)
# with open(dot_file) as f:
#   dot_graph = f.read()
# graphviz.Source(dot_graph)


