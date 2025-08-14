Decision Tree Classifier – Iris Dataset
📌 Project Overview

This project demonstrates how to build, train, and evaluate a Decision Tree Classifier using the famous Iris flower dataset.
It includes:

Data loading & exploration

Train-test splitting

Model training

Visualization of the decision tree

Evaluation using confusion matrix & classification report

Hyperparameter tuning

📂 Dataset

We use sklearn.datasets.load_iris(), which contains:

Features:

Sepal Length

Sepal Width

Petal Length

Petal Width

Target classes:

Setosa

Versicolor

Virginica

🛠️ Requirements

Install dependencies:
```

pip install pandas numpy seaborn matplotlib scikit-learn
```

🚀 Steps in the Project

Import Libraries

pandas, numpy, seaborn, matplotlib

sklearn for ML tasks

Load Dataset
```

from sklearn.datasets import load_iris
iris = load_iris()
```

Prepare Data

Convert features into a Pandas DataFrame

Assign target labels

Train-Test Split
```

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

```
Train Decision Tree
```
from sklearn.tree import DecisionTreeClassifier
treeclass = DecisionTreeClassifier(max_depth=2)  # Example with post-pruning
treeclass.fit(x_train, y_train)
```

Visualize the Tree
```

from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treeclass, filled=True)
```

Evaluate the Model
```
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_pred, y_test))
```

Hyperparameter Tuning

Example parameter grid:
```
params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter':  ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}
```
📊 Results

The trained decision tree is visualized for better interpretability.

Evaluation metrics like precision, recall, and F1-score are calculated.

Hyperparameter tuning options are explored for performance improvement.
