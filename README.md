# Heart-Disease-Classifier

### This project aims to classify heart disease cases using machine learning models such as Random Forest, Decision Tree, kNN, SVM, and more. The dataset used is [Heart Disease Dataset from Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?resource=download), which contains patient medical data to predict the likelihood of heart disease.

### Installation & Setup

#### Before running the project, make sure you have Python 3.x installed along with the required libraries.

#### Clone the Repository

git clone https://github.com/nikitasv457/Heart-Disease-Classifier.git

#### Run the Jupyter Notebook

jupyter notebook

Then, open heart_disease_classifier.ipynb and run all cells.

#### Required Libraries (Imports)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix