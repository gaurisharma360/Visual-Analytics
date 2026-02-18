import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1️⃣ Load Data
# --------------------------------------------------
def load_data(csv_path='bonn_eeg_combined.csv'):
    df = pd.read_csv(csv_path)

    X = df.drop(['ID', 'Y'], axis=1).values
    y = df['Y'].values

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")

    return X, y


# --------------------------------------------------
# 2️⃣ Split Data
# --------------------------------------------------
def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


# --------------------------------------------------
# 3️⃣ Train with GridSearch
# --------------------------------------------------
def train_model(X_train, y_train):

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='saga',        # supports l1 + l2
            max_iter=5000,
            random_state=42
        ))
    ])

    param_grid = {
        'clf__C': [1, 0.1, 0.01, 0.005, 0.001],
        'clf__penalty': ['l1', 'l2'],
        'clf__class_weight': [None, 'balanced']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_:.2%}")

    return grid.best_estimator_


# --------------------------------------------------
# 4️⃣ Evaluate
# --------------------------------------------------
def evaluate(model, X_train, X_test, y_train, y_test):

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print("\n==============================")
    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:  {test_acc:.2%}")
    print("==============================")

    print("\nClassification Report (Test):")
    print(classification_report(y_test, model.predict(X_test)))

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, model.predict(X_test)))


# --------------------------------------------------
# 5️⃣ Main
# --------------------------------------------------
if __name__ == "__main__":

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    evaluate(model, X_train, X_test, y_train, y_test)
