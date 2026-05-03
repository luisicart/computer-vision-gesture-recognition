import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    if 'handedness' not in df.columns:
        raise ValueError("Dataset must contain a 'handedness' column.")

    return df


def prepare_features(df):
    df = df.copy()
    df['handedness'] = df['handedness'].map({'Left': 0, 'Right': 1})

    if df['handedness'].isna().any():
        raise ValueError("Unexpected handedness values found. Expected 'Left' or 'Right'.")

    X = df.drop('label', axis=1)
    y = df['label']
    return X, y


def build_search_pipeline(random_state=42):
    pipeline = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        ]
    )

    param_distributions = {
        'classifier__n_estimators': [100, 200, 300, 400],
        'classifier__max_depth': [None, 10, 20, 30, 40],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__bootstrap': [True, False],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        scoring='accuracy',
        n_jobs=-1,
        cv=cv,
        verbose=2,
        random_state=random_state,
        return_train_score=True,
    )

    return search


def train_gesture_model(
    csv_path='./data/gestures/hand_landmarks_data.csv',
    model_path='models/custom_gesture_model.joblib',
    encoder_path='models/label_encoder.joblib',
    test_size=0.2,
    random_state=42,
):
    print(f'Loading dataset: {csv_path}')
    df = load_dataset(csv_path)

    print('Preparing features and labels...')
    X, y = prepare_features(df)

    print(f'Total samples: {len(df)}')
    print(f'Gesture classes: {sorted(y.unique())}')

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    print('Starting hyperparameter search...')
    search = build_search_pipeline(random_state)
    search.fit(X_train, y_train)

    print('\nBest hyperparameters:')
    for key, value in search.best_params_.items():
        print(f'  {key}: {value}')

    print(f'Best cross-validation accuracy: {search.best_score_:.4f}')

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print('\nTest set performance:')
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

    joblib.dump(best_model, model_path, compress=3)
    joblib.dump(label_encoder, encoder_path, compress=3)

    params_path = os.path.splitext(model_path)[0] + '_best_params.txt'
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write('Best hyperparameters:\n')
        for key, value in search.best_params_.items():
            f.write(f'{key}: {value}\n')
        f.write(f'Best cross-validation accuracy: {search.best_score_:.4f}\n')

    print(f'Optimization complete. Model saved to: {model_path}')
    print(f'Label encoder saved to: {encoder_path}')
    print(f'Hyperparameters saved to: {params_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a custom gesture recognition model.')
    parser.add_argument('--csv', default='./data/gestures/hand_landmarks_data.csv', help='Path to gesture CSV dataset')
    parser.add_argument('--model', default='models/custom_gesture_model.joblib', help='Output path for trained model')
    parser.add_argument('--encoder', default='models/label_encoder.joblib', help='Output path for label encoder')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data reserved for testing')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    train_gesture_model(
        csv_path=args.csv,
        model_path=args.model,
        encoder_path=args.encoder,
        test_size=args.test_size,
        random_state=args.random_state,
    )
