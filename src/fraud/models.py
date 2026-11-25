"""Three small deterministic pipelines and one always-approve baseline."""
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from fraud.features import CATEGORICAL, FEATURES, NUMERIC
from fraud.evaluate import best_threshold, metrics


def make_pipeline(name, config, positive_weight=1.0):
    seed = config['seed']
    estimators = {
        'Always approve': DummyClassifier(strategy='constant', constant=0),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=400,
                                                   random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=60, max_depth=9,
                                                min_samples_leaf=8, class_weight='balanced',
                                                n_jobs=1, random_state=seed),
        'LightGBM': LGBMClassifier(n_estimators=90, num_leaves=15, max_depth=5,
                                   learning_rate=0.06, min_child_samples=40,
                                   scale_pos_weight=positive_weight, n_jobs=1,
                                   random_state=seed, deterministic=True,
                                   force_col_wise=True, verbosity=-1),
    }
    numeric = [('impute', SimpleImputer(strategy='median'))]
    if name == 'Logistic Regression':
        numeric.append(('scale', StandardScaler()))
    preprocess = ColumnTransformer([
        ('numeric', Pipeline(numeric), NUMERIC),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='constant',
                                                          fill_value='unknown')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore',
                                                          sparse_output=False))]), CATEGORICAL),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([('preprocess', preprocess), ('model', estimators[name])])


def fit_models(train, validation, config, scale=1.0):
    rows, bundles = [], {}
    weight = (train.is_fraud == 0).sum() / train.is_fraud.sum()
    names = ['Always approve', 'Logistic Regression', 'Random Forest', 'LightGBM']
    for name in names:
        pipeline = make_pipeline(name, config, weight)
        start = perf_counter()
        with threadpool_limits(limits=1):
            pipeline.fit(train[FEATURES], train.is_fraud)
        seconds = perf_counter() - start
        probability = pipeline.predict_proba(validation[FEATURES])[:, 1]
        threshold, curve = best_threshold(validation.is_fraud, probability, config,
                                          validation.timestamp.dt.date, scale)
        scores = metrics(validation.is_fraud, probability, threshold, config)
        bundles[name] = {'pipeline': pipeline, 'threshold': threshold, 'name': name,
                         'validation_metrics': scores, 'curve': curve}
        rows.append({'model': name, 'threshold': threshold, 'train_seconds': seconds, **scores})
    return bundles, pd.DataFrame(rows).set_index('model')


def select_model(bundles, table):
    learned = table.drop(index='Always approve')
    name = learned.sort_values(['cost_per_txn', 'pr_auc'], ascending=[True, False]).index[0]
    return bundles[name]


def train_validation_scores(bundles, train, validation, config):
    rows = []
    for name, bundle in bundles.items():
        row = {'model': name}
        for split, frame in [('train', train), ('validation', validation)]:
            probability = bundle['pipeline'].predict_proba(frame[FEATURES])[:, 1]
            score = metrics(frame.is_fraud, probability, bundle['threshold'], config)
            row[f'{split}_pr_auc'] = score['pr_auc']
            row[f'{split}_recall'] = score['recall']
        rows.append(row)
    return pd.DataFrame(rows).set_index('model')


def importance(bundle):
    model = bundle['pipeline'].named_steps['model']
    values = (np.abs(model.coef_[0]) if hasattr(model, 'coef_') else model.feature_importances_)
    names = bundle['pipeline'].named_steps['preprocess'].get_feature_names_out()
    return pd.Series(values, index=names).nlargest(10).sort_values()


def save_bundle(bundle, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
