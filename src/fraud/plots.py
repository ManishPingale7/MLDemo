"""Readable matplotlib charts; red always denotes fraud."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

FRAUD = '#c0392b'
LEGIT = '#2874a6'
MODEL_COLORS = ['#777777', '#2874a6', '#16836b', '#8e44ad']


def setup():
    sns.set_theme(style='whitegrid', context='notebook', font_scale=1.15)
    plt.rcParams.update({'figure.figsize': (9, 4), 'figure.dpi': 100})


def baseline_chart(table):
    table[['accuracy', 'recall']].T.plot.bar(legend=False, color=[LEGIT])
    plt.title('Approving everything gives high accuracy and zero recall')
    plt.ylabel('Fraction of transactions / fraud caught')
    plt.xticks(rotation=0)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def missing_chart(raw):
    missing = raw.isna().mean().mul(100)
    missing[missing > 0].plot.bar(color=LEGIT)
    plt.title('Two fields need missing-value handling')
    plt.ylabel('Missing values (%)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def _rate(frame, column, ax, title):
    frame.groupby(column, observed=True).is_fraud.mean().mul(100).plot.bar(ax=ax, color=FRAUD)
    ax.set(title=title, ylabel='Fraud rate (%)', xlabel=column.replace('_', ' '))
    ax.tick_params(axis='x', rotation=35)


def eda_chart(frame, number, config):
    """Exactly five figures, with two panels for figures four and five."""
    if number == 1:
        counts = frame.is_fraud.value_counts().reindex([0, 1])
        counts.index = ['Legitimate', 'Fraud']
        counts.plot.bar(color=[LEGIT, FRAUD])
        plt.title(f'Fraud is rare: {frame.is_fraud.mean():.2%} of training-period rows')
        plt.ylabel('Transactions (count)')
        plt.xticks(rotation=0)
        text = f'Only {counts.iloc[1]:,} of {len(frame):,} training-period transactions are fraud.'
    elif number == 2:
        _rate(frame, 'hour', plt.gca(), 'Fraud risk varies by hour')
        plt.xlabel('Hour of day (0–23)')
        night = frame.loc[frame.hour < 6, 'is_fraud'].mean()
        day = frame.loc[frame.hour >= 6, 'is_fraud'].mean()
        text = f'Night-time fraud rate is {night:.2%}, compared with {day:.2%} at other hours.'
    elif number == 3:
        bins = np.geomspace(frame.amount_inr.min(), frame.amount_inr.max(), 45)
        for label, color, name in [(0, LEGIT, 'Legitimate'), (1, FRAUD, 'Fraud')]:
            plt.hist(frame.loc[frame.is_fraud == label, 'amount_inr'], bins=bins,
                     density=True, histtype='step', linewidth=2, color=color, label=name)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Amount (INR, logarithmic scale)')
        plt.ylabel('Density (per INR, logarithmic scale)')
        plt.title('Fraud and legitimate amounts overlap')
        plt.legend()
        ratios = (frame.amount_inr / frame.customer_avg_amount_inr).groupby(frame.is_fraud).median()
        text = (f'Median amount/usual-spend ratio is {ratios[1]:.2f} for fraud and '
                f'{ratios[0]:.2f} for legitimate payments; log amount compresses the long tail.')
    elif number == 4:
        _, axes = plt.subplots(1, 2, figsize=(13, 4))
        _rate(frame, 'merchant_category', axes[0], 'Risk differs across categories')
        _rate(frame, 'device_type', axes[1], 'Device groups have different observed rates')
        rates = frame.groupby('merchant_category').is_fraud.mean()
        text = f'{rates.idxmax()} has the highest observed category fraud rate, {rates.max():.2%}.'
    else:
        view = frame.assign(distance_bucket=pd.cut(frame.distance_from_home_km,
                            [-1, 20, config['features']['far_km'], np.inf],
                            labels=['0–20 km', '20–200 km', '>200 km']))
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        _rate(view, 'distance_bucket', axes[0], 'Distance groups show different risk')
        _rate(view, 'is_new_merchant', axes[1], 'First merchant payments have different risk')
        far = frame.distance_from_home_km > config['features']['far_km']
        text = (f'Fraud rate beyond {config["features"]["far_km"]} km is '
                f'{frame.loc[far, "is_fraud"].mean():.2%}; new-merchant payments have '
                f'{frame.loc[frame.is_new_merchant == 1, "is_fraud"].mean():.2%} fraud.')
    plt.tight_layout()
    plt.show()
    return text


def timeline(summary):
    import matplotlib.dates as mdates
    _, ax = plt.subplots(figsize=(10, 3))
    for i, row in summary.iterrows():
        left = mdates.date2num(row['start'])
        right = mdates.date2num(row['end'])
        ax.barh(row['split'], right-left, left=left, color=MODEL_COLORS[i+1])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.set(title='Training precedes validation, which precedes testing', xlabel='Transaction date')
    plt.tight_layout()
    plt.show()


def pr_curves(y, probabilities, title):
    for (name, probability), color in zip(probabilities.items(), MODEL_COLORS):
        precision, recall, _ = precision_recall_curve(y, probability)
        plt.plot(recall, precision, label=name, color=color)
    plt.axhline(np.mean(y), linestyle='--', color=FRAUD, label=f'Fraud rate {np.mean(y):.2%}')
    plt.xlabel('Recall (fraction of fraud caught)')
    plt.ylabel('Precision (fraction of alerts that are fraud)')
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def confusion(y, probabilities, thresholds):
    _, axes = plt.subplots(1, len(probabilities), figsize=(4 * len(probabilities), 3.8), squeeze=False)
    for ax, (name, probability) in zip(axes[0], probabilities.items()):
        ConfusionMatrixDisplay.from_predictions(y, probability >= thresholds[name], ax=ax,
                                                colorbar=False, cmap='Reds',
                                                display_labels=['Legit', 'Fraud'])
        ax.set_title(name, fontsize=11)
    plt.tight_layout()
    plt.show()


def metric_bars(table):
    table[['precision', 'recall', 'pr_auc']].plot.bar(figsize=(11, 4))
    plt.title('Recall and precision expose the tradeoff hidden by accuracy')
    plt.ylabel('Score (0–1)')
    plt.xlabel('Model at its validation-selected threshold')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def threshold_plot(curve, threshold):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    curve.plot(x='threshold', y=['precision', 'recall'], ax=axes[0])
    curve.plot(x='threshold', y='cost_per_txn', ax=axes[1], color=LEGIT)
    axes[0].set(title='The threshold changes which transactions are reviewed', ylabel='Score (0–1)')
    axes[1].set(title='Select minimum cost within the daily review budget', ylabel='Cost (INR / transaction)')
    for ax in axes:
        ax.axvline(threshold, color=FRAUD, linestyle='--', label=f'Selected {threshold:.2f}')
        ax.legend()
    plt.tight_layout()
    plt.show()


def importance_chart(values, name):
    values.plot.barh(color=LEGIT)
    plt.title(f'{name}: inputs used most strongly by the fitted model')
    plt.xlabel('Absolute coefficient / impurity importance / split count (model dependent)')
    plt.tight_layout()
    plt.show()


def flow():
    _, ax = plt.subplots(figsize=(12, 2))
    labels = ['commit', 'CI', 'tests', 'gate', 'registry alias']
    for i, label in enumerate(labels):
        ax.text(i, 0, label, ha='center', va='center',
                bbox={'boxstyle': 'round,pad=0.5', 'fc': '#e8f4fa'})
        if i < 4:
            ax.annotate('', xy=(i + 0.65, 0), xytext=(i + 0.3, 0),
                        arrowprops={'arrowstyle': '->'})
    ax.set(xlim=(-0.5, 4.7), ylim=(-0.5, 0.5), title='A passing gate permits a separate alias decision')
    ax.axis('off')
    plt.show()
