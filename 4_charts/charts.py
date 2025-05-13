import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_price_with_indicators(df, price_col="Close", indicators=None, title="Price with Indicators"):
    """
    Plot price with one or more indicator columns.
    indicators: list of column names to plot alongside price.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df[price_col], label=price_col, linewidth=2)
    if indicators:
        for ind in indicators:
            if ind in df.columns:
                plt.plot(df[ind], label=ind)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_volume(df, volume_col="Volume", title="Volume"):
    """
    Plot volume as a bar chart.
    """
    plt.figure(figsize=(14, 4))
    plt.bar(df.index, df[volume_col], color='skyblue')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, columns=None, title="Correlation Heatmap"):
    """
    Plot a heatmap of correlations between selected columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns
    corr = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """
    Plot feature importances for a fitted model (e.g., tree-based).
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("Model does not have feature_importances_ attribute.")
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pairplot(df, columns=None, hue=None, title="Pairplot"):
    """
    Plot pairwise relationships in the dataset.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns
    sns.pairplot(df[columns], hue=hue)
    plt.suptitle(title, y=1.02)
    plt.show()

def plot_target_distribution(df, target_col, bins=30, title="Target Distribution"):
    """
    Plot the distribution of the target variable.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(df[target_col], bins=bins, color='orange', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
