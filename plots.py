from numpy.typing import NDArray
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    actual: NDArray, predictions: NDArray, **kwargs
) -> plt.Figure:
    """
    Plot a confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(actual, predictions)
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return fig
