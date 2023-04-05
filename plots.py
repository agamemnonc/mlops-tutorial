from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(actual, predictions, size=(10, 10)):
    """
    Plot a confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(actual, predictions)
    fig = plt.figure(size=size)
    ax = fig.add_subplot(111)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return fig
