import matplotlib.pyplot as plt
import seaborn as sns


#  Plots spam and ham values
def plot_dataset_columns(data, xlabel_title, title):
    sns.countplot(data)
    plt.xlabel(xlabel_title)
    plt.title(title)
    plt.show()