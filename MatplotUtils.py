import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score

def show_histogram(list_data):
    binwidth=min(len(list_data),20) #number of bar

    plt.hist(list_data,bins=100)
    # plt.xlim([0, 10])
    plt.show()

def show_multi_data_histogram(list_data,labels):
    # listdata = [[v1,v2,...vn],[u1,u2,...un]] with 2 dataset

    colors = ['red', 'lime', 'tan']
    colors=colors[:len(labels)]
    binwidth = 20  # number of bar
    plt.hist(list_data,bins=binwidth, histtype='bar', color=colors, label=labels)
    plt.legend(prop={'size': 10})
    plt.show()

def show_confusion_matrix(y_test, y_predict):
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_predict)

    # Plot the confusion matrix.
    sns.heatmap(cm,
                annot=True)
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Prediction', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predict)
    print("Accuracy   :", accuracy)

