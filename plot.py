import  matplotlib as pltjdkddlfhjk

def plot():
    history = [result0] + history1 + history2 + history3 
    accuracies = [result['val_acc'] for result in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');