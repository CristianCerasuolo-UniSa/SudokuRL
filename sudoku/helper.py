import matplotlib.pyplot as plt
from IPython import display
import time
import os

SAVE_DIR = 'training' + os.sep + time.strftime('%Y-%m-%d')

plt.ion()
plt.rcParams['figure.figsize'] = (6, 4)


def plot(scores, label):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel(label)
    plt.plot(scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.show()

def plots(results: list, labels: list, save=False):
    plt.clf()
    for i in range(len(results)):
        plt.figure(i)
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel(labels[i])
        plt.plot(results[i], label=labels[i])
        if save:
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            plt.savefig(SAVE_DIR + os.sep + f"{labels[i]}.png")
            plt.close(i)


def plot_mean(scores, mean_scores, label, save = False):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel(label)
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))


    if save:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        plt.savefig(SAVE_DIR + os.sep + label + '.png')
        plt.close()
    else:
        plt.show(block=False)
        plt.pause(.1)
