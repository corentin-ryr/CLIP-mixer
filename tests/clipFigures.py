import matplotlib.pyplot as plt
import numpy as np


def clipFigures():
    top1Perf = [1.33, 0.63]
    top5Perf = [4.13, 2.03]
    cosinePerf = [0.571, 0.4928]


    # Bar chart for the 3 top perfs side by side
    fig, ax = plt.subplots(figsize=(5, 5))
    barWidth = 0.1
    plt.bar([0, 0.2], top5Perf, color="blue", width=barWidth, label="Top 5")
    plt.bar([0, 0.2], top1Perf, edgecolor="red", color="blue", width=barWidth, label="Top 1", hatch='/', lw=2.)
    # plt.bar([4, 5], cosinePerf, color="blue", width=barWidth, label="Cosine")
    plt.xticks([0, 0.2], ["Mixer", "Transformer"])
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("clipImagenet.png")
    plt.show()

    # Bar chart of the cosine perf
    fig, ax = plt.subplots(figsize=(5, 5))
    barWidth = 0.1
    plt.bar([0, 0.2], cosinePerf, color="blue", width=barWidth, label="Cosine")
    plt.xticks([0, 0.2], ["Mixer", "Transformer"])
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("clipSTSCosine.png")
    plt.show()


if __name__ == "__main__":
    clipFigures()