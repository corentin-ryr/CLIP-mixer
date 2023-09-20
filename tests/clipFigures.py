import matplotlib.pyplot as plt
import numpy as np


def clipFigures():
    top1Perf = [1.33, 0.63]
    top5Perf = [4.13, 2.03]
    cosinePerf = [[0.562, 0.320, 0.441, 0.414, 0.367, 0.569], 0.4928]
    cosinePerf = [np.mean(cosinePerf[0]), cosinePerf[1]]

    # Print the relative improvement of Mixer over Transformer
    print("Relative improvement of Mixer over Transformer")
    print("Top 1:", top1Perf[0] / top1Perf[1] )
    print("Top 5:", top5Perf[0] / top5Perf[1] )
    print("Cosine:", cosinePerf[0] / cosinePerf[1])


    # Bar chart for the 3 top perfs side by side
    fig, ax = plt.subplots(figsize=(5, 5))
    barWidth = 0.1
    plt.bar([0, 0.2], top5Perf, color="blue", width=barWidth)
    plt.bar([0, 0.2], top1Perf, edgecolor="red", color="blue", width=barWidth, hatch='/', lw=2.)
    # plt.bar([4, 5], cosinePerf, color="blue", width=barWidth, label="Cosine")
    plt.xticks([0, 0.2], ["Mixer", "Transformer"])
    plt.ylabel("Top 5% accuracy (Top 1% accuracy in red)")
    plt.savefig("clipImagenet.png")
    plt.show()

    # Bar chart of the cosine perf
    fig, ax = plt.subplots(figsize=(5, 5))
    barWidth = 0.1
    plt.bar([0, 0.2], cosinePerf, color="blue", width=barWidth)
    plt.xticks([0, 0.2], ["Mixer", "Transformer"])
    plt.ylabel("Average SRCC")
    plt.legend()
    plt.savefig("clipSTSCosine.png")
    plt.show()


if __name__ == "__main__":
    clipFigures()