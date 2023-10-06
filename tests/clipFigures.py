import matplotlib.pyplot as plt
import numpy as np


def clipFigures():
    top1Perf = [1.33, 0.63]
    top5Perf = [4.13, 2.03]
    cosinePerf = [[0.562, 0.320, 0.441, 0.414, 0.367, 0.569], [0.435, 0.451, 0.405, 0.363, 0.374]]
    cosinePerf = [np.mean(cosinePerf[0]), np.mean(cosinePerf[1])]
    print(cosinePerf)

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
    plt.savefig("clipSTSCosine.png")
    plt.show()

def clipFiguresStep16000():
    top1Perf = [11.76, 2.38]
    top5Perf = [24.89, 7.59]
    cosinePerf = [[0.6089, 0.4688, 0.4255, 0.4233, 0.6068, 0.4908], [0.6007, 0.4465, 0.4265, 0.4499, 0.6544, 0.5561]]
    cosinePerf = [np.mean(cosinePerf[0]), np.mean(cosinePerf[1])]
    print(cosinePerf)

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
    plt.savefig("clipSTSCosine.png")
    plt.show()


if __name__ == "__main__":
    clipFiguresStep16000()