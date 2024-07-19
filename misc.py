import matplotlib.pyplot as plt

def plot_01_loss():
    plt.figure(figsize=(4,3))
    plt.plot([0,0.5,0.5,1], [0,0,1,1], label="yt = 0")
    plt.plot([0,0.5,0.5,1], [1,1,0,0], label="yt = 1")
    plt.xlabel("ŷt")
    plt.xticks([0,0.5,1],["0","0.5","1"])
    plt.ylabel("loss")
    plt.yticks([0,1],["0","1"])
    plt.title("0/1 loss function")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_01_loss()
