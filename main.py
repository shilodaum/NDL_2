import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


# Ex3.1:
def part_a():
    x_s = np.random.uniform(-10, 2, 500)
    y_s = np.random.uniform(18, 45, 500)
    plt.scatter(x_s, y_s)
    plt.show()


# Ex3.2:
def part_b():
    i_s = [1, 2, 4]
    for idx, i in enumerate(i_s):
        mu1 = i
        std = 0.5 * i
        x1 = np.linspace(mu1 - 3 * std, mu1 + 3 * std, 100)
        plt.subplot(1, 3, idx + 1)
        plt.plot(x1, stats.norm.pdf(x1, mu1, std))
        mu2 = -i
        x2 = np.linspace(mu2 - 3 * std, mu2 + 3 * std, 100)
        plt.plot(x2, stats.norm.pdf(x2, mu2, std))

    plt.show()


# Ex3.3:
# ???????????
def part_c():
    # Inbal -> I
    # Shilo -> S
    pass


# Ex3.4:
def part_d():
    for x in [0, 40]:
        for y in [0, 2]:
            x_arr = np.random.normal(x, 5, 500)
            y_arr = np.random.uniform(y - 0.2, y + 0.2, 500)
            plt.scatter(x_arr, y_arr, c='b')
    plt.show()

# Ex3.5:
def part_e():
    pass

def main():
    part_d()


if __name__ == '__main__':
    main()
