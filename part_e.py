import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


def part_e():
    plt.figure(figsize=(5, 5))
    r = 1
    shift_x = 0.4
    shift_y = 0.4
    thetas0 = np.random.uniform(np.pi / 2, 3 * np.pi / 2, 500)
    thetas1 = np.random.uniform(-np.pi / 2, np.pi / 2, 500)
    x_s0 = r * np.sin(thetas0)
    y_s0 = r * np.cos(thetas0)

    x_s1 = r * np.sin(thetas1)
    y_s1 = r * np.cos(thetas1)
    plt.scatter(x_s0 + shift_x, y_s0 + shift_y, s=1,c='r')
    plt.scatter(x_s1 - shift_x, y_s1 - shift_y, s=1,c='r')

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()


if __name__ == '__main__':
    part_e()
