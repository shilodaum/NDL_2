import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cluster import KMeans, AgglomerativeClustering


# Ex3:
def part_a():
    x_s = np.random.uniform(-10, 2, 500)
    y_s = np.random.uniform(18, 45, 500)
    plt.scatter(x_s, y_s)
    plt.show()
    return x_s, y_s


# ?????????????????????
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

# ???????????
def part_c():
    # Inbal -> I
    # Shilo -> S
    pass

def part_d():
    x_final = []
    y_final = []
    for x in [0, 40]:
        for y in [0, 2]:
            x_arr = np.random.normal(x, 5, 500)
            y_arr = np.random.uniform(y - 0.2, y + 0.2, 500)
            x_final.extend(x_arr)
            y_final.extend(y_arr)
    return x_final, y_final

def part_e():
    x_final = []
    y_final = []
    r = 1
    shift_x = 0.4
    shift_y = 0.3
    thetas0 = np.random.uniform(np.pi / 2, 3 * np.pi / 2, 500)
    thetas1 = np.random.uniform(-np.pi / 2, np.pi / 2, 500)
    x_s0 = (r * np.sin(thetas0) + np.random.uniform(-0.05, 0.05, 500)) + shift_x
    y_s0 = (r * np.cos(thetas0) + np.random.uniform(-0.05, 0.05, 500)) + shift_y

    x_s1 = (r * np.sin(thetas1) + np.random.uniform(-0.05, 0.05, 500)) - shift_x
    y_s1 = (r * np.cos(thetas1) + np.random.uniform(-0.05, 0.05, 500)) - shift_y

    x_final.extend(x_s0)
    x_final.extend(x_s1)

    y_final.extend(y_s0)
    y_final.extend(y_s1)

    return x_final, y_final

def part_f():
    x_final = []
    y_final = []

    r = 1
    shift_x = 0.4
    shift_y = 0.35

    thetas0 = np.random.uniform(np.pi / 2, 3 * np.pi / 2, 500)
    thetas1 = np.random.uniform(-np.pi / 2, np.pi / 2, 500)

    x_s0 = (r * np.sin(thetas0) + np.random.uniform(-0.05, 0.05, 500)) + shift_x
    y_s0 = (r * np.cos(thetas0) + np.random.uniform(-0.05, 0.05, 500)) + shift_y

    x_s1 = (r * np.sin(thetas1) + np.random.uniform(-0.05, 0.05, 500)) - shift_x
    y_s1 = (r * np.cos(thetas1) + np.random.uniform(-0.05, 0.05, 500)) - shift_y

    x_final.extend(x_s0)
    x_final.extend(x_s1)
    y_final.extend(y_s0)
    y_final.extend(y_s1)

    # Add the sparsely connected
    thetas0 = np.random.uniform(3 * np.pi / 2, 3 * np.pi / 2 + 0.3, 10)
    thetas1 = np.random.uniform(np.pi / 2, np.pi / 2 + 0.3, 10)
    x_s0 = (r * np.sin(thetas0)) + shift_x
    y_s0 = (r * np.cos(thetas0)) + shift_y

    x_s1 = (r * np.sin(thetas1)) - shift_x
    y_s1 = (r * np.cos(thetas1)) - shift_y

    x_final.extend(x_s0)
    x_final.extend(x_s1)
    y_final.extend(y_s0)
    y_final.extend(y_s1)
    return x_final, y_final

def show_part_1_function():
    plt.figure(figsize=(5, 5))
    x, y = part_f()
    plt.scatter(x, y, c='b', s=1)
    plt.show()


# Ex4:
def run_K_means():
    for k in [2, 3, 4, 5]:
        for j in range(2):
            x, y = part_e()
            data = np.dstack((x, y)).reshape(len(x), 2)
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(data)

            filtered_label = []
            for i in range(k):
                filtered_label.append(data[labels == i])
                plt.scatter(filtered_label[i][:, 0], filtered_label[i][:, 1])

            plt.show()

def hierarchical_clustering():
    # Single linkage
    for k in [2, 4]:
        x, y = part_e()
        data = np.dstack((x, y)).reshape(len(x), 2)
        hierarchical = AgglomerativeClustering(n_clusters=k, linkage='single')
        labels = hierarchical.fit_predict(data)

        filtered_label = []
        for i in range(k):
            filtered_label.append(data[labels == i])
            plt.scatter(filtered_label[i][:, 0], filtered_label[i][:, 1])

        plt.show()

    # Complete linkage
    for k in [2, 4]:
        x, y = part_e()
        data = np.dstack((x, y)).reshape(len(x), 2)
        hierarchical = AgglomerativeClustering(n_clusters=k, linkage='complete')
        labels = hierarchical.fit_predict(data)

        filtered_label = []
        for i in range(k):
            filtered_label.append(data[labels == i])
            plt.scatter(filtered_label[i][:, 0], filtered_label[i][:, 1])

        plt.show()

    # Average linkage
    for k in [2, 4]:
        x, y = part_e()
        data = np.dstack((x, y)).reshape(len(x), 2)
        hierarchical = AgglomerativeClustering(n_clusters=k, linkage='average')
        labels = hierarchical.fit_predict(data)

        filtered_label = []
        for i in range(k):
            filtered_label.append(data[labels == i])
            plt.scatter(filtered_label[i][:, 0], filtered_label[i][:, 1])

        plt.show()

def main():
    hierarchical_clustering()

if __name__ == '__main__':
    main()
