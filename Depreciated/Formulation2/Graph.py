import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx

cumulative_regrets = {}
type_list = ['original', 'median', 'max', "individual"]
names     = ["Multi-G-UCB", "Multi-G-UCB-median", "Multi-G-UCB-max", "Indv-G-UCB"]

for type in type_list:
    cumulative_regrets[type] = np.load('Regular_id/150000-300-20/' + type + '/cumulative_regrets.npy')

T = cumulative_regrets[type].shape[1]
print(T)

# # Plot Mean Regret for different algorithm types
plt.clf()
palette = sns.color_palette()
for i, type in enumerate(type_list):
    yfit = np.mean(cumulative_regrets[type], axis=0)
    min_fit = np.min(cumulative_regrets[type], axis=0)
    max_fit = np.max(cumulative_regrets[type], axis=0)

    plt.plot(range(T), yfit, alpha = 0.9, color= palette[i], label = names[i])
    plt.fill_between(range(T), min_fit, max_fit,
                            color='gray', alpha=0.2)

plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
plt.grid(True)
plt.legend()
plt.title("Cumulative regret as a function of time")
plt.ylim(0, 250000)
plt.xlim(0, 150000)
# plt.show()
plt.savefig("final_av_cumulative_regret_comparison_error.png")

K = 300
p = 0.05

G = nx.erdos_renyi_graph(K, p, seed = 0)
tries = 0
while not nx.is_connected(G) and tries < 10:
    G = nx.erdos_renyi_graph(K, p, seed = 0 + tries)
    tries += 1
assert(nx.is_connected(G))

nx.write_gexf(G, "test.gexf")