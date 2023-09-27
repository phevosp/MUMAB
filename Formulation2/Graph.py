import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    plt.plot(range(T), np.mean(cumulative_regrets[type], axis = 0), alpha = 0.9, color= palette[i], label = names[i])

plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
plt.legend()
plt.title("Cumulative regret as a function of time")
plt.savefig("final_av_cumulative_regret_comparison.png")

# # Plot Average Regret for different algorithm types
plt.clf()
for i, type in enumerate(type_list):
    plt.plot(range(T), np.divide(np.mean(cumulative_regrets[type], axis = 0), range(1, T+1)), alpha = 0.9, color=palette[i], label = names[i])

plt.xlabel("Time")
plt.ylabel("Average Regret")
plt.xscale('log')
plt.legend()
plt.title("Average regret as a function of time")
plt.savefig("final_av_average_regret_comparison_log.png")