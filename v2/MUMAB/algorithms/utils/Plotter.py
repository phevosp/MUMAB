import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Plotter:
    def plot_cumulative_reward(reward_per_turn, max_per_turn, output_dir, T):
        plt.clf()
        plt.plot(range(T), np.cumsum(reward_per_turn), label = 'Observed')
        plt.plot(range(T), [max_per_turn * i for i in range(1, T+1)], label = 'Theoretical Max')
        plt.xlabel("Time")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative reward as a function of time")
        plt.legend()
        plt.savefig(output_dir + "/cumulative_reward.png")

    def plot_cumulative_regret(reward_per_turn, max_per_turn, output_dir, T):
        plt.clf()
        plt.plot(range(T), np.subtract([max_per_turn * i for i in range(1, T+1)], np.cumsum(reward_per_turn)))
        plt.xlabel("Time")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative regret as a function of time")
        plt.savefig(output_dir +  "/cumulative_regret.png")

    def plot_average_regret(reward_per_turn, max_per_turn, output_dir, T):
        plt.clf()
        plt.plot(range(T), np.divide(np.subtract([max_per_turn * i for i in range(1, T+1)], np.cumsum(reward_per_turn)), range(1, T+1)))
        plt.xlabel("Time")
        plt.ylabel("Average Regret")
        plt.title("Average regret as a function of time")
        plt.savefig(output_dir + "/av_regret.png")
    
    def plot_cumulative_regret_total(alg_cumulative_regrets, av_cum_regret, output_dir, T):
        plt.clf()
        for regret in alg_cumulative_regrets:
            plt.plot(range(T), regret, alpha = 0.4, color= 'grey')

        plt.plot(range(T), av_cum_regret, alpha = 0.7, color='orange')
        plt.xlabel("Time")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative regret as a function of time")
        plt.savefig(output_dir + "/av_cumulative_regret.png")

    def plot_average_regret_total(alg_cumulative_regrets, av_cum_regret, output_dir, T):
        plt.clf()
        for regret in alg_cumulative_regrets:
            plt.plot(range(T), np.divide(regret, range(1, T+1)), alpha = 0.4, color= 'grey')


        plt.plot(range(T), np.divide(av_cum_regret, range(1, T+1)), alpha = 0.7, color='orange')
        plt.xlabel("Time")
        plt.ylabel("Average Regret")
        plt.title("Average regret as a function of time")
        plt.savefig(output_dir + "/av_cumulative_regret.png")   
        np.save(output_dir + "/cumulative_regrets.npy", alg_cumulative_regrets)

    def plot_algs_mean_regret(cumulative_regrets, alg_names, alg_types, output_dir, T, log_scaled=False):
        fname = "av_cumulative_regret_comparison.png" if not log_scaled else "av_cumulative_regret_comparison_log.png"
        plt.clf()
        palette = sns.color_palette()
        for i, type in enumerate(alg_types):
            plt.plot(range(T), np.mean(cumulative_regrets[type], axis = 0), alpha = 0.9, color= palette[i], label = alg_names[i])

        plt.xlabel("Time")
        plt.ylabel("Cumulative Regret")
        if log_scaled: plt.xscale('log')
        plt.legend()
        plt.title("Cumulative regret as a function of time")
        plt.savefig(output_dir + fname)

    def plot_algs_avg_regret(cumulative_regrets, alg_names, alg_types, output_dir, T, log_scaled=False):
        fname = "av_average_regret_comparison.png" if not log_scaled else "av_average_regret_comparison_log.png"
        plt.clf()
        palette = sns.color_palette()
        for i, type in enumerate(alg_types):
            plt.plot(range(T), np.divide(np.mean(cumulative_regrets[type], axis = 0), range(1, T+1)), alpha = 0.9, color=palette[i], label = alg_names[i])

        plt.xlabel("Time")
        plt.ylabel("Average Regret")
        if log_scaled: plt.xscale('log')
        plt.legend()
        plt.title("Average regret as a function of time")
        plt.savefig(output_dir + fname)
    