import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Plotter:
    def plot_cumulative_reward(reward_per_turn, max_per_turn, output_dir, T, normalized):
        plt.clf()
        plt.plot(range(T), np.cumsum(reward_per_turn), label = 'Observed')
        plt.plot(range(T), [max_per_turn * i for i in range(1, T+1)], label = 'Theoretical Max')
        plt.xlabel("Time")
        ylabel = "Normalized Cumulative Reward" if normalized else "Cumulative Reward"
        plt.ylabel(ylabel)
        title = "Normalized cumulative reward as a function of time" if normalized else "Cumulative reward as a function of time"
        plt.title(title)
        plt.legend()
        save_name = "/normalized_cumulative_reward.png" if normalized else "/cumulative_reward.png"
        plt.savefig(output_dir + save_name)

    def plot_cumulative_regret(reward_per_turn, max_per_turn, output_dir, T, normalized):
        plt.clf()
        plt.plot(range(T), np.subtract([max_per_turn * i for i in range(1, T+1)], np.cumsum(reward_per_turn)))
        plt.xlabel("Time")
        ylabel = "Normalized Cumulative Regret" if normalized else "Cumulative Regret"
        plt.ylabel(ylabel)
        title  = "Normalized cumulative regret as a function of time" if normalized else "Cumulative regret as a function of time" 
        plt.title(title)
        save_name = "/normalized_cumulative_regret.png" if normalized else "/cumulative_regret.png"
        plt.savefig(output_dir + save_name)

    def plot_average_regret(reward_per_turn, max_per_turn, output_dir, T, normalized):
        plt.clf()
        plt.plot(range(T), np.divide(np.subtract([max_per_turn * i for i in range(1, T+1)], np.cumsum(reward_per_turn)), range(1, T+1)))
        plt.xlabel("Time")
        ylabel = "Normalized Average Regret" if normalized else "Average Regret"
        plt.ylabel(ylabel)
        title  = "Normalized average regret as a function of time" if normalized else "Average regret as a function of time"
        plt.title(title)
        save_name = "/normalized_average_regret.png" if normalized else "/average_regret.png"
        plt.savefig(output_dir + save_name)
    
    def plot_cumulative_regret_total(alg_cumulative_regrets, av_cum_regret, output_dir, T, normalized):
        plt.clf()
        for regret in alg_cumulative_regrets:
            plt.plot(range(T), regret, alpha = 0.4, color= 'grey')

        plt.plot(range(T), av_cum_regret, alpha = 0.7, color='orange')
        plt.xlabel("Time")
        ylabel = "Normalized Cumulative Regret" if normalized else "Cumulative Regret"
        plt.ylabel(ylabel)
        title  = "Normalized cumulative regret as a function of time" if normalized else "Cumulative regret as a function of time" 
        plt.title(title)
        save_name = "/normalized_av_cumulative_regret.png" if normalized else "/av_cumulative_regret.png"
        plt.savefig(output_dir + save_name)

    def plot_average_regret_total(alg_cumulative_regrets, av_cum_regret, output_dir, T, normalized):
        plt.clf()
        for regret in alg_cumulative_regrets:
            plt.plot(range(T), np.divide(regret, range(1, T+1)), alpha = 0.4, color= 'grey')

        plt.plot(range(T), np.divide(av_cum_regret, range(1, T+1)), alpha = 0.7, color='orange')
        plt.xlabel("Time")
        ylabel = "Normalized Average Regret" if normalized else "Average Regret"
        plt.ylabel(ylabel)
        title  = "Normalized average regret as a function of time" if normalized else "Average regret as a function of time"
        plt.title(title)
        save_fig_name = "/normalized_av_cumulative_regret.png" if normalized else "/av_cumulative_regret.png"
        plt.savefig(output_dir + save_fig_name)   
        save_arr_name = "/normalized_cumulative_regret.npy" if normalized else "/cumulative_regret.npy"
        np.save(output_dir + save_arr_name, alg_cumulative_regrets)

    def plot_algs_cum_regret(cumulative_regrets, alg_names, alg_types, output_dir, T, normalized, log_scaled=False):
        fname = "av_cumulative_regret_comparison.png" if not log_scaled else "av_cumulative_regret_comparison_log.png"
        if normalized:
            fname = "normalized_" + fname
        plt.clf()
        palette = sns.color_palette()
        for i, type in enumerate(alg_types):
            plt.plot(range(T), np.mean(cumulative_regrets[type], axis = 0), alpha = 0.9, color= palette[i], label = alg_names[i])

        plt.xlabel("Time")
        ylabel = "Normalized Cumulative Regret" if normalized else "Cumulative Regret"
        plt.ylabel(ylabel)
        if log_scaled: plt.xscale('log')
        plt.legend()
        title  = "Normalized cumulative regret as a function of time" if normalized else "Cumulative regret as a function of time" 
        plt.title(title)
        plt.savefig(output_dir + fname)

    def plot_algs_avg_regret(cumulative_regrets, alg_names, alg_types, output_dir, T, normalized, log_scaled=False):
        fname = "av_average_regret_comparison.png" if not log_scaled else "av_average_regret_comparison_log.png"
        if normalized:
            fname = "normalized_" + fname
        plt.clf()
        palette = sns.color_palette()
        for i, type in enumerate(alg_types):
            plt.plot(range(T), np.divide(np.mean(cumulative_regrets[type], axis = 0), range(1, T+1)), alpha = 0.9, color=palette[i], label = alg_names[i])

        plt.xlabel("Time")
        ylabel = "Normalized Average Regret" if normalized else "Average Regret"
        plt.ylabel(ylabel)
        if log_scaled: plt.xscale('log')
        plt.legend()
        title  = "Normalized average regret as a function of time" if normalized else "Average regret as a function of time"
        plt.title(title)
        plt.savefig(output_dir + fname)
   
    def plot_algs_avg_regret_ftypes(regret, function_types, type, alg_name, T, output_dir, normalized):
        fname = f"av_average_regret_comparison_{type}"
        if normalized:
            fname = "normalized_" + fname
        plt.clf()
        palette = sns.color_palette()
        for i, ftype in enumerate(function_types):
            plt.plot(range(T), np.divide(regret[ftype], range(1, T+1)), alpha = 0.9, color= palette[i], label=ftype)

        plt.xlabel("Time")
        ylabel = "Normalized Average Regret" if normalized else "Average Regret"
        plt.ylabel(ylabel)
        plt.legend()
        title  = f"Normalized average regret for algorithm {alg_name}" if normalized else f"Average regret for algorithm {alg_name}"
        plt.title(title)
        plt.savefig(output_dir + fname)

    def plot_algs_cum_regret_ftypes(regret, function_types, type, alg_name, T, output_dir, normalized):
        fname = f"av_cumulative_regret_comparison_{type}"
        if normalized:
            fname = "normalized_" + fname
        plt.clf()
        palette = sns.color_palette()
        for i, ftype in enumerate(function_types):
            plt.plot(range(T), regret[ftype], alpha = 0.9, color= palette[i], label=ftype)

        plt.xlabel("Time")
        ylabel = "Normalized Cumulative Regret" if normalized else "Cumulative Regret"
        plt.ylabel(ylabel)
        plt.legend()
        title  = f"Normalized cumulative regret for algorithm {alg_name}" if normalized else f"Cumulative regret for algorithm {alg_name}"
        plt.title(title)
        plt.savefig(output_dir + fname)
