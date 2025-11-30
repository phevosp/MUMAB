import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle


def load_params():
    parser = argparse.ArgumentParser(description="MUMAB hyper parameters")
    parser.add_argument(
        "--data_folder",
        default="C:/Users/phevo/Documents/Harvard/Research/RL/Multi-G-UCB/v2/output/baseline/linear",
        type=str,
    )
    parser.add_argument(
        "--type",
        type=str,
        default="comparison",
        choices=["comparison", "failures"],
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--zoom",
        type=bool,
        default=False,
        help="Define zoomed axes",
    )
    params = parser.parse_args()
    return params


def main():
    params = load_params()
    # List all CSV files
    csv_files = [
        f
        for f in os.listdir(params.data_folder)
        if f.endswith(".csv") and "intervals" not in f
    ]
    intervals = [
        f
        for f in os.listdir(params.data_folder)
        if f.endswith(".csv") and "intervals" in f
    ]

    print(params.data_folder)
    palette = sns.color_palette("Set1", len(csv_files))
    for i, f in enumerate(csv_files):
        regrets = np.loadtxt(
            f"{params.data_folder}/{f}", delimiter=",", dtype=float, ndmin=2
        )
        interval_file = f"{params.data_folder}/{f[1:-4]}_intervals.csv"  # [1:] is because of ordering hack

        total_transition_regret = (
            calculate_transition_regret(regrets[-1], interval_file)
            if os.path.exists(interval_file)
            else None
        )
        cumulative_regrets = np.cumsum(regrets, axis=1)
        random_trial = np.random.randint(cumulative_regrets.shape[0])
        random_fit = cumulative_regrets[random_trial]
        # mean_fit = np.mean(cumulative_regrets, axis=0)
        min_fit = np.min(cumulative_regrets, axis=0)
        max_fit = np.max(cumulative_regrets, axis=0)
        T = regrets.shape[1]
        label = f[:-4]
        label = (
            " ".join(
                [word.capitalize() for word in label[1:].split("_")]
            )  # [1:] is a hack for ordering
            if params.type == "failures"
            else label[1:]  # [1:] is a hack for ordering
        )
        plt.plot(
            range(T),
            # mean_fit,
            random_fit,
            alpha=0.9,
            color=palette[i],
            label=label,
        )
        plt.fill_between(range(T), min_fit, max_fit, color="gray", alpha=0.2)
        if total_transition_regret is not None:
            print(f"Total transition regret for {label}: {total_transition_regret}")
            print(
                f"Percentage of transition regret: {total_transition_regret / cumulative_regrets[-1, -1] * 100:.2f}%"
            )

    plt.xlabel("Time")
    plt.ylabel("Cumulative Regret")
    plt.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))
    plt.grid(True)
    plt.legend()
    plt.title("Cumulative regret as a function of time")
    if params.zoom:
        plt.xlim(0, 10**5)
        plt.ylim(0, 10**4)

    if params.type == "comparison":
        plt.savefig(
            f"{params.data_folder}/cumulative_regret{'_zoom' if params.zoom else ''}.png"
        )
    else:
        plt.show()


def calculate_transition_regret(regret, interval_file):
    """Calculate total transition regret from intervals file

    Args:
        regret (List[Float]): list of regret values at each time step
        interval_file (str): path to the intervals file

    Returns:
        int: total regret from transition phase
    """
    try:
        intervals = np.loadtxt(
            interval_file, delimiter=",", converters={0: float, 1: float}, ndmin=2
        )
    except Exception as e:
        print(f"Error loading intervals from {interval_file}: {e}")
        return None

    total_transition_regret = 0
    for interval in intervals:
        start, end = interval
        transition_regret = sum(regret[int(start) : int(end)])
        total_transition_regret += transition_regret
    return total_transition_regret


if __name__ == "__main__":
    main()
