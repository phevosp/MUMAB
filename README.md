# Multi-Agent-MAB

Code for conference paper published to American Control Conference (see https://arxiv.org/abs/2401.10383).

## Getting started: `main.py`

To simulate a multi-agent MAB instance, run `main.py` in the v2 folder with arguments as specified below. An MAB instance is run for each combination of algorithm type and transformation function type passed in. Graphs are initialized as Erdos-Renyi graphs with connectivity probability as specified. Arms are initialized to have a gaussian reward function with mean selected uniformly at random between 0.25 and 1 and noise 0.1. Outputs are saved in output_dir which is by default generated as T-K-M in the outputs folder. Subfolders are ordered by transformation function type, then algorithm type, then trials.

#### Arguments
`T`: number of time steps for which the MAB instance will be run. <br>
`K`: number of arms/nodes in the graph. <br>
`M`: number of agents. <br>
`p`: probability of edges in the Erdos-Renyi random creation of the graph.<br>
`num_trials`: the number of trials for which the instance should be run. The same graph is used for all trails.
`function_types`: The type of transformation functions initialized for each arm. Choices are 'log', 'collision', 'more_log', 'linear', 'constant', 'power'. Choice of the power function necessitates specifying the numerator and denominator of the power. Just as multiple transformation can be passed in, multiple power functions can be passed based on the length of the numerator and denomatinor arguments.<br>
`numer`: The numerator argument (see function_types)<br>
`denom`: The denominator argument (see function_types)<br>
`output_dirs`: Automatically populated, but can be modified if passed in as argument.<br>
`alg_types`: Type of algorithm we want to run. Options are 'indv', 'max', 'median', 'original'. Individual may not work with newest additions such as transition failures and sensor bias/noise.<br>
`normalized`: Whether the regret output should be "normalized".<br>
`agent_std_dev`: The list of standard deviations for sensor noise. Needs to be of length M.<br>
`agent_bias`: The list of biases for sensor noise, needs to be of length M.<br>
`agent_move_prob` and `agent_sample_prob`: Probability of success each time the agents move or sample.
`agent_move_gamma` and `agent_sample_gamma`: Exponential bases that govern decay of movement and sampling success probabilities.