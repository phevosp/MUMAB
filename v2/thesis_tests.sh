#!/bin/bash

# Bash script to execute the Python command with specific arguments.

python main.py --num_trials 10 --alg_types indv max simple robust UCRL2 --T 500000 --K 50 --M 4 --p 0.05 --output_flag baseline --function_types linear
python main.py --num_trials 10 --alg_types robust --T 500000 --K 50 --M 4 --p 0.05 --agent_std_dev 0.1 0.1 0.1 0.1 --agent_move_prob 0.9 0.9 0.9 0.9 --agent_sample_prob 0.9 0.9 0.9 0.9 --output_flag low_noise_high_probs --function_types power --numer 1 --denom 2
python main.py --num_trials 10 --alg_types robust --T 500000 --K 50 --M 4 --p 0.05 --agent_std_dev 0.4 0.4 0.4 0.4 --agent_move_prob 0.9 0.9 0.9 0.9 --agent_sample_prob 0.9 0.9 0.9 0.9 --output_flag high_noise_high_probs --function_types power --numer 1 --denom 2
python main.py --num_trials 10 --alg_types robust --T 500000 --K 50 --M 4 --p 0.05 --agent_std_dev 0.1 0.1 0.1 0.1 --agent_move_prob 0.5 0.5 0.5 0.5 --agent_sample_prob 0.5 0.5 0.5 0.5 --output_flag low_noise_low_probs --function_types power --numer 1 --denom 2
python main.py --num_trials 10 --alg_types robust --T 500000 --K 50 --M 4 --p 0.05 --agent_std_dev 0.4 0.4 0.4 0.4 --agent_move_prob 0.5 0.5 0.5 0.5 --agent_sample_prob 0.5 0.5 0.5 0.5 --output_flag high_noise_low_probs --function_types power --numer 1 --denom 2
python main.py --num_trials 10 --alg_types robust --T 500000 --K 50 --M 4 --p 0.05 --agent_std_dev 0.4 0.4 0.4 0.4 --agent_move_prob 0.5 0.6 0.7 0.8 --agent_sample_prob 0.5 0.6 0.7 0.8 --output_flag desynchronized_probs --function_types power --numer 1 --denom 2