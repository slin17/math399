#!/bin/bash

# var_0=$(python -c 'import model; model.regen_and_clean_correct_data(output_file= "ent_cc_regen_clean_train.csv")')

# (head -n 1 $var_0 && tail -n +2 $var_0 | sort -t',' -n -k1,2) > ./my_data/ent_cc_sorted_data.csv

python -c 'import model; model.generate_features(input_file="ent_cc_sorted_data.csv", features_file="feat0.csv", dicts_file="dict0.csv")'

# python -c 'import model; model.train_models("features1.csv")'

python -c 'import model; model.predict("cc_mod_public_leaderboard.csv", features_file="feat0.csv", dicts_file="dict0.csv", results_file="ret0.csv")'

python -c 'import model; model.eval_performance("ret0.csv")'

# python -c 'import model; model.regen_and_clean_correct_data(output_file= "cc_regen_clean_train.csv")'
