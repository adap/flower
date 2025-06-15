"""Plotting function to compare the reesults of the different approaches.

i.e. One with early stopping and one without early stopping.
"""

import pandas as pd
from results_ploting_utils import (
    plot_f1_convergence,
    plot_f1_convergence_with_stop_round,
    plot_f1_scores_baseline,
    plot_f1_scores_comparison,
    plot_global_rounds,
    plot_heat_map_of_table,
)

# Files paths - Global rounds vs Clients [F1 scores]
ES_f1_vs_round_path = "./dataframes/Validation_F1_Scores_vs_rounds_with_EA.csv"
NORMAL_f1_vs_round_path = "./dataframes/Validation_F1_Scores_vs_rounds_normal.csv"


# Distributed metrics datafram
ES_distributed_metrics_path = (
    "./fedfittech/Results_to_compare/"
    "dataframes/Client_Distributed_Metrics_100_round_with_EA.csv"
)
Normal_distributed_metrcs_path = (
    "./fedfittech/Results_to_compare/"
    "dataframes/Client_Distributed_Metrics_100_round_normal.csv"
)


# Clients labels f1 scores
ES_client_vs_labels_f1scores_path = (
    "./fedfittech/Results_to_compare/"
    "dataframes/Client_vs_label_F1scores_100_round_with_EA.csv"
)
NORMAL_client_vs_labels_f1scores_path = (
    "./fedfittech/Results_to_compare/"
    "dataframes/Client_vs_label_F1scores_100_round_normal.csv"
)


# Dataframes Global rounds vs Clients [F1 scores]
Df_ES_f1_vs_round = pd.read_csv(ES_f1_vs_round_path, sep=";")
Df_NORMAL_f1_vs_round = pd.read_csv(NORMAL_f1_vs_round_path, sep=";")


# Dataframes Distributed metrics
Df_ES_dist_metric = pd.read_csv(ES_distributed_metrics_path, sep=";")
Normal_dist_metr = pd.read_csv(Normal_distributed_metrcs_path, sep=";")


# Dataframes labels vs clients [f1 scores]
df_ES_client_vs_labels_f1scores = pd.read_csv(
    ES_client_vs_labels_f1scores_path, sep=";"
)


df_Normal_client_vs_labels_f1scores = pd.read_csv(
    NORMAL_client_vs_labels_f1scores_path, sep=";"
)
df_Normal_client_vs_labels_f1scores = df_Normal_client_vs_labels_f1scores.rename(
    columns={"Unnamed: 0": "Client_Id"}
)

# settings
df_Normal_client_vs_labels_f1scores = df_Normal_client_vs_labels_f1scores.set_index(
    "Client_Id"
)


# data_1 : To plot comparision of f1 scores for both EA and Normal approach
data_1 = {
    "Client_Id": Df_ES_dist_metric["Client_Id"],
    "F1_score_Normal": Normal_dist_metr["Validation F1 score"],
    "F1_score_ES": Df_ES_dist_metric["Validation F1 score"],
}

df_distributed_metrics_for_plot1 = pd.DataFrame(
    data_1
)  # bar plot two f1 score comparison

# data_3 to plot the compuatation saved bar plot
data_3 = {
    "Client_Id": Df_ES_dist_metric["Client_Id"],
    "F1_score_ES": Df_ES_dist_metric["Validation F1 score"],
    "F1_score_Normal": Normal_dist_metr["Validation F1 score"],
    "Training stop round": Df_ES_dist_metric["Training stop round"],
}

# plot baseline f1 scores
data_4 = {
    "Client_Id": Normal_dist_metr["Client_Id"],
    "F1_score_Normal": Normal_dist_metr["Validation F1 score"],
}


# To plot the compuatation saved bar plot
df_distributed_metrics_for_plot3 = pd.DataFrame(data_3)


# Function to plot the Barcharts, linegraph, Heatmaps
plot_f1_scores_baseline(data_4)

plot_f1_scores_comparison(df_distributed_metrics_for_plot1)

plot_f1_convergence(Df_NORMAL_f1_vs_round, path="./plots/")

plot_f1_convergence_with_stop_round(
    Df_ES_f1_vs_round, df_distributed_metrics_for_plot3, path="./plots/"
)
#
plot_heat_map_of_table(
    df_ES_client_vs_labels_f1scores,
    directory_name="Results_to_compare",
    type="Early_Stopping",
)
plot_heat_map_of_table(
    df_Normal_client_vs_labels_f1scores,
    directory_name="Results_to_compare",
    type="Normal",
)

plot_global_rounds(EA_dist_metric=Df_ES_dist_metric, path="./plots/", Global_rounds=100)
