import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results(model_dense_path, model_LKCN_path):
    """
    Plots training results for two models on the same graph.

    Args:
        model_dense_path (str): Path to the CSV file containing training results for model A.
        model_b_path (str): Path to the CSV file containing training results for model B.
    """

    # Read data from CSV files
    df_dense = pd.read_csv(model_dense_path)
    df_LKCN = pd.read_csv(model_LKCN_path)

    # Create the plot
    # plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    plt.figure(figsize=(6, 4.5))

    # Plot Model dense (Red) #B7E0FF,#E78F81
    # plt.plot(df_dense['Epoch'], df_dense['Loss'], label='dense Loss', color='red', linestyle='-')
    plt.plot(df_dense['Epoch'], df_dense['Train Accuracy'], label='Dense Train Acc', color='#B7E0FF', linestyle='-')
    plt.plot(df_dense['Epoch'], df_dense['Test Accuracy'], label='Dense Valid Acc', color='#B7E0FF', linestyle=':')

    # Plot Model LKCN (Blue)
    # plt.plot(df_LKCN['Epoch'], df_LKCN['Loss'], label='LKCN Loss', color='blue', linestyle='-')
    plt.plot(df_LKCN['Epoch'], df_LKCN['Train Accuracy'], label='LKCN Train Acc', color='#E78F81', linestyle='-')
    plt.plot(df_LKCN['Epoch'], df_LKCN['Test Accuracy'], label='LKCN Valid Acc', color='#E78F81', linestyle=':')


    # Set plot properties
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    # plt.title('Training Results Comparison')
    plt.ylim(0.25, 1)  # Set y-axis limits from 0 to 1
    plt.xlim(0, 300)

    # plt.xscale('log')
    # Remove all grid lines
    plt.grid(False)

    # Add only horizontal grid lines
    # plt.gca().yaxis.grid(True)  # Enable only horizontal lines (y-axis grid)
    plt.legend(loc='lower right')  # Change the location of the legend
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping

    plt.savefig("overfit.png") # Save the plot to a file
    plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# def plot_training_shaded_results(model_a_paths, model_b_paths):
#     """
#     Plots shaded line graphs for training results of two models, each run multiple times.

#     Args:
#         model_a_paths (list): List of paths to CSV files for multiple runs of model A.
#         model_b_paths (list): List of paths to CSV files for multiple runs of model B.
#     """

#     plt.figure(figsize=(10, 6))

#     def plot_shaded_line(paths, label, color):
#         """Helper function to plot a shaded line graph for a single model."""
#         all_data = []
#         for path in paths:
#             df = pd.read_csv(path)
#             all_data.append(df)

#         # Combine data from all runs
#         combined_data = pd.concat(all_data, axis=0, ignore_index=True)

#         # Group by epoch and calculate mean and standard deviation
#         grouped = combined_data.groupby('Epoch')
#         mean = grouped.mean()
#         std = grouped.std()


#         # 'Train Accuracy'
#         plt.plot(mean.index, mean['Train Accuracy'], label=f'{label} Train Accuracy', color=color, linestyle='--')
#         plt.fill_between(mean.index, mean['Train Accuracy'] - std['Train Accuracy'], mean['Train Accuracy'] + std['Train Accuracy'], color=color, alpha=0.1)

#         # 'Test Accuracy'
#         plt.plot(mean.index, mean['Test Accuracy'], label=f'{label} Test Accuracy', color=color, linestyle='-')
#         plt.fill_between(mean.index, mean['Test Accuracy'] - std['Test Accuracy'], mean['Test Accuracy'] + std['Test Accuracy'], color=color, alpha=0.1)



#     plot_shaded_line(model_a_paths, 'Dense', 'red')
#     plot_shaded_line(model_b_paths, 'LKCN', 'blue')


#     plt.xlabel('Epoch')
#     plt.ylabel('Value')
#     plt.title('Training Results Comparison')
#     plt.xlim(0, 300)
#     plt.ylim(0.25, 1)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()

#     plt.savefig("comparison_plot_shade.png")
#     plt.show()


# model_dense_path = "results/20241010-165317bnet-dense-breast-trainset0/bnet-dense_loss_accuracy.csv"
# model_lkcn_path = "results/20241010-170111breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0-breast-trainset0/breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0_loss_accuracy.csv" 

# model_a = "bnet-dense"
# model_b = "breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0"

# dense_path_1 = f"results/20241010-175929bnet-dense-breast-trainset0/{model_a}_loss_accuracy.csv"
# lkcn_path_1 = f"results/20241010-180722breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0-breast-trainset0/{model_b}_loss_accuracy.csv"

# dense_path_2 = f"results/20241010-182143bnet-dense-breast-trainset0//{model_a}_loss_accuracy.csv"
# lkcn_path_2 = f"results/20241010-182934breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0-breast-trainset0/{model_b}_loss_accuracy.csv"

# dense_path_3 = f"results/20241010-184354bnet-dense-breast-trainset0/{model_a}_loss_accuracy.csv"
# lkcn_path_3 = f"results/20241010-185146breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0-breast-trainset0/{model_b}_loss_accuracy.csv"

# dense_path_4 = f"results/20241010-190606bnet-dense-breast-trainset0/{model_a}_loss_accuracy.csv"
# lkcn_path_4 = f"results/20241010-191357breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0-breast-trainset0/{model_b}_loss_accuracy.csv"

# dense_path_5 = f"results/20241010-192818bnet-dense-breast-trainset0/{model_a}_loss_accuracy.csv"
# lkcn_path_5 = f"results/20241010-193609breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0-breast-trainset0/{model_b}_loss_accuracy.csv"

# Example usage (replace with your actual file paths)
# model_a_paths = [
#     dense_path_1,
#     dense_path_2,
#     dense_path_3,
#     dense_path_4,
#     dense_path_5
# ]

# model_b_paths = [
#     lkcn_path_1,
#     lkcn_path_2,
#     lkcn_path_3,
#     lkcn_path_4,
#     lkcn_path_5
# ]




if __name__ == '__main__':

    # Example usage
    model_dense_path = "results/20241010-165317bnet-dense-breast-trainset0/bnet-dense_loss_accuracy.csv"
    model_lkcn_path = "results/20241010-170111breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0-breast-trainset0/breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0_loss_accuracy.csv" 
    plt.rcParams.update({'font.size': 12}) 
    plot_training_results(model_dense_path, model_lkcn_path)
    # plot_training_shaded_results(model_a_paths, model_b_paths)