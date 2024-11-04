# import matplotlib.pyplot as plt
# import numpy as np

# data = [
#     [
#         [0.879747, 0.791667, 0.8, 0.808511, 0.820452, 0.892084],
#         [0.882353, 0.84375, 0.818182, 0.794118, 0.872865, 0.937284],
#         [0.589958, 0.741667, 0.644928, 0.570513, 0.792877, 0.659909]
#     ],
#     [
#         [0.8607594936708861, 0.7777777777777778, 0.7608695652173912, 0.7446808510638298, 0.8136339153000351, 0.8878665899942496],
#         [0.8627450980392157, 0.8333333333333334, 0.78125, 0.7352941176470589, 0.8518527021120585, 0.9005190311418685],
#         [0.5732217573221757, 0.725, 0.6304347826086957, 0.5576923076923077, 0.7970346592008035, 0.6584414581402533]
#     ]
# ]
# datasets = ['Breast', 'Prostate', 'Lung']
# models = ['Accuracy', 'Precision', 'F1', 'Recall', 'AUPR', 'AUC']

# fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create a figure and a set of subplots

# width = 0.35  # the width of the bars

# x = np.arange(len(models))  # the label locations

# for i in range(3):  # Iterate through the datasets
#     ax = axes[i]
#     rects1 = ax.bar(x - width/2, data[0][i], width, label='With Local Knowledge Constrain')
#     rects2 = ax.bar(x + width/2, data[1][i], width, label='Without Local Knowledge Constrain')

#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('Scores')
#     ax.set_title(datasets[i])
#     ax.set_xticks(x)
#     ax.set_xticklabels(models)
#     ax.legend()

#     def autolabel(rects):
#         """Attach a text label above each bar in *rects*, displaying its height."""
#         for rect in rects:
#             height = rect.get_height()
#             ax.annotate('{:.2f}'.format(height),
#                         xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 3),  # 3 points vertical offset
#                         textcoords="offset points",
#                         ha='center', va='bottom')


#     autolabel(rects1)
#     autolabel(rects2)

# plt.tight_layout()  # Adjust subplot parameters for a tight layout
# plt.savefig('different_cancers1.png') 
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

data = [
    [
        [0.879747, 0.791667, 0.8, 0.808511, 0.820452, 0.892084],
        [0.882353, 0.84375, 0.818182, 0.794118, 0.872865, 0.937284],
        [0.589958, 0.741667, 0.644928, 0.570513, 0.792877, 0.659909]
    ],
    [
        [0.8607594936708861, 0.7777777777777778, 0.7608695652173912, 0.7446808510638298, 0.8136339153000351, 0.8878665899942496],
        [0.8627450980392157, 0.8333333333333334, 0.78125, 0.7352941176470589, 0.8518527021120585, 0.9005190311418685],
        [0.5732217573221757, 0.725, 0.6304347826086957, 0.5576923076923077, 0.7970346592008035, 0.6584414581402533]
    ]
]
datasets = ['Breast', 'Prostate', 'Lung']
models = ['Accuracy', 'Precision', 'F1', 'Recall', 'AUPR', 'AUC']
plt.rcParams.update({'font.size': 10}) 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

width = 0.35

x = np.arange(len(models))

# 紫色系 #9568D1 #EF8BA2
# 红蓝色系 #B7E0FF,#E78F81
for i in range(3):
    ax = axes[i]
    rects1 = ax.bar(x - width/2, data[0][i], width, label='With Local Knowledge Constrain', color='#E78F81')
    rects2 = ax.bar(x + width/2, data[1][i], width, label='Without Local Knowledge Constrain', color='#B7E0FF')

    ax.set_ylabel('Scores', fontsize=12.5)
    ax.set_title(datasets[i], fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12.5)
    ax.set_ylim(0.4, 1) # Set y-axis limit starting from 0.4

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),  # Format to two decimal places
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    ax.legend(loc='upper left', fontsize=12.5) # Place legend at upper left corner


plt.tight_layout()
plt.savefig('different_cancers.png') 
plt.show()