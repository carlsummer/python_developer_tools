# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/30/2021 9:19 AM
# @File:sklean_utils
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


def make_confusion_matrix(cf,
                          cm_path='./cm/',
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    https://github.com/DTrimarchi10/confusion_matrix
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    plt.savefig(os.path.join(cm_path, "Confusion_Matrix_all.png"), dpi=200, bbox_inches='tight')
    plt.show()


def estimate_acc_f1_p_r(gt_labels, preds):
    """计算准确率，精确率，召回率，混淆矩阵，f1score"""
    accuracy = accuracy_score(gt_labels, preds)  # 准确率
    precision = precision_score(gt_labels, preds)  # 精确率
    recall = recall_score(gt_labels, preds)  # 召回率
    f1score = f1_score(y_true=gt_labels, y_pred=preds, average='macro')  # f1score
    cm = confusion_matrix(gt_labels, preds)  # 混淆矩阵
    return accuracy, precision, recall, f1score, cm


def plot_confusion_matrix(cm, labels, cm_path='./cm/', title='Confusion Matrix', cmap=plt.cm.Blues):
    """画混淆矩阵图"""
    if not os.path.exists(cm_path):
        os.makedirs(cm_path)
    fig, ax = plt.subplots()  # figsize=(6,6)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    xlocations = np.array(range(len(labels)))
    ax.set(xticks=xlocations,
           yticks=xlocations,
           # ... and label them with the respective list entries
           xticklabels=labels,
           yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black")
    #     ax.figure.savefig(cm_path + 'Confusion_Matrix', format='png')
    plt.savefig(os.path.join(cm_path, 'Confusion_Matrix.png'), format='png')

    # Normalized
    fign, axn = plt.subplots()
    im = axn.imshow(cm, interpolation='nearest', cmap=cmap)
    axn.figure.colorbar(im, ax=axn)
    xlocations = np.array(range(len(labels)))
    axn.set(xticks=xlocations,
            yticks=xlocations,
            # ... and label them with the respective list entries
            xticklabels=labels,
            yticklabels=labels,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')
    plt.setp(axn.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(axn.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fmt = '.3f'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axn.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="black")
    axn.figure.savefig(os.path.join(cm_path, "Normalized_Confusion_Matrix.png"), format='png')
    plt.show()
