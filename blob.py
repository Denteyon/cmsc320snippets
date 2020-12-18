from sklearn.metrics import confusion_matrix

def plot_confusion_heatmap(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index= [i for i in "ABC"], columns= [i for i in "DEF"]) # for columns
    # if you want more analysis on confusion matrix and change annot=True to annot=annot
    # cm_sum = np.sum(cm, axis=1, keepdims=True)
    # cm_perc = cm / cm_sum.astype(float) * 100
    # nrows, ncols = cm.shape
    # for i in range(nrows):
    #     for j in range(ncols):
    #         c = cm[i, j]
    #         p = cm_perc[i, j]
    #         if i == j:
    #             s = cm_sum[i]
    #             annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
    #         elif c == 0:
    #             annot[i, j] = ''
    #         else:
    #             annot[i, j] = '%.1f%%\n%d' % (p, c)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, cmap="Blues", annot=True)

from sklearn.model_selection import learning_curve

def plot_learning_curve(axes=None, ylim=None, train_sizes, train_scores, test_scores, fit_times):
   
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title("Learning curve")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot fit_time vs score
    axes[1].grid()
    axes[1].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[1].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[1].set_xlabel("fit_times")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Performance of the model")

    # Plot n_samples vs fit_times
    axes[2].grid()
    axes[2].plot(train_sizes, fit_times_mean, 'o-')
    axes[2].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[2].set_xlabel("Training examples")
    axes[2].set_ylabel("fit_times")
    axes[2].set_title("Scalability of the model")

    return plt


# usage
fig, axes = plt.subplots(3, 2, figsize=(10, 15))

plot_learning_curve(axes=axes[:, 1], ylim=(0.7, 1.01), train_sizes, train_scores, test_scores, fit_times)

plt.show()
