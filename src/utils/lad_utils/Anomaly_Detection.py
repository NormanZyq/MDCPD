import numpy as np
import pylab as plt
from numpy import linalg as LA
from sklearn.preprocessing import normalize

from src.utils.lad_utils import lad_util

'''
find the arithmetic average as typical behavior
'''


def average_typical_behavior(context_vecs):
    avg = np.mean(context_vecs, axis=0)
    return avg


'''
find the left singular vector of the activity matrix
'''


def principal_vec_typical_behavior(context_vecs):
    activity_matrix = context_vecs.T
    u, s, vh = np.linalg.svd(activity_matrix, full_matrices=False)
    # print ("shape of each vector is: " + str(context_vecs[0].shape))
    # print ("shape of typical vector is: " + str(u[:,0].shape))
    return u[:, 0]


'''
compute the z score as defined by Akoglu and Faloutsos in EVENT DETECTION IN TIME SERIES OF MOBILE COMMUNICATION GRAPHS
Z = 1-u^(T)r
'''


def compute_Z_score(cur_vec, typical_vec):
    # print (cur_vec[0:6])
    # print (typical_vec[0:6])
    cosine_similarity = abs(np.dot(cur_vec, typical_vec) / LA.norm(cur_vec) / LA.norm(typical_vec))
    z = (1 - cosine_similarity)
    return z


'''
compute the spearman rank correlation with graph statistics
'''


def set_non_negative(z_scores):
    for i in range(len(z_scores)):
        if (z_scores[i] < 0):
            z_scores[i] = 0
    return z_scores


'''
plot different anomaly scores and how they correspond with real world
dataset = "synthetic" or "USLegis" or "canVote" or "UCI"
'''


def plot_anomaly_score(dataset, fname, scores, score_labels, events):
    for k in range(len(scores)):
        scores[k] = set_non_negative(scores[k])

    max_time = len(scores[0])
    t = list(range(0, max_time))
    plt.rcParams.update({'figure.autolayout': True})
    plt.rc('xtick')
    plt.rc('ytick')
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(1, 1, 1)
    colors = ['#fdbb84', '#43a2ca', '#bc5090', '#e5f5e0', '#fa9fb5', '#c51b8a']
    for i in range(len(scores)):
        ax.plot(t, scores[i], color=colors[i], ls='solid', lw=0.8, label=score_labels[i])

    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]

        if (dataset == "USLegis"):
            plt.annotate(str(97 + event),  # this is the text
                         (event, max_score),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(8, 0),  # distance from text to points (x,y)
                         ha='center',
                         fontsize=4)  # horizontal alignment can be left, right or center

        elif (dataset == "canVote"):
            plt.annotate(str(2006 + event),  # this is the text
                         (event, max_score),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(8, 0),  # distance from text to points (x,y)
                         ha='center',
                         fontsize=4)  # horizontal alignment can be left, right or center

        # (dataset == "synthetic" or "UCI")
        else:
            plt.annotate(str(event),  # this is the text
                         (event, max_score),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(8, 0),  # distance from text to points (x,y)
                         ha='center',
                         fontsize=4)  # horizontal alignment can be left, right or center

    addLegend = True

    for event in events:
        # plt.axvline(x=event,color='k', linestyle="--", linewidth=0.5)
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        if (addLegend):
            ax.plot(event, max_score, marker="*", markersize=5, color='#de2d26', ls='solid', lw=0.5,
                    label="detected anomalies")
            addLegend = False
        else:
            ax.plot(event, max_score, marker="*", color='#de2d26', ls='solid', lw=0.5)

    '''
    specify the xticks here
    '''

    # first day is April 14th 2004
    # UCI Message
    if (dataset == "UCI"):
        labels = ["May", "June", "July", "August", "September", "October", "November"]
        for i in range(len(labels)):
            labels[i] = str(labels[i])
        time_gaps = list(range(17, 195, 30))
        plt.xticks(time_gaps, labels, rotation='horizontal')
        ax.set_xlabel('day', fontsize=6)

    # US Legislative
    if (dataset == "USLegis"):
        labels = list(range(97, 109, 1))
        for i in range(len(labels)):
            labels[i] = str(labels[i])
        plt.xticks(t, labels, rotation='horizontal')
        ax.set_xlabel('Congress', fontsize=6)

    # canVote
    if (dataset == "canVote"):
        labels = list(range(2006, 2020, 1))
        for i in range(len(labels)):
            labels[i] = str(labels[i])
        plt.xticks(t, labels, fontsize=8, rotation='horizontal')
        ax.set_xlabel('year', fontsize=6)

    if (dataset == "synthetic"):
        ax.set_xlabel('time point', fontsize=6)

    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    ax.set_ylabel('anomaly score', fontsize=6)
    plt.legend(fontsize=5)
    plt.savefig(fname + '.pdf')

    print("plotting anomaly scores complete")


def plot_anomaly_and_spectro(fname, scores, score_labels, events, eigen_file):
    # labels = list(range(97,109,1))
    labels = list(range(1, 35, 1))
    scores[0] = set_non_negative(scores[0])

    fig, axs = plt.subplots(2)
    plt.rcParams.update({'figure.autolayout': True})

    diag_vecs = lad_util.load_object(eigen_file)
    diag_vecs = np.transpose(np.asarray(diag_vecs))  # let time be x-axis
    diag_vecs = np.flip(diag_vecs, 0)

    max_time = len(scores[0])
    t = list(range(0, max_time))
    colors = ['#fdbb84', '#43a2ca', '#bc5090', '#e5f5e0', '#fa9fb5', '#c51b8a']
    for i in range(len(scores)):
        axs[0].plot(t, scores[0], color=colors[i], ls='solid', label=score_labels[i])

    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]

        axs[0].annotate(str(labels[event]),  # this is the text
                        (event, max_score),  # this is the point to label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, -12),  # distance from text to points (x,y)
                        ha='center')  # horizontal alignment can be left, right or center

    addLegend = True

    for event in events:
        # plt.axvline(x=event,color='k', linestyle="--", linewidth=0.5)
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        if (addLegend):
            axs[0].plot(event, max_score, marker="*", color='#de2d26', ls='solid', label="detected anomalies")
            addLegend = False
        else:
            axs[0].plot(event, max_score, marker="*", color='#de2d26', ls='solid')

    # US senate
    for i in range(len(labels)):
        labels[i] = str(labels[i])

    axs[0].set_xlabel('Congress')
    axs[0].set_ylabel('anomaly score')
    plt.tight_layout()
    axs[0].legend()
    axs[0].set_xticks(t)
    axs[0].set_xticklabels(labels)

    axs[1].set_xlabel('Congress')
    axs[1].set_ylabel('rank')
    axs[1].set_xticks(t)
    axs[1].set_xticklabels(labels)

    axs[1].imshow(diag_vecs, aspect='auto')

    plt.savefig(fname + 'anomalySpectro.pdf')


def change_detection(spectrums, principal=True, percent_ranked=0.05, window=5, initial_window=10, difference=False):
    z_scores = []
    counter = 0
    for i in range(0, initial_window):
        z_scores.append(0)  # these points should never be picked

    # compute the z score for each signature vector after initial window
    # 1. find typical behavior
    # 2. compute anomaly score
    for i in range(initial_window, len(spectrums)):
        if (principal):
            typical_vec = principal_vec_typical_behavior(spectrums[i - window:i])
        else:
            typical_vec = average_typical_behavior(spectrums[i - window:i])
        cur_vec = spectrums[i]
        z_scores.append(compute_Z_score(cur_vec, typical_vec))

    '''
    change score
    '''

    # check the change in z score instead
    if (difference):
        z_scores = difference_score(z_scores)

    z_scores = np.asarray(z_scores)
    num_ranked = int(round(len(z_scores) * percent_ranked))
    outliers = z_scores.argsort()[-num_ranked:][::-1]
    outliers.sort()
    return (z_scores, outliers)


def change_detection_two_windows(spectrums, principal=True, percent_ranked=0.05, window1=5, window2=15,
                                 initial_window=20, difference=True):
    z_scores = []
    z_shorts = []
    z_longs = []
    counter = 0
    for i in range(0, initial_window):
        z_shorts.append(0)
        z_longs.append(0)
        z_scores.append(0)

        # compute the z score for each signature vector after initial window
    # 1. find typical behavior
    # 2. compute anomaly score
    for i in range(initial_window, len(spectrums)):

        # 1. compute short term window first
        if (principal):
            typical_vec = principal_vec_typical_behavior(spectrums[i - window1:i])
        else:
            typical_vec = average_typical_behavior(spectrums[i - window1:i])
        cur_vec = spectrums[i]
        z_short = compute_Z_score(cur_vec, typical_vec)
        z_shorts.append(z_short)

        # 2. compute long term window
        if (principal):
            typical_vec = principal_vec_typical_behavior(spectrums[i - window2:i])
        else:
            typical_vec = average_typical_behavior(spectrums[i - window2:i])
        cur_vec = spectrums[i]
        z_long = compute_Z_score(cur_vec, typical_vec)
        z_longs.append(z_long)

    # check the change in z score instead
    if (difference):
        z_shorts = difference_score(z_shorts)
        z_longs = difference_score(z_longs)

    z_scores = [0] * len(z_shorts)
    for i in range(len(z_scores)):
        z_scores[i] = max(z_shorts[i], z_longs[i])

    z_scores = np.asarray(z_scores)
    num_ranked = int(round(len(z_scores) * percent_ranked))
    outliers = z_scores.argsort()[-num_ranked:][::-1]
    outliers.sort()
    return (z_shorts, z_longs, z_scores, outliers)


def difference_score(z_scores):
    z = []
    for i in range(len(z_scores)):
        if (i == 0):
            z.append(z_scores[0])
        else:
            z.append(z_scores[i] - z_scores[i - 1])
    return z


def detection_with_shortwindow(eigen_file, timestamps=195, percent_ranked=0.05, window=10,
                               initial_window=15, difference=True):
    principal = True
    spectrums = lad_util.load_object(eigen_file)
    spectrums = np.asarray(spectrums).real
    spectrums = spectrums.reshape((timestamps, -1))

    spectrums = normalize(spectrums, norm='l2')

    print("window is " + str(window))
    print("initial window is " + str(initial_window))
    print(spectrums.shape)
    (z_scores, anomalies) = change_detection(spectrums, principal=principal,
                                             percent_ranked=percent_ranked, window=window,
                                             initial_window=initial_window, difference=difference)
    print("found anomalous time stamps are")
    print(anomalies)

    return z_scores


def compute_accuracy(z_scores, real_events, percent_ranked):
    z_scores = np.asarray(z_scores)
    num_ranked = int(round(len(z_scores) * percent_ranked))
    outliers = z_scores.argsort()[-num_ranked:][::-1]
    outliers.sort()
    anomalies = outliers
    correct = 0

    for anomaly in anomalies:
        if anomaly in real_events:
            correct = correct + 1

    return (correct / len(real_events))


def detection_with_bothwindows(eigen_file="UCI_eigs_slices.pkl", timestamps=195, percent_ranked=0.05, window1=10,
                               window2=30, initial_window=30, difference=True):
    principal = True
    spectrums = lad_util.load_object(eigen_file)
    spectrums = np.asarray(spectrums)

    spectrums = spectrums.real
    spectrums = spectrums.reshape((timestamps, -1))

    spectrums = normalize(spectrums, norm='l2')

    print("short window is " + str(window1))
    print("long window is " + str(window2))
    print("initial window is " + str(initial_window))
    print(spectrums.shape)
    (z_shorts, z_longs, z_scores, anomalies) = change_detection_two_windows(spectrums, principal=principal,
                                                                            percent_ranked=percent_ranked,
                                                                            window1=window1, window2=window2,
                                                                            initial_window=initial_window,
                                                                            difference=difference)
    print("found anomalous time stamps are")
    print(anomalies)

    events = anomalies
    return (z_shorts, z_longs, z_scores, events)
    # plot_anomaly_score(fname, scores, score_labels, events, real_events)
    # scores = [z_scores]
    # score_labels = ["anomaly score"]
    # plot_anomaly_and_spectro(fname, scores, score_labels, events, real_events, eigen_file=eigen_file)

    # diffscore = difference_score(z_scores)
    # plot_diff_and_nodiff("SBM", diffscore, z_scores, percent_ranked)
    # return (z_shorts,z_longs,z_scores, events)


'''
-------------------------------------------------------------------------------------------------------------------
the functions below are specific functions to run for each dataset with hyperparameter encoded
the real events below are used for evaluation to calculate the hits@k
in real world datasets these are points mentioned in other paper and is considered as weak evaluation
'''


def rdyn(data_name):
    timestamps = 33
    percent_ranked = 0.20
    eigen_file = 'experiments/cpd/lad/{}/singular.pkl'.format(data_name)
    fname = "RDyn"
    difference = True

    window1 = 1
    window2 = 2
    initial_window = 2
    (z_shorts, z_longs, z_scores, events) = detection_with_bothwindows(eigen_file=eigen_file, timestamps=timestamps,
                                                                       percent_ranked=percent_ranked, window1=window1,
                                                                       window2=window2, initial_window=initial_window,
                                                                       difference=difference)


    return events
