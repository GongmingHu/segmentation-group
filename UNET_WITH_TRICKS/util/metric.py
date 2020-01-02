import numpy as np

# jaccard distance
def jaccard(x, y):

    # binarize
    x = x > 0.5
    y = y > 0.5

    # stabilizing constant
    eps = 1e-10

    # compute jaccard
    intersection = np.sum(np.multiply(x, y))
    union = np.sum(x) + np.sum(y) - intersection
    return (intersection + eps) / (union + eps)

# dice coefficient
def dice(x, y):

    # binarize
    x = x > 0.5
    y = y > 0.5

    # stabilizing constant
    eps = 1e-10

    # compute dice
    intersection = np.sum(np.multiply(x, y))
    return 2*(intersection + eps) / (np.sum(x) + np.sum(y) + eps)


# accuracy, precision, recall, f-score
def accuracy_metrics(output, target, threshold, training=False):
    if training:
        # binarize
        x = output > threshold
        y = target > threshold

        # stabilizing constant
        eps = 1e-10

        tp = np.sum(np.multiply(x, y))
        tn = np.sum(np.multiply(1 - x, 1 - y))
        fp = np.sum(np.multiply(x, 1 - y))
        fn = np.sum(np.multiply(1 - x, y))

        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f_score = (2 * (precision * recall) + eps) / (precision + recall + eps)

        return accuracy, precision, recall, f_score

    else:
        thresholds = np.linspace(0.7, 0.9, 20)
        accuracy_list = []
        precision_list = []
        recall_list = []
        f_score_list = []
        for single_threshold in thresholds:
            # binarize
            x = output > single_threshold
            y = target > single_threshold

            # stabilizing constant
            eps = 1e-10

            tp = np.sum(np.multiply(x, y))
            tn = np.sum(np.multiply(1-x, 1-y))
            fp = np.sum(np.multiply(x, 1-y))
            fn = np.sum(np.multiply(1-x, y))

            accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)
            f_score = (2 * (precision * recall) + eps) / (precision + recall + eps)

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f_score_list.append(f_score)

        max_position = np.argmax(f_score_list)


        return accuracy_list[max_position], precision_list[max_position], \
               recall_list[max_position], f_score_list[max_position], thresholds[max_position]
