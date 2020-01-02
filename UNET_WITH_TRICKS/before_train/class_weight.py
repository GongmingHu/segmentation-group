import numpy as np


# 中值频率平衡
def medium_frequence_balance(count):
    frequence = np.array(count) / sum(count)

    sort_frequence = sorted(frequence)
    size = len(frequence)

    if size % 2 == 0:
        median = (sort_frequence[size//2] + sort_frequence[size//2-1])/2
    if size % 2 == 1:
        median = sort_frequence[(size - 1) // 2]

    class_weight = median / frequence

    return class_weight


class_weight = medium_frequence_balance(count=[1482445, 29974835])
print(class_weight)
