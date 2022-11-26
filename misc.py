import pandas as pd
import numpy as np
from collections import OrderedDict
from collections import Counter
from itertools import chain
from config import get_config
import torch

args = get_config()

def df_label_counter(df, colName):
    """Count labels in dataframe"""
    if isinstance(df, str):
        df = pd.read_csv(df)
    labels = df[colName].tolist()
    labels = list(map(lambda x: x.split(","), labels))
    flattenLabels = list(chain(*labels))
    counter = Counter(flattenLabels)
    return counter

def compute_sampler_weight(csvPath, colName):
    """Build weights of sampler in order to train with over-sampling strategy"""
    # Load dataframe and transform label column
    df = pd.read_csv(csvPath)
    lbls = df[colName].apply(lambda x: x.split(","))

    # Define weight mapper
    counter = df_label_counter(df, colName)
    weightMapper = {c: 1/(np.log(count)+1) for c, count in counter.items()}

    # Compute weights using indexing labels of least image count
    weights = [None] * df.shape[0]
    for i, lbl in enumerate(lbls):
        if len(lbl) == 1:
            weights[i] = weightMapper[lbl[0]]
            continue

        minCount = np.argmin([counter[l] for l in lbl])
        weights[i] = weightMapper[lbl[minCount]]

    Mapper = {str(i): 0 for i in range(args.num_classes)}
    Mapper.update(weightMapper)
    loss_weight = torch.FloatTensor(list(Mapper.values()))

    # Write weights out
    return weights, loss_weight