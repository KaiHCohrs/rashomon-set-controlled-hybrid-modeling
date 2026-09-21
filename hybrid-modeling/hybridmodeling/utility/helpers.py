def all_scores(model, X, Y):
    score = model.v_score(X, Y)
    range = Y.max() - Y.min()
    mean = Y.mean()
    NRMSEr = score[0] / range
    NRMSEm = score[0] / mean
    return score, NRMSEr, NRMSEm


def convert_wd(value):
    if value == "None":
        value = None
    else:
        value = float(value)
    return value
