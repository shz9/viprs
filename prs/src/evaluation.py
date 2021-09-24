from scipy import stats


def compute_r2(prs, measured_trait):
    _, _, r_val, _, _ = stats.linregress(prs, measured_trait)
    return r_val ** 2
