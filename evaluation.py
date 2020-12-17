import dask.array as da
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy import stats
from joblib import Parallel, delayed
from multiprocessing import Pool


class PRSEvaluator(object):

    def __init__(self, models, gdl):

        self.models = models
        self.gdl = gdl

    def fit_models(self):
        for m in self.models:
            m.fit()

    def get_heritability_estimates(self):

        h2_results = {}

        for model in self.models:
            h2_results.update({type(model).__name__: model.get_heritability()})

        return h2_results

    def k_fold_cross_validation(self, k=5):

        ind_idx = np.arange(self.gdl.N)
        np.random.shuffle(ind_idx)

        kf = KFold(n_splits=k)

        avg_r2 = {type(m).__name__: [] for m in self.models}
        pooled_r2 = {type(m).__name__: [] for m in self.models}

        for i, (train, test) in enumerate(kf.split(ind_idx)):
            print(f"Cross validation iteration {i}")

            self.gdl.set_training_samples(train_idx=train)
            self.gdl.set_testing_samples(test_idx=test)
            self.gdl.perform_gwas()

            self.fit_models()
            for model in self.models:

                #model.fit()
                prs = model.predict_phenotype()

                pooled_r2[type(model).__name__].append(
                    pd.DataFrame({'True': self.gdl.phenotypes[test],
                                  'Predicted': prs})
                )

                avg_r2[type(model).__name__].append(
                    evaluate_predictive_performance(self.gdl.phenotypes[test], prs)['R2']
                )

        for k, v in pooled_r2.items():
            df = pd.concat(v)
            pooled_r2[k] = evaluate_predictive_performance(df['True'], df['Predicted'])['R2']

        avg_r2 = {k: np.mean(v) for k, v in avg_r2.items()}

        self.gdl.set_training_samples()
        self.gdl.set_testing_samples()

        return {
            'Pooled R2': pooled_r2,
            'Average R2': avg_r2
        }


def evaluate_predictive_performance(true_phenotype, pred_phenotype):

    _, _, r_val, _, _ = stats.linregress(pred_phenotype, true_phenotype)

    return {
        'R2': r_val**2
    }


def evaluate_heritability_w_simulations(models, gdl, n_traits=10):

    results = []

    print("Evaluating Heritability with Simulations...")
    for _ in tqdm(range(n_traits)):
        gdl.simulate()
        evaluator = PRSEvaluator(models, gdl)
        evaluator.fit_models()
        results.append(evaluator.get_heritability_estimates())

    return pd.DataFrame(results)


def evaluate_prediction_w_simulations(models, gdl, n_traits=10, k=5):

    results = []

    for i in tqdm(range(n_traits)):
        print(f"> Processing Trait {i}")
        gdl.simulate()
        evaluator = PRSEvaluator(models, gdl)
        m_eval = pd.DataFrame(evaluator.k_fold_cross_validation(k=k))
        m_eval['Trait'] = i
        results.append(m_eval)

    return pd.concat(results)
