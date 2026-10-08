"""Lightweight tabular estimator fitted only on fictional training subjects."""

from sklearn.linear_model import LogisticRegression

from ai_cvd.features.episode import PartitionPreprocessor


class TabularBaseline:
    def __init__(self, config, seed=17):
        self.config = config
        self.seed = seed
        self.subset = config["models"]["neural_context_subset"]

    def fit(self, rows, *, fitting_patients, forbidden_patients=()):
        self.preprocessor = PartitionPreprocessor(self.config).fit(
            rows, fitting_patients=fitting_patients, forbidden_patients=forbidden_patients
        )
        x = self.preprocessor.transform(rows, self.subset)
        self.estimator = LogisticRegression(C=1.0, max_iter=200, random_state=self.seed)
        self.estimator.fit(x, [int(r["label"] == 3) for r in rows])
        return self

    def predict_proba(self, rows):
        return self.estimator.predict_proba(self.preprocessor.transform(rows, self.subset))[:, 1]
