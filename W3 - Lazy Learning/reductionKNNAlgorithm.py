from kNNAlgorithm import kNNAlgorithm
import ENNTh
import GCNN
import DROP3

class reductionKNNAlgorithm(kNNAlgorithm):

    def __init__(self, K=3):
        super().__init__(K=K)
        self.reduction_type = "GCNN"
        self.reduction_function = None

    def setReductionFunction(self, reduction, value):
        module = globals()[reduction]
        class_ = getattr(module, reduction)
        if reduction == "GCNN":
            instance = class_(self.X_train_df, self.y_train, rho=value)
        elif reduction == "ENNTh":
            instance = class_(self.X_train_df, self.y_train, mu=value)
        else:
            instance = class_(self.X_train_df, self.y_train, K=value, Knn_k=self.K)
        self.reduction_function = getattr(instance, "apply" + reduction)

    def fit(self, X, y, distance="Minkowski", voting="Majority", P=1, weighting="Equal", reduction="GCNN", value=0.3):
        super().fit(X, y, distance=distance, voting=voting, P=P, weighting=weighting)
        self.setReductionFunction(reduction, value=value)
        if reduction == "Drop3":
            kwargs = {
                'distance': distance,
                'voting': voting,
                'weighting': weighting
            }
            X, y = self.reduction_function(**kwargs)
        else:
            X, y = self.reduction_function()
        super().fit(X, y, distance=distance, voting=voting, weighting=weighting)

