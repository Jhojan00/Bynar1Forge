import numpy as np
import pickle


class BinaryNeuronalNetwork:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        learning_rate: float,
        cycles: int,
        neurons_like: list,
        cost_widget,
    ):
        # Initialize
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.cycles = cycles
        self.neurons_like = neurons_like
        self.cost_widget = cost_widget
        self.m = self.X.shape[1]
        self.l = len(self.neurons_like)

        # Create bias and weights in cache
        np.random.seed(24)
        self.neurons_like = [self.X.shape[0]] + self.neurons_like

        self.cache = {
            "A": {0: X},
            "Z": {},
            "W": {
                layer: np.random.rand(
                    self.neurons_like[layer], self.neurons_like[layer - 1]
                )
                * np.sqrt(1 / self.neurons_like[layer - 1])
                for layer in range(1, self.l + 1)
            },
            "B": {
                layer: np.zeros((self.neurons_like[layer], 1))
                for layer in range(1, self.l + 1)
            },
        }

    def binary_cost(self, A: np.array):
        cost = (
            -np.sum(self.Y * np.log(A + 1e-8) + (1 - self.Y) * np.log(1 - A + 1e-8))
            / self.m
        )
        return np.squeeze(cost)

    def train(self):
        cost = []
        partial_cycles = []
        for c in range(self.cycles):
            # Forward propagation
            for layer in range(1, self.l + 1):
                A = self.cache["A"][layer - 1]
                W = self.cache["W"][layer]
                B = self.cache["B"][layer]
                Z = W.dot(A) + B

                A_2 = np.tanh(Z) if layer != self.l else 1 / (1 + np.exp(-Z))

                self.cache["A"][layer] = A_2
                self.cache["Z"][layer] = Z

            # Backward propagation
            DA = None
            for layer in reversed(range(1, self.l + 1)):
                A = self.cache["A"][layer]
                Z = self.cache["Z"][layer]
                A_prev = self.cache["A"][layer - 1]
                W = self.cache["W"][layer]

                if layer == self.l:
                    DZ = A - self.Y

                else:
                    DZ = DA * (1 - (A) ** 2)

                DW = (1 / self.m) * DZ.dot(A_prev.T)
                DB = (1 / self.m) * np.sum(DZ, axis=1, keepdims=True)
                DA = W.T.dot(DZ)

                self.cache["W"][layer] -= self.learning_rate * DW
                self.cache["B"][layer] -= self.learning_rate * DB

            if c % 100 == 0:
                cost.append(self.binary_cost(self.cache["A"][self.l]))
                partial_cycles.append(c)
                self.cost_widget.line_chart(
                    {"Cost": cost, "Cycles": partial_cycles},
                    x="Cycles",
                    y="Cost",
                    x_label="Cycles",
                    y_label="Binary cost (J)",
                )

    def predict(self, X: np.ndarray):
        A = X
        for layer in range(1, self.l + 1):
            Z = self.cache["W"][layer].dot(A) + self.cache["B"][layer]
            A = np.tanh(Z) if layer != self.l else 1 / (1 + np.exp(-Z))

        return (A > 0.5).astype(int)

    def save_mind(self):
        with open("minds/model_mind.pkl", "wb") as f:
            pickle.dump(self.cache, f)

    def load_mind(self, file):
        with open(file, "rb") as f:
            load_cache = pickle.load(f)

        self.cache = load_cache
