import svm
from classifier import Classifier


class PairwiseClassifier(Classifier):
    def train(self, number_of_classes: int, kernel_type: str, params: list, X: list, Y: list, c: float):
        number_of_svm = (number_of_classes - 1) * number_of_classes / 2
        self.machine = [svm.SVM() for machine in range(number_of_svm)]
        for n, machine in enumerate(self.machine):
            y_for_svm = [element[n] for element in Y]
            machine.train(kernel_type, params, X, y_for_svm, c)

    def classify(self, vector: list) -> int:
        results = [machine.get_distance(vector) for machine in self.machine]
        classes = [sum(element) for element in self.identify_affiliation(results)]
        maximum = max(classes)
        if maximum > 0:
            return classes.index(max(results))
        else:
            return -1

    @staticmethod
    def identify_affiliation(data: list) -> list:
        n = (1 + (1 + 8 * len(data)) ** 0.5) / 2
        if int(n) != n:
            raise ValueError("Data is wrong size")
        result = []
        n = int(n)
        for i in range(n):
            temp = []
            for element in result:
                temp.append(element[i - 1])
            temp += data[sum(list(range(n - 1, n - 1 - i, -1))):sum(list(range(n - 1, n - 2 - i, -1)))]
            result.append(temp)
        return result
