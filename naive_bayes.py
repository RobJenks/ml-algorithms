import math
import random
from functools import reduce
from statistics import mean, stdev
from typing import List, Tuple, Dict
from common import io


def run_naive_bayes():
    datasets = [("Diabetes classification", "data/pima-indians-diabetes-dataset.csv"),
                ("Iris classification", "data/iris.csv")]

    for dataset, path in datasets:
        print("\n> Executing naive Bayes for \"{}\" dataset".format(dataset))
        run_for_dataset(path)


def run_for_dataset(path):
    data = load_data(path)

    train, test = generate_datasets(data, 0.67)
    print("Creating training ({} records) and test ({} records) datasets".format(len(train), len(test)))

    print("Generating model on training dataset")
    model = generate_model(train)

    train_predictions = predict_dataset(model, train)
    print("Prediction accuracy on training dataset: {}".format(calculate_accuracy(train, train_predictions)))

    test_predictions = predict_dataset(model, test)
    print("Prediction accuracy on test dataset: {}".format(calculate_accuracy(test, test_predictions)))


def load_data(path):
    return [[try_numeric(x) for x in line.split(',')]
            for line in io.read_file(path).splitlines(False)[1:]]


def generate_datasets(data, training_pc) -> Tuple[List, List]:
    training_n = int(training_pc * len(data))

    test = list(data)
    training = []

    for _ in range(training_n):
        training.append(test.pop(random.randrange(0, len(test))))

    return training, test


def generate_model(dataset) -> Dict:
    classes = get_classes(dataset)
    return {x: calculate_stats(y) for x, y in classes.items()}


def calculate_stats(dataset):
    stats = [(mean(feature), stdev(feature)) for feature in zip(*[x[0:-1] for x in dataset])]

    return stats


def get_classes(dataset) -> Dict[float, List]:
    classes = {}

    for x in dataset:
        if x[-1] not in classes:
            classes[x[-1]] = []
        classes[x[-1]].append(x)

    return classes


# Calculate Gaussian PDF for a value given the corresponding dataset mean/stdev
def gaussian_pdf(x, x_bar, sd):
    ex = math.exp(-(math.pow(x - x_bar, 2) / (2 * math.pow(sd, 2))))
    return (1.0 / (math.sqrt(2 * math.pi) * sd)) * ex


# Calculates the combined probability for a class given a set of features, based on
# an existing model of Gaussian PDFs per class & feature
def calculate_class_prob(model: Dict, features: List):
    prob = {}
    for cls, stats in model.items():
        # Product of each feature probability, based on its corresponding stats taken from the model
        prob[cls] = reduce(lambda x, y: x * y,
                           [gaussian_pdf(x, stat[0], stat[1]) for x, stat in zip(features, stats)], 1.0)

    return prob


# Predicts the class of an input based on its feature vector and the existing model
def predict(model: Dict, features: List):
    class_prob = calculate_class_prob(model, features)

    # Simply take the most likely class as our prediction
    return max([(prob, cls) for (cls, prob) in class_prob.items()])[1]


# Generates predictions for an entire dataset, given an existing model
def predict_dataset(model: Dict, dataset):
    return [predict(model, x) for x in dataset]


# Calculate the model prediction accuracy against pre-labelled data
def calculate_accuracy(dataset, predictions):
    return sum(pred == data[-1] for (data, pred) in zip(dataset, predictions)) / len(dataset)


def try_numeric(s):
    try:
        return float(s)
    except ValueError:
        return s
