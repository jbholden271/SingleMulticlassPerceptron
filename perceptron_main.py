import sys
import csv
import statistics


def read_csv(csv_path):
    """Read in input data from a csv."""
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # remove header
        next(reader)
        attributes = []
        labels = []
        for row in reader:
            attributes.append([float(r) for r in row[:-1]])
            labels.append(row[-1])
        data = [row + [labels[index]] for index, row in enumerate(attributes)]
        return data


def standardize(examples):
    """Transform data to use z-scores instead of raw values."""
    classes = [samp[-1] for samp in examples]
    transpose = [[examples[j][i] for j in range(len(examples))] for i in range(len(examples[0]) - 1)]
    attr_info = [(statistics.mean(attr), statistics.stdev(attr)) for attr in transpose]
    for i, r in enumerate(transpose):
        for j, c in enumerate(r):
            transpose[i][j] = (c - attr_info[i][0]) / attr_info[i][1]
    double_transpose = [[transpose[j][i] for j in range(len(transpose))] + [classes[i]] for i in
                        range(len(transpose[0]))]
    return double_transpose


def learn_weights(examples):
    """Learn attribute weights for a multiclass perceptron."""
    weights = {}  # one set of weights for each class
    iterations = 0
    mistakes = 1
    min_mistakes = 1000
    while mistakes > 0 and iterations < 1000:
        mistakes = 0
        for example in examples:
            if example[-1] not in weights:
                weights[example[-1]] = [0 for _ in range(len(example) - 1)]
            pred = max([(w, sum([e1 * e2 for e1, e2 in zip(weights[w], example[:-1])])) for w in weights],
                       key=lambda x: x[1])
            if pred[0] != example[-1]:
                mistakes += 1
                weights[pred[0]] = [x1 - x2 for x1, x2 in zip(weights[pred[0]], example[:-1])]
                weights[example[-1]] = [x1 + x2 for x1, x2 in zip(weights[example[-1]], example[:-1])]
        iterations += 1
        min_mistakes = min(min_mistakes, mistakes)
    print("Weights learned in", iterations, "iterations with", min_mistakes, "misclassifications.")
    return weights


def print_weights(weights):
    for c, wts in sorted(weights.items()):
        print("class {}: {}".format(c, ",".join([str(w) for w in wts])))


if __name__ == '__main__':
    data_path = "wine.csv"
    training_data = read_csv(data_path)

    training_data = standardize(training_data)
    class_weights = learn_weights(training_data)
    print_weights(class_weights)







