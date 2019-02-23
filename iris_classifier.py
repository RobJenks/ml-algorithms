from random import randint
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def iris_classification():
    data = read_data()

    print("Dataset dimensions: {}\n".format(data.shape))
    print("Dataset head:\n{}\n".format(data.head(3)))

    print("Statistics:\n{}\n".format(data.describe()))

    print("Distribution by class:\n{}\n".format(data.groupby('class').size()))

    # 80/20 training to validation split
    training, validation = construct_datasets(data, 0.2)
    print("Training set: {}, Validation set: {}".format(training.shape, validation.shape))

    # Training and comparison of model results
    training_results = train_models(training)

    print("\nTraining results:")
    for k, v in training_results.items():
        print("> {}: Mean={:.3f}, StdDev={:.3f}".format(k, v.mean(), v.std()))

    # Verify KNN model against validation set
    print("\nValidation of K-Nearest Neighbours model predictions against validation set:\n")
    train_x, train_y = training.values[:, 0:4], training.values[:, 4]
    valid_x, valid_y = validation.values[:, 0:4], validation.values[:, 4]

    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)

    predictions = knn.predict(valid_x)
    print("Accuracy score: {}\n".format(accuracy_score(valid_y, predictions)))
    print("Confusion matrix:\n\n{}\n".format(confusion_matrix(valid_y, predictions)))
    print("Classification report:\n\n{}\n".format(classification_report(valid_y, predictions)))


def read_data():
    return pandas.read_csv('data/iris.csv',
                           names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])


def construct_datasets(data, validation_pc):
    return model_selection.train_test_split(data, test_size=validation_pc, random_state=randint(0, 2**32))


def train_models(training_set):
    x = training_set.values[:, 0:4]
    y = training_set.values[:, 4]
    random_seed = randint(0, 2**32)     # Same seed for all models -> same training set applied to each

    models = [
        ('Logistic regression', LogisticRegression(solver='liblinear', multi_class='ovr')),
        ('Linear discriminant analysis', LinearDiscriminantAnalysis()),
        ('K-Nearest Neighbours', KNeighborsClassifier()),
        ('Classification and Regression Trees', DecisionTreeClassifier()),
        ('Gaussian Naive Bayes', GaussianNB()),
        ('Support Vector Machine', SVC(gamma='auto'))
    ]

    # 10-fold cross validation; train on 9/10 and test on 1/10 of data, for all such combinations of train/test
    selection = model_selection.KFold(n_splits=10, random_state=random_seed)

    return {name: model_selection.cross_val_score(model, x, y, cv=selection, scoring='accuracy')
            for name, model in models}


def plot_data(data):
    data.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    data.hist()
    plt.show()

    scatter_matrix(data)
    plt.show()


# Entry point
if __name__ == "__main__":
    main()
