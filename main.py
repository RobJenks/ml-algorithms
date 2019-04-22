import sys
import gradient_descent
import kmeans_clustering
import naive_bayes
import iris_classifier


def content():
    return [
        # Linear regression via gradient descent and minimising squared error cost function
        ('gradient-descent', gradient_descent.run_linear_regression_gradient_descent),

        # K-Means clustering of arbitrarily-dimensioned datasets
        ('kmeans-clustering', kmeans_clustering.run_kmeans_clustering),

        # Naive-Bayes Gaussian classification
        ('naive-bayes', naive_bayes.run_naive_bayes),

        # Supervised learning models applied to iris classification dataset
        ('iris-classification', iris_classifier.iris_classification)
    ]


def main():
    [process() for (key, process) in content() if key in sys.argv]


# Entry point
if __name__ == "__main__":
    main()
