import sys
import gradient_descent
import iris_classifier


def content():
    return [
        # Linear regression via gradient descent and minimising squared error cost function
        ('gradient-descent', gradient_descent.run_linear_regression_gradient_descent),

        # Supervised learning models applied to iris classification dataset
        ('iris-classification', iris_classifier.iris_classification)
    ]


def main():
    for id, process in content():
        if id in sys.argv:
            process()


# Entry point
if __name__ == "__main__":
    main()
