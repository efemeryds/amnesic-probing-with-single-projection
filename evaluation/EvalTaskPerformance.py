import numpy as np
import gc
np.random.seed(10)

class EvalTaskPerformance:

    def __init__(self, model, parameters):
        self.classifier_class = model(**parameters)

    def learn_cls(self, x_train, y_train, x_dev, y_dev):
        # self.classifier_class.fit(x_train, y_train)

        batch_size = 100000
        for i in range(0, len(x_train), batch_size):
            X_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Train the classifier on the current batch
            self.classifier_class.partial_fit(X_batch, y_batch, classes=np.unique(y_train))

            del X_batch
            del y_batch
            gc.collect()

        acc = self.classifier_class.score(x_dev, y_dev)
        return acc

    def eval_task_performance(self, x_train, y_train, x_dev, y_dev, x_train_no_label, x_dev_no_label):
        task_results = {}

        # shuffled
        acc = self.learn_cls(x_train, y_train, x_dev, y_dev)

        # can it learn task without the information
        acc_inlp = self.learn_cls(x_train_no_label, y_train, x_dev_no_label, y_dev)

        task_results['task_acc_vanialla'] = acc
        task_results['task_acc_p'] = acc_inlp
        return task_results
