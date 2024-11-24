# Code with some modifications from https://github.com/yanaiela/amnesic_probing/tree/a96e3067a9f9918099015a7173c94830584bbe61
import gc

import numpy as np
import torch
import wandb

np.random.seed(10)


class PytorchClassifier:
    def __init__(self, m: torch.nn.Module, device: str):
        self.m = m  # .to(device)
        self.device = device

    def eval(self, x_dev: np.ndarray, y_dev: np.ndarray) -> float:
        x_dev = torch.tensor(x_dev).float()  # .to(self.device)
        y_dev = torch.tensor(y_dev)  # .to(self.device)
        test_dataset = torch.utils.data.TensorDataset(x_dev, y_dev)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4096,
                                                 shuffle=False)
        acc = self._eval(testloader)
        print("Eval accuracy: ", acc)
        del x_dev
        del y_dev
        del testloader
        gc.collect()
        return acc

    def _eval(self, testloader: torch.utils.data.DataLoader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                vectors, labels = data
                outputs = self.m(vectors)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print('Accuracy of the network on test: %d %%' % (
        #         100 * correct / total))
        return correct / total

    def get_probs(self, x: np.ndarray, y) -> np.ndarray:
        X = torch.tensor(x).float()  # .to(self.device)
        Y = torch.tensor(y)  # .to(self.device)
        test_dataset = torch.utils.data.TensorDataset(X, Y)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4096,
                                                 shuffle=False)
        probs = self._get_probs(testloader)
        # print("Eval accuracy: ", acc)
        return probs

    def _get_probs(self, testloader: torch.utils.data.DataLoader):
        softmax_logits = []
        with torch.no_grad():
            for data in testloader:
                vectors, labels = data
                outputs = self.m(vectors)
                probs = torch.softmax(outputs.data, dim=1)
                softmax_logits.append(probs.cpu().numpy())

        return np.array(softmax_logits)

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_dev: np.ndarray, y_dev: np.ndarray, epochs=1, save_path=None,
              use_wandb: bool = False):

        train_batch_size = 32
        dev_batch_size = 2048

        selectivity_results = {}
        print("Starts training....")

        x_train = torch.tensor(x_train)  # .to(self.device)
        y_train = torch.tensor(y_train)  # .to(self.device)
        x_dev = torch.tensor(x_dev)  # .to(self.device)
        y_dev = torch.tensor(y_dev)  # .to(self.device)

        print("Tensor Dataset")
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                                   shuffle=True)
        dev_dataset = torch.utils.data.TensorDataset(x_dev, y_dev)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=dev_batch_size,
                                                 shuffle=False)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.m.parameters(), lr=0.0001)

        print("Evaluation")
        acc = self._eval(dev_loader)
        best_acc = -1

        print("Dev accuracy before training: ", acc)

        if save_path:
            torch.save(self.m, save_path)

        selectivity_results['selectivity_dev_accuracy_before'] = acc

        print("Run epochs")
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.m(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            acc = self._eval(dev_loader)
            print("Dev acc during training:", acc)

            if acc > best_acc:
                best_acc = acc

                if save_path:
                    print("New best dev acc reached. Saving model to", save_path)
                    torch.save(self.m, save_path)

        print('Finished Training')

        acc = self._eval(dev_loader)

        selectivity_results['selectivity_dev_accuracy_after'] = acc
        selectivity_results['train_batch_size'] = train_batch_size
        selectivity_results['dev_batch_size'] = dev_batch_size

        print("Dev accuracy after training: ", acc)

        del x_train
        del y_train
        del x_dev
        del y_dev
        del train_dataset
        del train_loader
        del dev_dataset
        del dev_loader
        del criterion
        del optimizer

        gc.collect()

        return selectivity_results
