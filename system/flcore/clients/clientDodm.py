import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientDodm(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.alpha = args.alpha  # 1.0
        num_layers = len(list(self.model.parameters()))
        self.alpha = [args.alpha]*num_layers

        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)

        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        print(self.sample_per_class.detach().int())

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        self.model.train()
        self.model_per.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                self.optimizer.zero_grad()
                output = self.model(x)
                loss_bsm = self.loss(output, y)
                loss_bsm.backward()
                self.optimizer.step()

                rep = self.model_per.base(x)
                out_g = self.model_per.head(rep)
                loss = self.loss(out_g, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step()

                self.alpha_update()


        for idx, (lp, p) in enumerate(zip(self.model_per.parameters(), self.model.parameters())):
            lp.data = (1 - self.alpha[idx]) * p + self.alpha[idx] * lp

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model == None:
            model = self.model_per
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model_per.base(x)
                output = self.model_per.head(rep)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model_per.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model_per.base(x)
                out_g = self.model_per.head(rep)
                output = out_g
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]


        return losses, train_num


    def alpha_update(self):
        for idx, (l_params, p_params) in enumerate(zip(self.model.parameters(), self.model_per.parameters())):
            alpha = self.alpha[idx]
            grad_alpha = 0

            dif = p_params.data - l_params.data
            grad = alpha * p_params.grad.data + (1-alpha) * l_params.grad.data
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))  # .T是转置的意思，.dot是点乘的意思

            grad_alpha += 0.01*alpha
            alpha = alpha - self.learning_rate*grad_alpha
            alpha = np.clip(alpha.item(), 0.0,1.0)
            self.alpha[idx] = alpha

