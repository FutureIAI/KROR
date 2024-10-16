import copy
import torch
import torch.nn as nn
from scipy.stats import kurtosis, skew
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from myNetwork import network
from iCIFAR100 import iCIFAR100



class protoAugSSL:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc*4, feature_extractor)
        self.radius = 0
        self.prototype = None
        self.class_label = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.train_dataset = iCIFAR100('./dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('./dataset', test_transform=self.test_transform, train=False, download=True)
        self.train_loader = None
        self.test_loader = None
        self.constcos = 0.9
        self.cos_a = 0.9
        self.inh = 32
        self.similar_t = []

        self.learning_rate = 0.001
        self.cos_all =[]
        self.threshold = 0

    
    def map_new_class_index(self, y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    def setup_data(self, shuffle, seed):
        train_targets = self.train_dataset.targets
        test_targets = self.test_dataset.targets
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = range(len(order))
        self.class_order = order
        print(100*'#')
        print(self.class_order)

        self.train_dataset.targets = self.map_new_class_index(train_targets, self.class_order)
        self.test_dataset.targets = self.map_new_class_index(test_targets, self.class_order)

    def beforeTrain(self, current_task):
        self.model.eval()
        if current_task == 0:
            classes = [0, self.numclass]
        else:
            classes = [self.numclass-self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if current_task > 0:
            self.model.Incremental_learning(4*self.numclass)
        self.model.train()
        self.model.to(self.device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader

    def train(self, current_task, old_class=0):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-4)
        scheduler = StepLR(opt, step_size=45, gamma=0.1)
        accuracy = 0
        for epoch in range(self.epochs):
            scheduler.step()
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)

                # self-supervised learning based label augmentation
                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, 32, 32)
                target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)
                opt.zero_grad()
                loss = self._compute_loss(images, target, old_class, epoch)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % self.args.print_freq == 0:
                accuracy = self._test(self.test_loader)
                print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
        self.count = 0
        self.protoSave(self.model, self.train_loader, current_task)


    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        lable_total = dict()
        lable_correct = dict()
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            outputs = outputs[:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
            labels = np.array(labels.cpu())
            predicts = np.array(predicts.cpu())
            for i in range(len(labels)):
                if labels[i] not in lable_total:
                    lable_total[labels[i]] = 0
                    lable_correct[labels[i]] = 0
                lable_total[labels[i]] += 1
                if labels[i] == predicts[i]:
                    lable_correct[labels[i]] += 1
        sorted(lable_correct.keys())
        sorted(lable_total.keys())
        labels_accuracy = [lable_correct[i] / lable_total[i] for i in range(len(lable_correct))]
        print(labels_accuracy)
        if self.numclass != self.args.fg_nc:
            print("new")
            print(sum(labels_accuracy[-self.task_size:])/self.task_size)
            print("old")
            print(sum(labels_accuracy[:-self.task_size])/(len(labels_accuracy)-self.task_size))
        accuracy = correct.item() / total
        self.model.train()
        return accuracy
    def p_ball(self, feature, c=1.0):
        norm = torch.norm(feature, p = 2, dim = -1, keepdim=True)
        rie_feature = (1/(1+c*norm**2)) * feature
        return rie_feature
    def _compute_loss(self, imgs, target, old_class=0, epoch=0):
        output = self.model(imgs)
        output, target = output.to(self.device), target.to(self.device)
        loss_cls = nn.CrossEntropyLoss()(output/self.args.temp, target.long())
        if self.old_model is None:
            return loss_cls
        else:
            feature = self.model.feature(imgs)
            feature_old = self.old_model.feature(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)
            feature = self.p_ball(feature)
            feature = feature.cpu().detach().numpy()
            loss_inhAug = torch.tensor(0).to(self.device)
            label_similar = {key: 0 for key in range(old_class)}
            prototype = np.array(self.prototype)
            prototype_dict = {key: 0 for key in range(old_class)}
            target = target.cpu().numpy().tolist()
            class_inherit = {key: [] for key in list(set(target))}
            total_t = 0
            similar_t = 0
            for j in range(len(prototype)):
                for i in range(feature.shape[0]):
                    cos = np.dot(prototype[j], feature[i]) / (np.linalg.norm(prototype[j]) * np.linalg.norm(feature[i]))
                    if cos < self.cos_a:
                        prototype_dict[j] += 1
                        total_t += 1
                    if cos >= self.cos_a:
                        label_similar[j] += 1
                        class_inherit[target[i]].append(j)
                        similar_t += 1
            normalized_f = {key:[] for key in list(set(target))}
            for i in range(feature.shape[0]):
                normalized_f[target[i]].append(torch.from_numpy(np.asarray((feature[i]))).to(self.device))

            target_list = list(set(target))
            list_index = {key : [] for key in target_list}
            for index, item in enumerate(target):
                list_index[item].append(index)
            class_dimension = {key : [] for key in target_list}
            class_importance = []
            for i in target_list:
                feature_message = []
                for j in list_index[i]:
                    feature_message.append(feature[j])
                class_variance = (np.var(feature_message, axis=0))
                class_skewness = skew(feature_message, axis=0)
                class_kurtosis = kurtosis(feature_message, axis=0)

                var_norm = class_variance/np.max(class_variance)
                skew_norm = np.abs(class_skewness)/np.max(np.abs(class_skewness))
                kurt_norm = class_kurtosis/np.max(class_kurtosis)

                importance_score = (0.5 * var_norm + 0.3 * skew_norm + 0.2 * kurt_norm)

                for other_class in target_list:
                    if other_class != i:
                        other_feature_massage = np.array([feature[k] for k in list_index[other_class]])
                        mean_new_class = np.mean(feature_message, axis=0)
                        std_new_class = np.std(feature_message, axis=0)
                        mean_other_class = np.mean(other_feature_massage, axis=0)
                        std_other_class = np.std(other_feature_massage, axis=0)
                        kl_scores = np.log(std_other_class/std_new_class) + (std_new_class**2 + (mean_new_class-mean_other_class)**2)/(2*std_other_class**2) -0.5
                        importance_score += 1.0*kl_scores * (1 / len(target_list))
                class_importance.append(importance_score)
                median = np.sort(importance_score)[self.inh]
                for k in range(len(importance_score)):
                    if importance_score[k] <= median:
                        class_dimension[i].append(k)
            inh_f = copy.deepcopy(feature)
            for x,i in enumerate(target_list):
                inh_aug = []
                inh_aug_label = []
                if len(class_inherit[i]) > 0:
                    for j in list_index[i]:
                        selected_prototype = np.random.choice(class_inherit[i])
                        for k in range(len(class_dimension[i])):
                            similarity = 1 - abs(self.prototype[selected_prototype][k] - feature[j][k])
                            replace_prob = (class_importance[x][k] + similarity) / 2
                            if np.random.rand() < replace_prob:
                                inh_f[j][k] = self.prototype[selected_prototype][k]
                        inh_aug.append(inh_f[j])
                        inh_aug_label.append(i)
                        if len(inh_aug) >= self.args.batch_size:
                            break
                    inh_aug = torch.from_numpy(np.float32(np.asarray(inh_aug))).float().to(self.device)
                    inh_aug_label = torch.from_numpy(np.asarray(inh_aug_label)).to(self.device)
                    soft_inh_aug = self.model.fc(inh_aug)
                    loss_inhAug = loss_inhAug + nn.CrossEntropyLoss()(soft_inh_aug / self.args.temp, inh_aug_label.long())

            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))
            for _ in range(self.args.batch_size):
                np.random.shuffle(index)
                temp = self.prototype[index[0]] + np.random.normal(0, 1, 512) * self.radius
                proto_aug.append(temp)
                proto_aug_label.append(4*self.class_label[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            soft_feat_aug = self.model.fc(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug/self.args.temp, proto_aug_label.long())

            return loss_cls + 20 * loss_protoAug + self.args.kd_weight * loss_kd + 2 * loss_inhAug

    def afterTrain(self):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()

    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(loader):
                feature = model.feature(images.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task != 0:
            prototypes = prototype + self.prototype
            mean_prototype = np.mean(prototypes, axis=0)
            diff = prototypes - mean_prototype
            cov_matrix = np.dot(diff.T, diff) / (len(prototypes[0]) - 1)
            eigenvalues, eigenvectors = np.linalg.eigh((cov_matrix))
            Q, _ = np.linalg.qr(eigenvectors)
            W = Q.T
            W_inv = torch.inverse(torch.from_numpy(W))

            for proto in prototype:
                count = 0
                for i in range(len(self.prototype)):
                    proto_old = self.prototype[i]
                    smilarity = np.dot(proto, proto_old) / (np.linalg.norm(proto) * np.linalg.norm(proto_old))
                    if smilarity > self.cos_a:
                        count += 1
                        old_rie = W @ proto_old
                        new_rie = W @ proto
                        diff = new_rie - old_rie
                        D = np.sqrt(diff ** 2)
                        G = np.dot(W, np.dot(D, W.T))
                        diff = self.prototype[i] - proto
                        G_prime = G + count * np.outer(diff, diff.T)
                        o_proto = torch.matmul(W_inv,torch.from_numpy(np.dot(G_prime, self.prototype[i]))).numpy()
                        a = 0.01/(len(prototypes)*1.5)
                        self.prototype[i] = (self.prototype[i] + a * o_proto)/(1+a)

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
        else:
            self.prototype = prototype + self.prototype
            self.class_label = class_label + self.class_label

        cos_a = []
        for proto in self.prototype:
            c = []
            for i in range(len(self.prototype)):
                c.append(np.dot(proto, self.prototype[i]) / (np.linalg.norm(proto) * np.linalg.norm(self.prototype[i])))
            cos_a.extend(c)
        cos_a.sort()
        self.cos_all = copy.deepcopy(cos_a)
        l = len(self.prototype)
        a = 0-l
        cos_a = cos_a[:a]
        if cos_a[-1] > self.constcos:
            index = self.first_large(cos_a, self.constcos)
            cos_a = cos_a[index:]
            if len(cos_a) > current_task:
                self.cos_a = cos_a[int(len(cos_a) / 3.5)]
        else:
            self.cos_a = self.constcos

    def first_large(self, nums, target):
        left, mid = 0, -1
        right = len(nums)
        while left <= right:
            mid = int((right - left) / 2 + left)
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return mid

