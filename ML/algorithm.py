import numpy as np
import math
import sys,os
from mpl_toolkits.mplot3d import Axes3D
import time

sys.path.append(os.path.join(os.path.dirname(__file__),'algorithm_module'))

import matplotlib.pyplot as plt
from functions import *
from cluster import Centroid,Single,Complete,Ward,Average

import heap
from scipy.cluster.hierarchy import dendrogram



class KNeighborsClassifier:
    def __init__(self,n_neighbors):
        self.n_neighbors = n_neighbors
        
    def fit(self,x,t):
        self.x = x
        self.t = t

    def predict(self,x):
        ans_list = list()
        for data in x:
            tmp_t = self.t
            tmp_x = self.x
            neighbors = dict()
            sub = list()
            dists = dist(data,self.x)
            for j in range(self.n_neighbors):
                argmin = np.argmin(dists)
                sub.append(tmp_t[argmin])
                if tmp_t[argmin] in neighbors.keys():
                    neighbors[tmp_t[argmin]] -= 1 
                else:
                    neighbors[tmp_t[argmin]] = -1
                dists = np.concatenate([dists[:argmin],dists[argmin+1:]])
                tmp_t = np.concatenate([tmp_t[:argmin],tmp_t[argmin+1:]])
                tmp_x = np.concatenate([tmp_x[:argmin],tmp_x[argmin+1:]])
            
            sorted_neighbors = sorted(neighbors.items(),key=lambda x:x[1])
            
            samp = sorted_neighbors[0][0]
            majority_list = [samp]
            num = sorted_neighbors[0][1]
            for i in range(1,len(sorted_neighbors)):
                if num != sorted_neighbors[i][1]:
                    break
                else : majority_list.append(sorted_neighbors[i][0])
            
            for i in range(len(sub)):
                if sub[i] in majority_list:
                    ret = sub[i]
                    break
            
            ans_list.append(ret)
        
        return np.array(ans_list)

    def accuracy(self,x,t):
        y = self.predict(x)
        acc = np.sum(y == t)/float(y.shape[0])

        return acc

    @classmethod
    def leave_one_out_accuracy(cls,x,t,n_neighbors):
        correct = 0
        for i in range(x.shape[0]):
            mask = np.array([j != i for j in range(x.shape[0])])
            tmp_x = x[mask]
            tmp_t = t[mask]
            clf = cls(n_neighbors = n_neighbors)
            clf.fit(tmp_x,tmp_t)
            mask = np.array([j == i for j in range(x.shape[0])])
            pred = clf.predict(x[mask])
            if pred[0] == t[i]:
                correct += 1
        
        acc = correct/float(x.shape[0])
        return acc
    


class KMeans:
    def __init__(self,n_clusters):
        self.n_clusters = n_clusters
        self.x = None
        self.classify = None
        self.represent = None
        self.done = False
        self.color_list = [
            'tab:blue',
            'tab:green',
            'tab:red',
            'y',
            'k',
            'tab:orange',
            'tab:purple',
            'tab:blown',
            'tab:pink',
            'tab:gray',
            'tab:cyan',
            'tab:olive'
        ]
        self.cmap = plt.get_cmap("tab10")

    def fit(self,x):
        self.x = x
        index = np.array([i for i in range(self.x.shape[0])])
        choice = np.random.choice(index,self.n_clusters,replace = False)
        represent = self.x[choice]
        classify = list()
        pre = None
        while pre != classify:
            next_represent = [list() for i in range(self.n_clusters)]
            pre = classify
            classify = list()
            for data in self.x:
                dists = dist(data,represent)
                argmin = np.argmin(dists)
                classify.append(argmin)
                next_represent[argmin].append(data)
            
            for i,data_list in enumerate(next_represent):
                next_represent[i] = np.sum(next_represent[i],axis = 0)/float(len(next_represent[i]))
            
            represent = np.array(next_represent)
        self.represent = represent
        self.classify = np.array(classify)
        self.done = True 

        print('<< successfully classified >>')
        return self.classify

    def visualize(self,argx = 0,argy = 1,target = np.array([])):
        if not self.done:
            print('please fit a data')
            return
        
        flag = False
        if target.shape[0]:
            classify = target
            
        else :
            classify = self.classify
            flag = True

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('argx : ' + str(argx))
        ax.set_ylabel('argy : ' + str(argy))
        for i in range(self.n_clusters):
            mask = classify == i
            scatter_x = self.x[mask][:,argx]
            scatter_y = self.x[mask][:,argy]
            ax.scatter(
                scatter_x,
                scatter_y,
                s=30,
                c=self.cmap(i)
            )
            represent_x = self.represent[i][argx]
            represent_y = self.represent[i][argy]
            if flag:
                ax.scatter(
                    represent_x,represent_y,
                    s = 200,
                    c = self.cmap(i),
                    marker = '^',
                    linewidth="2",
                    edgecolors='k',
                )
        #plt.savefig('{}-{}plot{}.png'.format(argx,argy,flag))
        plt.show()

    def visualize3D(self,argx = 0,argy = 1,argz = 2,target = np.array([])):
        if not self.done:
            print('please fit a data')
            return

        flag = False
        if target.shape[0]:
            classify = target
        else :
            classify = self.classify
            flag = True

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_xlabel('argx : ' + str(argx))
        ax.set_ylabel('argy : ' + str(argy))
        ax.set_zlabel('argz : ' + str(argz))

        for i in range(self.n_clusters):
            mask = classify == i
            scatter_x = self.x[mask][:,argx]
            scatter_y = self.x[mask][:,argy]
            scatter_z = self.x[mask][:,argz]
            ax.scatter(
                scatter_x,
                scatter_y,
                scatter_z,
                s=30,
                c=self.cmap(i)
            )
            represent_x = self.represent[i][argx]
            represent_y = self.represent[i][argy]
            represent_z = self.represent[i][argz]
            if flag:
                ax.scatter(
                    represent_x,
                    represent_y,
                    represent_z,
                    s = 200,
                    c = self.cmap(i),
                    marker = '^',
                    linewidth="2",
                    edgecolors='k'
                )

        #plt.savefig('{}-{}-{}plot{}.png'.format(argx,argy,argz,flag))
        plt.show()

    

class MultipleLinearRegression:
    def __init__(self):
        self.x = None
        self.t = None
        self.weight = None

    def fit(self,x,t):
        self.x = self.amend(x)
        self.t = t
        self.weight = np.dot(np.linalg.pinv(self.x),t)
        return self

    def predict(self,x):
        x = self.amend(x)
        ret = np.dot(x,self.weight)
        return ret

    def amend(self,x):
        con = np.array([1])
        sub = list()
        for data in x:
            data = np.concatenate([data,con])
            sub.append(data)
        return np.array(sub)

    def score(self,x,t):
        t_pred = self.predict(x)
        score = r2_score(t,t_pred)
        return score

class HierarchicClustering:
    def __init__(self,method = 'ward'):
        self.clusters = None
        if method == 'centroid':
            self.clusters = Centroid()
        elif method == 'single':
            self.clusters = Single()
        elif method == 'complete':
            self.clusters = Complete()
        elif method == 'average':
            self.clusters = Average()
        elif method == 'ward':
            self.clusters = Ward()

    def fit(self,data):
        self.result = self.clusters.fit(data)
        return self.result

    def visualize(self):
        dendrogram(self.result)
        plt.show()

class SoftmaxRegression:
    def __init__(self,iter,learning_rate,momentum = False,mu = 0):
        self.iter = iter
        self.learning_rate = learning_rate
        self.acc_list = list()
        self.delta_weigt = None
        self.momentum = momentum
        self.mu = mu

    def train(self,data,target,mu = 0.1):
        if not self.momentum:
            mu = 0
        self.acc_list = list()
        i_size = data.shape[1]
        o_size = target.shape[1]
        self.weight = np.zeros((i_size,o_size))
        self.delta_weight = np.zeros_like(self.weight)
        self.bias = np.zeros(o_size)
        self.input = data
        acc = None
        start = time.time()
        for i in range(self.iter):
            p = self.predict(data)
            d_weight = np.dot(self.input.T,(p-target))
            self.delta_weight = mu*self.delta_weight - self.learning_rate*d_weight
            self.weight += self.delta_weight
            pre_acc = acc
            acc = self.accuracy(data,target)
            self.acc_list.append(acc)
            print('iter => {}, train accuracy => {}'.format(
                i+1,
                acc
            ))
        elapsed_time = time.time()-start
        print('elapsed_time => ',elapsed_time)


    def predict(self,data):
        p = np.dot(data,self.weight) + self.bias
        c = np.max(p,axis=1,keepdims=True)
        exp_p = np.exp(p-c)
        sum_exp_p = np.sum(exp_p,axis = 1,keepdims=True)
        out = exp_p/sum_exp_p
        return out

    def accuracy(self,data,target):
        pred = self.predict(data)
        pred = np.argmax(pred,axis=1)
        target = np.argmax(target,axis = 1)
        accuracy = np.sum(target == pred)/target.shape[0]
        return accuracy

    def evaluation_matrix(self,data,target):
        pred = self.predict(data)
        n_class = target.shape[1]
        matrix = np.zeros((n_class,n_class))
        for i in range(data.shape[0]):
            matrix[np.argmax(pred[i]),np.argmax(target[i])] += 1

        v_sum = np.sum(matrix,axis=0)
        h_sum = np.sum(matrix,axis=1)

        precision = np.diag(matrix)/h_sum
        recall = np.diag(matrix)/v_sum
        F1 = 2*precision*recall/(precision + recall)
        ret = np.array([precision,recall,F1]).T
        return ret 

    def graph(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        t = np.linspace(1,self.iter,self.iter)
        ax.plot(t,self.acc_list)
        ax.set_xlabel('iter')
        ax.set_ylabel('accuracy')
        plt.title('transition of accuracy')
        plt.show()



class OptimizedHierarchicClustering:
    def __init__(self,method = 'ward'):
        dic = {
            'centroid' : 0,
            'single' : 1,
            'complete' : 2,
            'average' : 3,
            'ward':4
        }
        self.method = dic[method]
        self.done = False
        self.cmap = plt.get_cmap("tab10")

    def fit(self,data,n_class = -1):
        self.data = data
        self.heap = heap.Heap(data,self.method)
        self.result = list()
        while self.heap.size > 1:
            ret = self.heap.update()
            self.result.append(ret)
            if self.heap.size == n_class:
                self.classify = np.array(self.heap.getClassify())
                self.done = True
                print("got classificatin")
        self.result = np.array(self.result)
        
        return self.result
    
    def dendrogram(self):
        dendrogram(self.result)
        plt.show()

    def visualize(self,argx = 0,argy = 1,target = np.array([])):
        if not self.done:
            print('please fit a data')
            return

        if target.shape[0]:
            classify = target
            
        else :
            classify = self.classify

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('argx : ' + str(argx))
        ax.set_ylabel('argy : ' + str(argy))
        for i in range(self.data.shape[0]):
            mask = classify == i
            scatter_x = self.data[mask][:,argx]
            scatter_y = self.data[mask][:,argy]
            ax.scatter(
                scatter_x,
                scatter_y,
                s=30,
                c=self.cmap(i)
            )
        #plt.savefig('{}-{}plot{}.png'.format(argx,argy,flag))
        plt.show()

    def visualize3D(self,argx = 0,argy = 1,argz = 2,target = np.array([])):
        if not self.done:
            print('please fit a data')
            return

        if target.shape[0]:
            classify = target
        else :
            classify = self.classify

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_xlabel('argx : ' + str(argx))
        ax.set_ylabel('argy : ' + str(argy))
        ax.set_zlabel('argz : ' + str(argz))

        for i in range(self.data.shape[0]):
            mask = classify == i
            scatter_x = self.data[mask][:,argx]
            scatter_y = self.data[mask][:,argy]
            scatter_z = self.data[mask][:,argz]
            ax.scatter(
                scatter_x,
                scatter_y,
                scatter_z,
                s=30,
                c=self.cmap(i)
            )

        #plt.savefig('{}-{}-{}plot{}.png'.format(argx,argy,argz,flag))
        plt.show()