import numpy as np


class Cluster:
    def __init__(self,data,index):
        self.data = np.array([data])
        self.size = 1
        self.index = index

    def merge(self,cluster,index):
        self.index = index
        data = cluster.data
        n_size = cluster.size
        self.size += n_size
        self.data = np.concatenate([self.data,data])
        dist = np.sqrt(np.sum((self.center() - cluster.center())**2))
        return dist

    def calc_all(self,cluster):
        dist_list = list()
        for d_self in self.data:
            for d_target in cluster.data:
                dist = np.sqrt(np.sum((d_self-d_target)**2))
                dist_list.append(dist)
        return dist_list

    def center(self):
        c = np.sum(self.data,axis = 0)/self.size
        return c

    def distance(self,cluster):
        pass

    

class Shortest(Cluster):
    def distance(self,cluster):
        dist_list = self.calc_all(cluster)
        dist_list.sort()
        return dist_list[0]

class Longest(Cluster):
    def distance(self,cluster):
        dist_list = self.calc_all(cluster)
        dist_list.sort()
        dist_list.revese()
        return dist_list[0]

class Average(Cluster):
    def distance(self,cluster):
        dist_list = calc_all(cluster)
        ave = np.sum(np.array(dist_list))/len(dist_list)
        return ave

class Ward(Cluster):
    def distance(self,cluster):
        g_self = self.center()
        g_target = cluster.center()
        dist = self.size*cluster.size/(self.size + cluster.size)
        dist *= np.sum((g_self - g_target)**2)
        return dist
