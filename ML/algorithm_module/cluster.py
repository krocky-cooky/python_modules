import numpy as np


class Cluster:
    def __init__(self,data,index):
        self.data = np.array([data])
        self.size = 1
        self.index = index
        self.ids = [index]

    def merge(self,cluster,index):
        self.index = index
        data = cluster.data
        n_size = cluster.size
        self.size += n_size
        self.ids += cluster.ids
        self.data = np.concatenate([self.data,data])
        dist = np.sqrt(np.sum((self.center() - cluster.center())**2))
        return dist


    def center(self):
        c = np.sum(self.data,axis = 0)/self.size
        return c

class Clusters:
    def __init__(self):
        self.clusters = list()
        self.log = list()
        

    def fit(self,data):
        self.size = data.shape[0]
        self.index = self.size - 1
        for i in range(self.size):
            d = data[i]
            cluster = Cluster(d,i)
            self.clusters.append(cluster)
        
        self.map = [list() for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                dist = np.sqrt(np.sum((data[i]-data[j])**2))
                self.map[i].append(dist)
        
        while self.size > 1:
            distance_list = list()
            for i in range(self.size-1):
                for j in range(i+1,self.size):
                    distance = self.distance(i,j)
                    distance_list.append([distance,[i,j]])
            
            sorted_distance_list = sorted(distance_list,key=lambda x : x[0])
            [a,b] = sorted_distance_list[0][1]
            print('number of clusters : {} ,{} and {} are merged'.format(self.size,a,b))
            self.index += 1
            tmp_a = self.clusters[a].index
            tmp_b = self.clusters[b].index
            dist = self.clusters[a].merge(self.clusters[b],self.index)
            self.log.append([
                tmp_a,
                tmp_b,
                sorted_distance_list[0][0],
                self.clusters[a].size
            ])
            tmp = self.clusters
            self.clusters = tmp[:b] + tmp[b+1:]
            self.size -= 1
        
        self.result = np.array(self.log)
        return self.result

        

    def distance(self,i,j):
        distance_list = list()
        for id_i in self.clusters[i].ids:
            for id_j in self.clusters[j].ids:
                distance_list.append(self.map[id_i][id_j])
        return distance_list

class Centroid(Clusters):
    def distance(self,i,j):
        g_i = self.clusters[i].center()
        g_j = self.clusters[j].center()

        dist = np.sum((g_i-g_j)**2)
        dist = np.sqrt(dist)
        return dist

class Single(Clusters):
    def distance(self,i,j):
        distance_list = super().distance(i,j)
        distance_list.sort()
        return distance_list[0]

class Complete(Clusters):
    def distance(self,i,j):
        distance_list = super().distance(i,j)
        distance_list.sort()
        distance_list.reverse()
        return distance_list[0]

class Average(Clusters):
    def distance(self,i,j):
        distance_list = super().distance(i,j)
        ave = np.sum(np.array(distance_list))/len(distance_list)
        return ave

class Ward(Clusters):
    def distance(self,i,j):
        g_i = self.clusters[i].center()
        g_j = self.clusters[j].center()
        sz_i = self.clusters[i].size
        sz_j = self.clusters[j].size


        dist = sz_i*sz_j/(sz_i + sz_j)
        dist *= np.sum((g_i - g_j)**2)
        dist = np.sqrt(dist)
        return dist
