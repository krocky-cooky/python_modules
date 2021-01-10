import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),'neural_module'))

from losses import SumSquare,CrossEntropy
from layer import hiddenAndOutputLayer,inputLayer
from score import *

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import pickle


class neuralNetwork:
    SETTINGS = {
        'activation' : {
            'hidden' : 'relu',
            'output' : 'identity'
        },
    }

    def __init__(
        self,
        learning_rate = 0.1,
        epoch = 20000,
        batch_size = 100,
        loss_func="square",
        optimizer = 'normal',
        optimize_initial_weight = True,
        log_frequency = 100,
        mu = 0.5
    ):
        self.layers = list()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.loss_list = list()
        self.acc_list = list()
        self.loss_func = None
        self.log_freq = log_frequency
        self.cmap = plt.get_cmap('tab10')
        self.mu = mu
        self.optimizer = optimizer
        self.trained = False
        self.SETTINGS = neuralNetwork.SETTINGS
        self.optimize_initial_weight = optimize_initial_weight

        if loss_func == "square":
            self.loss_func = SumSquare()
        elif loss_func == "cross_entropy":
            self.loss_func = CrossEntropy()
        else :
            raise Exception('誤差関数が正しくありません')
        

    def set_layer(self,layer_list):
        """
        layer_listに層のサイズを入力してニューラルネットワークの層構造を定義する。
        ex) layer_list = [64,[50,40],10]
        """
        self.layers = list()
        input_size = layer_list[0]
        output_size = layer_list[2]
        hidden_layers = layer_list[1] 
        
        input_layer = inputLayer()
        self.layers.append(input_layer)
        former = input_size
        for sz in hidden_layers:
            layer = hiddenAndOutputLayer(
                input_size=former,
                output_size=sz,
                learning_rate=self.learning_rate,
                activation=self.SETTINGS['activation']['hidden'],
                optimize_initial_weight = self.optimize_initial_weight,
                optimizer = self.optimizer,
                mu = self.mu
            )
            self.layers.append(layer)
            former = sz

        output_layer = hiddenAndOutputLayer(
            input_size=former,
            output_size=output_size,
            activation=self.SETTINGS['activation']['output'],
            learning_rate=self.learning_rate,
            optimize_initial_weight = self.optimize_initial_weight,
            optimizer = self.optimizer,
            mu = self.mu
        )
        self.layers.append(output_layer)

        print('<< successfully layers are updated >>')
        return self
    
    

    def predict(self,input):
        """
        予測メソッド
        """
        vector = input
        for layer in self.layers:
            vector = layer(vector)
        return vector

    def loss(self,y,t):
        res = self.loss_func(y,t)
        return res
    

    def backward_propagation(self,y,t):
        delta = self.loss_func.backward(y,t)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)


    def train(self,x,t):
        pass


    def accuracy(self,x,t):
        pass

        
    def visualize(
        self,
        acc_bound = 0,
    ):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        t = np.linspace(1,self.epoch,self.epoch)
        plot1 = ax1.plot(
            t,
            self.acc_list,
            label = 'accuracy',
            c = self.cmap(0)
        )
        ax1.set_ylim([acc_bound,1.1])
        ax2 = ax1.twinx()
        plot2 = ax2.plot(
            t,
            self.loss_list,
            label = 'loss',
            c = self.cmap(1)
        )
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()  
        ax1.legend(
            h1+h2,
            l1+l2,
            loc='upper right'
        )
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('accuracy')
        ax2.set_ylabel('loss')
        plt.title('transition of accuracy and loss')
        plt.show()

    def save_json(self,file_name):
        if not self.trained:
            raise Exception

        """
        トレーニング済みニューラルネットワークを保存する。
        保存するもの・・・
        SETTINGS
        layer_list
        learning_rate
        epoch
        batch_size
        loss_func
        optimizer
        log_frequency
        mu
        weight
        loss_list
        acc_list
        """

        if not os.path.exists('./json_logs'):
            os.mkdir('logs')
        path = os.path.join(os.getcwd() ,'json_logs/' + file_name)
        file = {
            'SETTINGS' : self.SETTINGS,
            'constructor' : {
                'learning_rate' : self.learning_rate,
                'epoch' : self.epoch,
                'batch_size' : self.batch_size,
                'loss_func' : self.loss_func,
                'optimizer' : self.optimizer,
                'log_frequency' : self.log_freq,
                'mu' : self.mu
            },
            'layers' : [],
            'loss_list' : self.loss_list,
            'acc_list'  : self.acc_list
        }

        for i,layer in enumerate(self.layers):
            description = dict()
            if i == 0:
                description['constructor'] = {
                    'size' : layer.size
                }
            else:
                description['constructor'] = {
                    'input_size' : layer.weight.shape[0],
                    'output_size' : layer.weight.shape[1],
                    'activation' : layer.which_activation,
                    'learning_rate' : layer.learning_rate,
                    'optimizer' : layer.optimizer_name,
                }

                if layer.optimizer_name == "momentum":
                    description['constructor']['mu'] = layer.optimizer.mu

                description['weight'] = layer.weight.tolist()
                description['bias'] = layer.bias.tolist()
            
            file['layers'].append(description)
        
        json_data = json.dumps(file)


        with open(path,mode='wb') as f:
            f.write(json_data.encode('utf-8'))

    @classmethod
    def load_json(cls,file):
        default_setting = cls.SETTINGS
        path = os.path.join(os.getcwd(),'json_logs/' + file)
        with open(path,mode = 'rb') as f:
            byt = f.read()
        data = json.loads(byt.decode('utf-8'))
        cls.SETTINGS = data['SETTINGS']
        net = cls(**data['constructor'])
        
        for i,description in enumerate(data['layers']):
            if i == 0:
                layer = inputLayer(**description['constructor'])
                net.layers.append(layer)
            elif i == len(data['layers'])-1:
                layer = outputLayer(**description['constructor'])
                layer.weight = np.array(description['weight'])
                layer.bias = np.array(description['bias'])
                net.layers.append(layer)
            else:
                layer = hiddenLayer(**description['constructor'])
                layer.weight = np.array(description['weight'])
                layer.bias = np.array(description['bias'])
                net.layers.append(layer)

        net.acc_list = data['acc_list']
        net.loss_list = data['loss_list']
        net.trained = True
        cls.SETTINGS = default_setting
        print('successfully network was constructed!')
        return net

    def save(self,file_name):
        if not self.trained:
            raise Exception

        """
        トレーニング済みニューラルネットワークを保存する。
        保存するもの・・・
        SETTINGS
        layer_list
        learning_rate
        epoch
        batch_size
        loss_func
        optimizer
        log_frequency
        mu
        weight
        loss_list
        acc_list
        """

        if not os.path.exists('./logs'):
            os.mkdir('logs')
        path = os.path.join(os.getcwd() ,'logs/' + file_name)
        with open(path,mode = 'wb') as f:
            pickle.dump(self,f)
        

    @classmethod
    def load(cls,file):
        path = os.path.join(os.getcwd(),'logs/' + file)
        with open(path,mode = 'rb') as f:
            net = pickle.load(f)
        
        print('successfully network was constructed!')
        return net


class Classification(neuralNetwork):
    def train(self,x,t):
        self.loss_list = list()
        self.acc_list = list()
        train_size = x.shape[0]
        batch_size = self.batch_size
        start = time.time()
        for i in range(self.epoch):
            batch = np.random.choice(train_size,batch_size)
            x_batch = x[batch]
            t_batch = t[batch]
            losses = self.loss(y,t_batch)
            acc = self.accuracy(x_batch,t_batch)
            self.loss_list.append(losses)
            self.acc_list.append(acc)
            if i%self.log_freq == 0:
                elapsed = time.time() - start
                
                '''
                途中経過を表示
                '''
                word = '--------- epoch' + str(i) + ' ---------'
                print(word)
                print('loss : ' + str(losses))
                print('accuracy : ' + str(acc))
                print('time : {} [sec]'.format(elapsed))
                word = '-'*len(word)
                print(word + '\n')

            self.backward_propagation(y,t_batch)

        elapsed = time.time() - start
        train_acc = self.accuracy(x,t)  
        print('\n')
        print('<< All training epochs ended. >>')

        '''
        トレーニングセットの正答率とトレーニングにかかった時間を結果として表示する。
        '''
        word = '========= result ========='
        print(word)
        print('Elapsed time : {} [sec]'.format(elapsed))
        print('Train set accuracy : {}'.format(train_acc))
        word = '='*len(word)
        print(word)
        self.trained = True
        return (elapsed,train_acc)

    def accuracy(self,x,t):
        y = self.predict(x)
        y_sub = np.argmax(y,axis=1)
        t_sub = np.argmax(t,axis=1)
        acc = np.sum(y_sub == t_sub)/float(y.shape[0])
        return acc

class Regression(neuralNetwork):
    def __init__(
        self,
        learning_rate = 0.1,
        epoch = 20000,
        batch_size = 100,
        loss_func="square",
        optimizer = 'normal',
        optimize_initial_weight = True,
        log_frequency = 100,
        mu = 0.5,
        accuracy_function = 'r2_score'
    ):
        super().__init__(
            learning_rate,
            epoch,
            batch_size,
            loss_func,
            optimizer,
            optimize_initial_weight,
            log_frequency,
            mu
        )
        self.ac_fun = accuracy_function
        accuracy_functions = [
            'r2_score',
            'rmse',
            'mae'
        ]
        if not accuracy_function in accuracy_functions:
            raise Exception('精度関数が正しくありません')
        

    def train(self,x,t):
        self.loss_list = list()
        batch_size = self.batch_size
        train_size = x.shape[0]
        for i in range(self.epoch):
            batch = np.random.choice(train_size,batch_size)
            x_batch = x[batch]
            t_batch = t[batch]
            y = self.predict(x_batch)
            loss = self.loss(y,t_batch)
            acc = self.accuracy(y,t_batch)
            self.loss_list.append(loss)
            self.acc_list.append(acc)

            if i%self.log_freq == 0:
                print('epoch:{} , accuracy:{} , loss:{}'.format(
                    i+1,
                    acc,
                    loss
                ))

            self.backward_propagation(y,t_batch)

        print('========= result ========')
        print('accuracy : {}\nloss : {}'.format(
            self.accuracy(self.predict(x),t),
            self.loss(x,t)
        ))

        
    def accuracy(self,y,t):
        if self.ac_fun == 'r2_score':
            return r2_score(y,t)
        elif self.as_fun == 'rmse':
            return rmse(y,t)
        elif self.as_fun == 'mae':
            return mae(y,t)
        


