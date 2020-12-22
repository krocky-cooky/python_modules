import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),'neural_module'))

from functions import euler_loss,cross_entropy_loss
from layer import hiddenLayer,inputLayer,outputLayer

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import pickle


class neuralNetwork:
    SETTINGS = {
        'activation' : {
            'hidden' : 'Relu',
            'output' : 'identity'
        },
        'optimize_initial_weight' : True,
    }

    def __init__(
        self,
        learning_rate = 0.1,
        epoch = 20000,
        batch_size = 100,
        loss_func="euler",
        optimizer = 'normal',
        log_frequency = 100,
        mu = 0.5
    ):
        self.layers = list()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.loss_list = list()
        self.acc_list = list()
        self.loss_func = loss_func
        self.log_freq = log_frequency
        self.cmap = plt.get_cmap('tab10')
        self.mu = mu
        self.optimizer = optimizer
        self.trained = False
        self.SETTINGS = neuralNetwork.SETTINGS
        

    def set_layer(self,layer_list):
        """
        layer_listに層のサイズを入力してニューラルネットワークの層構造を定義する。
        ex) layer_list = [64,[50,40],10]
        """
        self.layers = list()
        input_size = layer_list[0]
        output_size = layer_list[2]
        hidden_layers = layer_list[1] 
        
        input_layer = inputLayer(input_size)
        self.layers.append(input_layer)
        former = input_size
        for sz in hidden_layers:
            layer = hiddenLayer(
                input_size=former,
                output_size=sz,
                learning_rate=self.learning_rate,
                activation=neuralNetwork.SETTINGS['activation']['hidden'],
                optimize_initial_weight = neuralNetwork.SETTINGS['optimize_initial_weight'],
                optimizer = self.optimizer,
                mu = self.mu
            )
            self.layers.append(layer)
            former = sz

        output_layer = outputLayer(
            input_size=former,
            output_size=output_size,
            activation=neuralNetwork.SETTINGS['activation']['output'],
            learning_rate=self.learning_rate,
            optimize_initial_weight = neuralNetwork.SETTINGS['optimize_initial_weight'],
            optimizer = self.optimizer,
            mu = self.mu
        )
        self.layers.append(output_layer)

        print('<< successfully layers are updated >>')
    
    

    def predict(self,input):
        """
        予測メソッド
        """
        vector = input
        for layer in self.layers:
            vector = layer.process(vector)
        return vector

    def loss(self,y,t):
        """
        誤差を計算
        """
        if self.loss_func == 'euler':
            res = euler_loss(y,t)
        elif self.loss_func == 'cross_entropy':
            res = cross_entropy_loss(y,t)
        return res
    
    def dif(self,y,t):
        """
        誤差の逆伝播
        """
        if self.loss_func == 'euler':
            res = euler_loss(y,t,div = True)
        elif self.loss_func == 'cross_entropy':
            res = cross_entropy_loss(y,t,div = True)
        return res


    def backword_propagation(self,y,t):
        dif = self.dif(y,t)
        layers = self.layers[1:]
        for layer in reversed(layers):
            layer.update_delta(dif)
            dif = layer.send_backword()
            layer.update_weight()


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
            y = self.predict(x_batch)
            losses = self.loss(y,t_batch)
            y_sub = np.argmax(y,axis = 1)
            t_sub = np.argmax(t_batch,axis = 1)
            acc = np.sum(y_sub == t_sub)/float(batch_size)
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

            self.backword_propagation(y,t_batch)

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

        if not os.path.exists('./logs'):
            os.mkdir('logs')
        path = os.path.join(os.getcwd() ,'logs/' + file_name)
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
        path = os.path.join(os.getcwd(),'logs/' + file)
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
