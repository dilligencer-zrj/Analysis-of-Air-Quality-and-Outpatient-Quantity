#coding=utf-8
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from dataset import dataset,dataset2
from network import LSTM,MultiHeadAttention,CNN,RNN,GRU,MHA
import os
model_dir = '/home/liutao/project/Analysis-of-Air-Quality-and-Outpatient-Quantity/ckpt/'

TIME_STEP = 128
INPUT_SIZE = 16
HIDDEN_SIZE = 32
LR = 0.01
#EPOCH = 1000
EPOCH = 100

#rnn = LSTM(INPUT_SIZE=INPUT_SIZE,HIDDEN_SIZE=HIDDEN_SIZE)
#rnn = RNN(INPUT_SIZE=INPUT_SIZE, HIDDEN_SIZE=HIDDEN_SIZE)
#rnn = MultiHeadAttention(8,'RNN',64,64)

args = {
    'window' : 24 * 7,
    'highway_window' : 0,
    'dropout' : 0.2,
    'output_fun' : 'sigmoid',
    'input_size' : INPUT_SIZE,
    'hidden_size' : HIDDEN_SIZE,
    'rnn_layers' : 1,
    'd_k' : 64,
    'd_v' : 64,
    'n_head' : 8,
    'CNN_kernel' : 6
}
#rnn = CNN(args=args)
#rnn = GRU(args=args)
rnn = MHA(args=args)


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

LoadModel = False


def train(train_loader,num_e):
    torch.manual_seed ( 1 )
    if LoadModel:
        checkpoint = torch.load(model_dir + '{}.ckpt'.format(num_e))
        rnn.load_state_dict(checkpoint['state_dict'])
        print ('Loading model~~~~~~~~~~', num_e)

    for e in range(EPOCH):
        print ( 'epoch>>>>>>> ', e )
        rnn.train()
        for i,(data,label) in enumerate(train_loader):
            data = data.view(-1,1,INPUT_SIZE)
            label = label.view(-1,1)
            data = data.type(torch.FloatTensor)
            label = label.type ( torch.FloatTensor )
            data = Variable(data)
            label = Variable(label)

            prediction = rnn(data)
            # prediction = prediction.view(-1,1)[-1]
            # loss = loss_func ( prediction, label[0] )
            prediction = prediction.view ( -1, 1 )
            loss = loss_func ( prediction, label )
            print (loss.data.numpy())
            optimizer.zero_grad ( )
            loss.backward ( )
            optimizer.step ( )

        if (e + 1) % 10 == 0:
            state_dict = rnn.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save(
                {
                    'epoch': e,
                    'save_dir': model_dir,
                    'state_dict': state_dict,
                }, os.path.join(model_dir, '%d.ckpt' % e))



def val(val_loader,e=0):
    checkpoint = torch.load ( model_dir + '%d.ckpt' % e )
    rnn.load_state_dict ( checkpoint['state_dict'] )
    rnn.eval ( )
    result=[]
    target = []
    bias = []
    for i, (data, label) in enumerate ( val_loader ):
        data = data.view ( -1, 1, INPUT_SIZE )
        label = label.view ( -1 )
        data = data.type ( torch.FloatTensor )
        label = label.type ( torch.FloatTensor )
        data = Variable ( data )


        prediction = rnn ( data )

        # prediction = prediction[-1].view ( -1 )
        prediction = prediction.view ( -1 )
        result.extend(prediction.data.numpy())
        target.extend(label.data.numpy())
        bias.extend(abs(prediction.data.numpy() -label.data.numpy()) ** 2)

    # for i in range(len(result)):
    #     print(result[i],  target[i])
    print('average bias>>>>>' , sum(bias)*1.0 / len(bias))

    plt.plot(range(len(result)),result,'r')
    plt.plot(range(len(result)),target,'b')
    plt.show()



if __name__ == '__main__':
    file_1 = '/home/lixiaoyu/project/airQuality/Analysis-of-Air-Quality-and-Outpatient-Quantity/file/weather.csv'
    file_2 = '/home/lixiaoyu/project/airQuality/Analysis-of-Air-Quality-and-Outpatient-Quantity//file/outpatient.csv'
    dir = '/home/lixiaoyu/project/airQuality/Analysis-of-Air-Quality-and-Outpatient-Quantity/preprocessed_data'

    # train_dateset = dataset(file_1=file_1,file_2=file_2,phase='train')
    # train_loader = DataLoader(train_dateset,batch_size=TIME_STEP,shuffle=False)

    train_dateset = dataset2(dir = dir,phase='train')
    train_loader = DataLoader(train_dateset,batch_size=TIME_STEP,shuffle=False)

    train(train_loader=train_loader,num_e=0)
    val_dateset = dataset2 ( dir = dir, phase='val' )
    val_loader = DataLoader ( val_dateset, batch_size=TIME_STEP, shuffle=False )
    for i in range(9,EPOCH,10):
        # print(i)
        val ( val_loader=val_loader, e=i )

