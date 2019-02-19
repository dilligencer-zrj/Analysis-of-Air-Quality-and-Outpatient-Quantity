# coding=utf-8
from torch.utils.data import DataLoader, Dataset
import csv
import numpy as np
from utils import cal_mean_var
division=[11000.0,1000.0]
dir = '/home/dilligencer/code/一附院时序数据回归/数据处理/'
TIME_STEP = 2


class dataset(Dataset):
    def __init__(self,file_1,file_2,phase = 'train'):
        weather_list = []
        outpatient_list = []
        with open ( file_1, 'r' ) as f:
            weather = csv.reader ( f )
            for row in weather:
                tmp=[]
                for i in row:
                    tmp.append(float(i))
                weather_list.append ( tmp )
        with open ( file_2, 'r' ) as f:
            outpatient = csv.reader ( f )
            for row in outpatient:
                tmp = []
                for i in row:
                    tmp.append ( float ( i ) )
                outpatient_list.append ( tmp )
        weather_array = np.array ( weather_list )
        outpatient_array = np.array ( outpatient_list )
        train_num = int(len(weather_array)*0.7)

        if phase == 'train':
            self.data = weather_array[:730]
            self.label = outpatient_array[:730]
            self.mean ,self.var = cal_mean_var(self.data)
        else:
            self.data = weather_array[730:]
            self.label = outpatient_array[730:]
            self.mean, self.var = cal_mean_var ( weather_array[:730] )


    def __getitem__(self, idx):
        vector = (self.data[idx] - self.mean) / (self.var + 1e-6)
        label = self.label[idx]
        return vector,label[1]/division[1]


    def __len__(self):
        return len(self.label)

class dataset2(Dataset):
    def __init__(self,dir,phase = 'train'):
        self.phase = phase
        breath_num = np.load ( dir + '/呼吸科门诊量.npy' )


        SO2 = np.load ( dir + '/SO2浓度.npy' )
        NO2 = np.load ( dir + '/NO2浓度.npy' )
        PM10 = np.load ( dir + '/PM10浓度.npy' )
        CO = np.load ( dir + '/CO浓度.npy' )
        O38h = np.load ( dir + '/O38h浓度.npy' )
        PM2_5 = np.load ( dir + '/PM2-5浓度.npy' )


        SO2 = SO2 / float ( SO2.max ( ) )
        NO2 = NO2 / float ( NO2.max ( ) )
        PM10 = PM10 / float ( PM10.max ( ) )
        CO = CO / float ( CO.max ( ) )
        O38h = O38h / float ( O38h.max ( ) )
        PM2_5 = PM2_5 / float ( PM2_5.max ( ) )
        time = []
        for i in np.arange ( 1, 50 ):
            time.append ( [i, i] )
        time = np.array ( time )
        time = time.reshape ( -1 )
        time = np.hstack ( (time, time, time) )
        Y = (breath_num ).astype ( int )

        print ( SO2[:98].mean ( ) )
        print ( SO2[98:196].mean ( ) )
        print ( SO2[196:].mean ( ) )

        SO2 = SO2 / float ( SO2.max ( ) )
        NO2 = NO2 / float ( NO2.max ( ) )
        PM10 = PM10 / float ( PM10.max ( ) )
        CO = CO / float ( CO.max ( ) )
        O38h = O38h / float ( O38h.max ( ) )
        PM2_5 = PM2_5 / float ( PM2_5.max ( ) )
        time = time / float ( time.max ( ) )
        print ( Y.max ( ) )
        Y = Y / float ( Y.max ( ) )
        print(Y.max())

        X = np.vstack([SO2, NO2, PM10, CO, PM2_5, O38h, time])
        # X = np.vstack ( [SO2, time] )
        X = X.transpose ( (1, 0) )
        length = len ( Y )


        if phase == 'train':
            self.data, self.label =np.concatenate((X[:int ( length * 1 / 3 )],X[int ( length * 2 / 3 ):])), np.concatenate((Y[:int ( length * 1 / 3 )],Y[int ( length * 2 / 3 ):]))

        elif phase == 'val':
            self.data, self.label =  X[int ( length * 1 / 3 ):int ( length * 2 / 3 )], Y[int ( length * 1 / 3 ):int ( length * 2 / 3 )]



    def __getitem__(self, idx):

        # batch_data = []
        # batch_label = []
        # if self.phase == 'train':
        #     for i in range(TIME_STEP):
        #         batch_data.append(self.X[idx - i])
        #     batch_label.append(self.label[idx])
        # elif self.phase == 'val':
        #     for i in range(TIME_STEP):
        #         batch_data.append(self.X[self.base_num + idx - i])
        #     batch_label.append(self.label[idx])
        #
        # return np.array(batch_data),np.array(batch_label)
        return  self.data[idx],self.label[idx]


    def __len__(self):
        return len(self.label)