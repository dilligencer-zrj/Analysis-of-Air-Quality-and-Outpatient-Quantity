# coding:utf-8
import matplotlib.pyplot as plt
import csv
import numpy as np

def plot(file_1,file_2):
    weather_list = []
    outpatient_list = []
    with open ( file_1, 'r' ) as f:
        weather = csv.reader ( f )
        for row in weather:
            tmp = []
            for i in row:
                tmp.append ( float ( i ) )
            weather_list.append ( tmp )
    with open ( file_2, 'r' ) as f:
        outpatient = csv.reader ( f )
        for row in outpatient:
            tmp = []
            for i in row:
                tmp.append ( float ( i ) )
            outpatient_list.append ( tmp )
    weather_array = np.array(weather_list)
    outpatient_array = np.array(outpatient_list)

    Invalid_index = np.where( outpatient_array[:, 0] < 1000 )
    Valid_index = np.where ( outpatient_array[:, 0] > 1000 )

    # with open('/home/dilligencer/code/一附院时序数据回归/file/new_weather.csv','w') as csvfile:
    #     writer = csv.writer ( csvfile )
    #     writer.writerows(weather_array[Valid_index])
    with open('/home/dilligencer/code/一附院时序数据回归/file/new_outpatient.csv','w') as csvfile:
        writer = csv.writer ( csvfile )
        writer.writerows(outpatient_array[Valid_index])

    # for k in range(6,1,-1):
    #     base_num = 211
    #     for j in range ( 2 ):
    #         tmp_list = []
    #         for i in range ( len ( Invalid_index[0] ) )[1:]:
    #             if (Invalid_index[0][i] - Invalid_index[0][i - 1] == 7):
    #                 tmp_list.append ( outpatient_array[Invalid_index[0][i] - k, j] )
    #         print (tmp_list)
    #         plt.subplot ( base_num )
    #         plt.plot ( range ( len ( tmp_list ) ), tmp_list, 'r' )
    #         base_num += 1
    #     plt.show ( )



        # plt.xlabel('time')
    # plt.subplot ( 611 )
    # plt.plot ( np.arange ( 0, 359 ), weather_array[730:, 0] ,'r')
    # plt.subplot ( 612 )
    # plt.plot ( np.arange ( 0, 359 ), weather_array[730:, 1], 'r' )
    # plt.subplot ( 613 )
    # plt.plot ( np.arange ( 0, 359 ), weather_array[730:, 2], 'r' )
    # plt.subplot ( 614 )
    # plt.plot ( np.arange ( 0, 359 ), weather_array[730:, 3], 'r' )
    # plt.subplot ( 615 )
    # plt.plot ( np.arange ( 0, 359 ), weather_array[730:, 4], 'r' )
    # plt.subplot ( 616 )
    # plt.plot ( np.arange ( 0, 359 ), weather_array[730:, 5], 'r' )

    # plt.subplot(211)
    # plt.plot ( np.arange ( 0, 15 ), outpatient_array[:15, 0],'b' )
    # plt.subplot ( 212 )
    # plt.plot ( np.arange ( 0, 15 ), outpatient_array[:15, 1], 'b' )

    # for i in range(6):
    #     idcs = weather_array[730:, i].argsort()
    #     plt.subplot(211)
    #     plt.plot(weather_array[730:, i][idcs],outpatient_array[730:,0][idcs],'r')
    #     plt.subplot(212)
    #     plt.plot ( weather_array[730:, i][idcs],outpatient_array[730:,1][idcs],'b' )
    #     plt.show()



if __name__ == '__main__':
    weather_file = '/home/dilligencer/code/一附院时序数据回归/file/weather.csv'
    outpatient = '/home/dilligencer/code/一附院时序数据回归/file/outpatient.csv'
    plot(file_1=weather_file,file_2=outpatient)