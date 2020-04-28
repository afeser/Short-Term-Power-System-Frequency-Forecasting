'''
Manage data, divide, optimize for persistance etc.
'''
import pandas as pd
import numpy as np
import os



class DataManipulator:
    def __init__(self):
        '''
        Just define global variables.
        '''
        self.percent           = [70, 15, 15] # train / validation / test
        self.targetDatasetSize = 60 * 24 * 60 # minutes in 60 days
        self.fileName          = 'data/1year_frequency_train'
        # Where should new data files(e.g. .csv files) be stored?
        self.targetDataDir     = 'data'

    def divideToSets(self, clip=None, datasetStartIndex=0):
        '''
        Divite the minute data to sets of train - validation - test. The percentage
        of each set is defined inside the class.

        - Give clip value to cut(clip) data (given for all sets, applied to all)
        - Give start index to start reading data with given interval. Otherwise
        start from 0. Result dataset size.
        '''

        # Take one of intervals to create data
        data       = pd.read_csv(self.fileName + '_minute.csv')
        data.index = pd.to_datetime(data['dtm']).dt.tz_localize(None)

        data = data.drop(['dtm'], axis=1)

        uzunluk = self.targetDatasetSize

        tempData       = data[datasetStartIndex:datasetStartIndex + uzunluk]

        trainData      = tempData[:int(uzunluk*self.percent[0]/100)][:clip]
        validationData = tempData[ int(uzunluk*self.percent[0]/100):int(uzunluk*(self.percent[0]+self.percent[1])/100)][:clip]
        testData       = tempData[ int(uzunluk*(self.percent[0]+self.percent[1])/100):][:clip]

        print('Saving into ' + self.targetDataDir + '/TrainData.csv')
        trainData.to_csv(self.targetDataDir + '/TrainData.csv')

        print('Saving into ' + self.targetDataDir + '/ValidationData.csv')
        validationData.to_csv(self.targetDataDir + '/ValidationData.csv')

        print('Saving into ' + self.targetDataDir + '/TestData.csv')
        testData.to_csv(self.targetDataDir + '/TestData.csv')

    def getPersistanceErrors(self, data=None, alsoPrint=False):
        '''
        Calculate persistance MSE values.

        '''
        trainData      = data[0]
        validationData = data[1]
        testData       = data[2]

        trainData      = np.array(trainData['f'])
        validationData = np.array(validationData['f'])
        testData       = np.array(testData['f'])


        trainMSE      = np.sum((trainData[     :-1] - trainData[1     :])**2) / (trainData.shape[0]-1)
        validationMSE = np.sum((validationData[:-1] - validationData[1:])**2) / (validationData.shape[0]-1)
        testMSE       = np.sum((testData[      :-1] - testData[1      :])**2) / (testData.shape[0]-1)

        if alsoPrint:
            print('{0:40s} : {1:5f} / {2:5f} / {3:5f}'.format('Persistance model MSE for continuous outputs with minute data (train / validation / test)', trainMSE, validationMSE, testMSE))

        return [trainMSE, validationMSE, testMSE]

    def downsample_to_minute(self):
        '''
        Downsample data in seconds to data in minutes and save as csv.

        Give name of file containing second resolution data. Do not include csv extension
        '''
        data = pd.read_csv(self.fileName + '.csv')
        data.index = pd.to_datetime(data['dtm']).dt.tz_localize(None)
        data = data.resample('1T').first()

        data = data.drop(['dtm'], axis=1)

        data.to_csv(self.fileName + '_minute.csv')

    def maximizeTestError(self):
        '''
        Find dataset(train + validation + test) start index that maximizes the
        test error for persistance model.
        '''
        shiftLen   = 1 * 24 * 60 # 1 gun kaydirarak git
        datasetLen = self.targetDatasetSize

        data = pd.read_csv(self.fileName + '_minute.csv')

        shift = 0
        worst = {'index' : -1, 'val' : 0}

        while shift*shiftLen+datasetLen < len(data):
            tempData = data[shift*shiftLen:shift*shiftLen+datasetLen]
            testData = tempData[datasetLen*(self.percent[0] + self.percent[1])//100:]


            w = self.getPersistanceErrors(data=[testData, testData, testData])[2]
            if worst['val'] < w:
                worst['val']   = w
                worst['index'] = shift

            shift = shift + 1

        print('Maximum error is ' + str(worst['val']) + ', at start minute index ' + str(shiftLen * worst['index']) +' meaning at day ' + str(worst['index']))

        return shiftLen * worst['index']

    def createDataset(self, mode='maximizeTestError'):
        '''
        Directly create dataset from seconds file.
        Write minute data, train set, validation set, test set.

        Ask user for how data will be used(which portions etc.).
        '''

        if mode == 'maximizeTestError':
            if (not os.path.exists(self.fileName + '_minute.csv')):
                self.downsample_to_minute()

            datasetStartIndex = self.maximizeTestError()

            self.divideToSets(datasetStartIndex=datasetStartIndex)

        elif mode == 'maximizeTestError-testMode':
            if (not os.path.exists(self.fileName + '_minute.csv')):
                self.downsample_to_minute()

            datasetStartIndex = self.maximizeTestError()

            self.divideToSets(clip=144*2, datasetStartIndex=datasetStartIndex)





### Run...
dm = DataManipulator()
dm.targetDataDir = 'data5'
# dm.fileName = 'data/1year_frequency_train2018'
dm.targetDatasetSize = 90 * 24 * 60
# dm.createDataset('maximizeTestError-testMode')
dm.createDataset('maximizeTestError')
