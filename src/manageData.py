'''
Manage data, divide, optimize for persistance etc.
'''
import pandas as pd
import numpy as np
import datetime as dt
import os
import logging
import shutil


class DataManipulator:
    def __init__(self, fileName, outputFileName):
        '''
        Just define global variables.

        Important, lines are deleted since they were confusing.
        Some methods of this class should be corrected!

        DataManipulator('input_file_name.csv')
        '''
        # Where should new data files(e.g. .csv files) be stored?
        self.fileName = fileName
        self.outputFileName = outputFileName


    def createSpecificDateLength(self, specific_date, length, format_date):
        '''
        Creates a data set starting from the specific date supplied and going length number of days.
        The result is written to CreateSpecificDateLength.csv.

        Examples:
            createSpecificDateLength('2020-01-01', 3, '%Y-%m-%d') # create dataset for days '2020-01-01', '2020-01-02', '2020-01-03'
            Doc : https://www.journaldev.com/23365/python-string-to-datetime-strptime
        '''
        data = pd.read_csv(self.fileName + '_minute.csv')
        new_data = pd.DataFrame()

        days = [dt.datetime.strftime(dt.datetime.strptime(specific_date, format_date) + dt.timedelta(days=day_counter), format_date) for day_counter in range(length)]

        new_data = data[data['dtm'].map(lambda x: x.split()[0] in days)]

        new_data.index = new_data['dtm']
        new_data = new_data.drop([new_data.keys()[0]], axis=1)
        new_data.to_csv(self.outputFileName)



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

    def getPersistanceErrors(self, data, alsoPrint=False):
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
        logging.info('Starting function downsample_to_minute...')

        logging.info('Reading data from ' + str(self.fileName))
        data = pd.read_csv(self.fileName)


        logging.info('Setting table index to dtm')
        data.index = pd.to_datetime(data['dtm']).dt.tz_localize(None)
        logging.info('Downsampling')
        data = data.resample('1T').first()

        logging.info('Dropping extra column')
        data = data.drop(['dtm'], axis=1)

        logging.info('Writing downsample result to ' + str(self.fileName + '_minute.csv'))
        data.to_csv(self.fileName + '_minute.csv')


        del data

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

    def createDataset(self, mode='maximizeTestError', args=[]):
        '''
        Directly create dataset from seconds file.
        Write minute data, train set, validation set, test set.

        Ask user for how data will be used(which portions etc.).

        test_start_date_length:
            Give test start date, number of days as length and date format.
            Example : createDataset('test_start_date_length', args = ['2018-08-11', 9, '%Y-%m-%d']) # note args[0] should be the same string as in the csv source file
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

        elif mode == 'test_start_date_length':
            start_date   = args[0]
            length       = args[1]
            date_format  = args[2]
            if (not os.path.exists(self.fileName + '_minute.csv')):
                self.downsample_to_minute()


            self.createSpecificDateLength(start_date, length, date_format)








### Run...
logging.getLogger().setLevel(logging.DEBUG)

### Dataset 1
def create_dataset1():
    if os.path.exists('data/data1'):
        shutil.rmtree('data/data1')
    os.makedirs('data/data1')


    os.symlink('../DemandData_2017.csv', 'data/data1/TrainLoad.csv')
    os.symlink('../DemandData_2017.csv', 'data/data1/ValidationLoad.csv')
    os.symlink('../DemandData_2017.csv', 'data/data1/TestLoad.csv')

    dm = DataManipulator('data/2017_frequency.csv', 'data/data1/TrainData.csv')
    dm.createDataset('test_start_date_length', ['2017-08-11', 42, '%Y-%m-%d'])
    dm = DataManipulator('data/2017_frequency.csv', 'data/data1/ValidationData.csv')
    dm.createDataset('test_start_date_length', ['2017-09-22', 9, '%Y-%m-%d'])
    dm = DataManipulator('data/2017_frequency.csv', 'data/data1/TestData.csv')
    dm.createDataset('test_start_date_length', ['2017-10-01', 9, '%Y-%m-%d'])


### Dataset 2 in the article -> only test set taken from dataset 1
def create_dataset2():
    if not os.path.exists('data/data1'):
        raise 'Data set 1 does not exist'

    if os.path.exists('data/data2'):
        shutil.rmtree('data/data2')
    os.makedirs('data/data2')

    os.symlink('../data1/TrainData.csv', 'data/data2/TrainData.csv')
    os.symlink('../data1/ValidationData.csv', 'data/data2/ValidationData.csv')

    os.symlink('../DemandData_2017.csv', 'data/data2/TrainLoad.csv')
    os.symlink('../DemandData_2017.csv', 'data/data2/ValidationLoad.csv')
    os.symlink('../DemandData_2018.csv', 'data/data2/TestLoad.csv')

    dm = DataManipulator('data/2018_frequency.csv', 'data/data2/TestData.csv')
    #dm.targetDataDir = 'data2'
    #dm.targetDatasetSize = 60 * 24 * 60
    # dm.createDataset('maximizeTestError-testMode')
    #dm.createDataset('maximizeTestError')
    dm.createDataset('test_start_date_length', ['2018-07-02', 9, '%Y-%m-%d'])



create_dataset1()
create_dataset2()
