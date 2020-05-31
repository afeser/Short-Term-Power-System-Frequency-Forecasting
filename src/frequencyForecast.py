#!/usr/bin/env python3
'''
Important notes :
    1) The code is initially written based on validation / train sets. When validation set
    is included. Changing all names are required. Instead of changing all variable names,
    new variable name is created for test set, called 'realTest'. Therefore, 'test' variable
    represents validation set whereas 'realTest' variable represents the actual test set.
'''


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, GRU, SimpleRNN
import numpy
import time
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import pickle
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import random
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
import sys
import datetime
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import threading
import pathlib
from tensorflow.keras.layers import Dropout
from multiprocessing.pool import ThreadPool
os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # str(random.randint(-1,1))
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)



class FrequencyForecaster:
    def __init__(self, dirName, enableSecondForecasting=False, forecastModel='LSTM'):
        '''
        Initialize FrequencyForecaster,
        Give directory name for the object files, outputs, object data
        By default everything is false, manually set them with object.variable
        '''
        if not os.path.exists(dirName):
            pathlib.Path(dirName).mkdir(parents=True, exist_ok=True)
            # os.mkdir(dirName)

        # Parameters for LSTM input (we need to know the input shape)
        # These are dynamic (2 month will not yield OneHotEncoding with 12 length)
        self.enableTimeFeaturesLookback        = False
        self.enableTimeFeaturesForecastHorizon = False
        self.enableExcursionFeature            = False
        self.enableLoadData                    = False
        self.enableClassification              = False
        self.enableSecondForecasting           = enableSecondForecasting # DO NOT CHANGE BY object.enableSecondForecasting
        self.enableSelectiveLoadFeatures       = False # When True, only selected features in data are used, wherease False implies ignore selectiveLoadFeatures variable and include all load features
        self.selectiveLoadFeatureNames         = [
                                                'I014_TSD'
                                        ]

        # Time feature filters
        self.enableMonthofYear  = False
        self.enableDayofWeek    = False
        self.enableHourofDay    = False
        self.enableMinuteofHour = False
        self.binBoundaryType    = 'StandardDeviation'

        self.forecastHorizon      = 1
        self.directoryName        = dirName
        self.dataDirectory        = 'data'
        self.loss                 = 'mse'
        self.optimizer            = 'adam'
        self.stateful             = True

        # Calculate from the data...
        # self.ds_train             = 1440 * 30 * 2
        # self.ds_test              = self.ds_train // 2
        # if self.enableSecondForecasting:
        #     self.ds_train = self.ds_train * 60
        #     self.ds_test  = self.ds_test  * 60


        self.lb        = 12
        self.batchSize = 1
        self.nnum      = 128
        self.lnumber   = 2
        self.dropout   = 0

        # Store errors for every epoch
        self.errors    = {
            'trainMAE' : [],
            'testMAE'  : [],
            'realTestMAE' : [],
            'persistenceTrainMAE' : [],
            'persistenceTestMAE'  : [],
            'persistenceRealTestMAE' : [],
            'realTestMAXMAE' : [],
            'persistenceRealTestMAXMAE' : [],
            'realTestNum15mHz' : [],
            'persistenceRealTestNum15mHz' : [],
            'realTestHistogramMAE' : [],
            'persistenceRealTestHistogramMAE' : [],


            'trainMSE' : [],
            'testMSE'  : [],
            'realTestMSE' : [],
            'persistenceTrainMSE' : [],
            'persistenceTestMSE'  : [],
            'persistenceRealTestMSE' : [],

            'trainMAPE' : [],
            'testMAPE'  : [],
            'realTestMAPE' : [],
            'persistenceTrainMAPE' : [],
            'persistenceTestMAPE'  : [],
            'persistenceRealTestMAPE' : []

        }

        self.figure = plt.figure(figsize=(16, 12))

        self.nn = None

        if forecastModel == 'LSTM':
            self.forecastModel = LSTM # can be changed to GRU..
        elif forecastModel == 'GRU':
            self.forecastModel = GRU
        elif forecastModel == 'SimpleRNN':
            self.forecastModel = SimpleRNN
        else:
            raise NameError('Requested model ' + str(forecastModel) + ' not found!')


    def plot(self, figNum, actualOutput, lstmOuts, customTag=''):
        print('Plotting ' + str(figNum) + '. ' + customTag + ' output...')
        # lstmOuts -> predictions inverse transformed
        # actualOutput -> data read from file

        # forecastLong : forecast with given forecastHorizon
        forecastLong   = [lstmOuts[i//self.forecastHorizon*self.forecastHorizon][i%self.forecastHorizon] for i in range(lstmOuts.shape[0])]
        startingPoints = numpy.array([lstmOuts[i][0] for i in range(lstmOuts.shape[0])])



        plt.plot(actualOutput, 'go', markersize=9)
        plt.plot(startingPoints, 'rX')



        title = ''
        title = title + customTag + '\n'
        title = title + ('nnum = ' + str(self.nnum) + ', ')
        title = title + ('ds_train = ' + str(self.ds_train) + ', ')
        title = title + ('lb = ' + str(self.lb) + ', ')
        title = title + ('lnumber = '  + str(self.lnumber) + '\n')
        title = title + ('loss = ' + str(self.loss) + ', ')
        title = title + ('optimizer = ' + str(self.optimizer) + ', ')
        title = title + ('forecastHorizon = ' + str(self.forecastHorizon) + ', ')
        title = title + ('currentEpoch = ' + str(figNum) + '\n')
        plt.title(title, fontsize=6)

        plt.legend(['Correct Output', 'Predicted Output'])
        plt.savefig(self.directoryName + '/' + customTag + str(figNum) + '.png', bbox_inches="tight")
        plt.cla()

        print('Done!')

    def readPrepareData(self):
        '''
        Read and prepare data, read directly from csv file and preprocess it.
        MinMaxScaler, input dimensions, feature addition etc. all done here.
        This method eventually sets the self parameters for further reference
        '''
        ## Read data...
        def readData(scaler):
            ### Train data
            trainActualData = pd.read_csv(self.dataDirectory + '/TrainData.csv', comment='#')
            trainExtraData  = pd.read_csv(self.dataDirectory + '/TrainLoad.csv', comment='#')

            # UTC binlerce sorun yarattigi icin timezone olarak None aliyorum...
            # Veriyi de buna gore duzenliyorum...
            trainActualData.index = pd.to_datetime(trainActualData['dtm']).dt.tz_localize(None)
            if not self.enableSecondForecasting:
                trainActualData = trainActualData.resample('1T').first()

            train_date_time = list(trainActualData.index)
            trainOutputDat = list(trainActualData['f'])


            # Convert date to datetime object - convert to tz-aware form for local time differences
            trainExtraData['SETTLEMENT_DATE'] = pd.to_datetime(trainExtraData['SETTLEMENT_DATE']).dt.tz_localize(None)
            # Convert half hours to timedelta object
            trainExtraData = trainExtraData.apply(lambda x : (x-1) * 30 * datetime.timedelta(minutes=1) if x.name == 'SETTLEMENT_PERIOD' else x)
            # Create datetime variable
            trainExtraData['DATETIME'] = trainExtraData['SETTLEMENT_DATE'] + trainExtraData['SETTLEMENT_PERIOD']


            ### Test data
            testActualData = pd.read_csv(self.dataDirectory + '/ValidationData.csv', comment='#')
            testExtraData  = pd.read_csv(self.dataDirectory + '/ValidationLoad.csv', comment='#')

            # UTC binlerce sorun yarattigi icin timezone olarak None aliyorum...
            # Veriyi de buna gore duzenliyorum...
            testActualData.index = pd.to_datetime(testActualData['dtm']).dt.tz_localize(None)
            if not self.enableSecondForecasting:
                testActualData = testActualData.resample('1T').first()

            test_date_time = list(testActualData.index)
            testOutputDat  = list(testActualData['f'])


            # Convert date to datetime object - convert to tz-aware form for local time differences
            testExtraData['SETTLEMENT_DATE'] = pd.to_datetime(testExtraData['SETTLEMENT_DATE']).dt.tz_localize(None)
            # Convert half hours to timedelta object
            testExtraData = testExtraData.apply(lambda x : (x-1) * 30 * datetime.timedelta(minutes=1) if x.name == 'SETTLEMENT_PERIOD' else x)
            # Create datetime variable
            testExtraData['DATETIME'] = testExtraData['SETTLEMENT_DATE'] + testExtraData['SETTLEMENT_PERIOD']


            ### RealTest data
            realTestActualData = pd.read_csv(self.dataDirectory + '/TestData.csv', comment='#')
            realTestExtraData  = pd.read_csv(self.dataDirectory + '/TestLoad.csv', comment='#')

            # UTC binlerce sorun yarattigi icin timezone olarak None aliyorum...
            # Veriyi de buna gore duzenliyorum...
            realTestActualData.index = pd.to_datetime(realTestActualData['dtm']).dt.tz_localize(None)
            if not self.enableSecondForecasting:
                realTestActualData = realTestActualData.resample('1T').first()

            realTest_date_time = list(realTestActualData.index)
            realTestOutputDat  = list(realTestActualData['f'])


            # Convert date to datetime object - convert to tz-aware form for local time differences
            realTestExtraData['SETTLEMENT_DATE'] = pd.to_datetime(realTestExtraData['SETTLEMENT_DATE']).dt.tz_localize(None)
            # Convert half hours to timedelta object
            realTestExtraData = realTestExtraData.apply(lambda x : (x-1) * 30 * datetime.timedelta(minutes=1) if x.name == 'SETTLEMENT_PERIOD' else x)
            # Create datetime variable
            realTestExtraData['DATETIME'] = realTestExtraData['SETTLEMENT_DATE'] + realTestExtraData['SETTLEMENT_PERIOD']


            scaler.fit(numpy.concatenate([trainOutputDat, testOutputDat, realTestOutputDat]).reshape(-1, 1))



            self.ds_train = len(trainOutputDat)
            self.ds_test  = len(testOutputDat)
            self.ds_realTest = len(realTestOutputDat)

            return trainOutputDat, trainExtraData, train_date_time,  testOutputDat, testExtraData, test_date_time,  realTestOutputDat, realTestExtraData, realTest_date_time
        ## Prepare data
        def prepareData(Xa, extraData, date_time, dataType, scaler):
            '''
            Data Representation :
            [samples, timesteps, features]

            Each timestep :
            [consumptionValue, 24 vector of One Hot Encoding of Input, forecast * 24 vector of One Hot Encoding of Output]
            Example :
            [0.13, 0, 0, 1, 0, ..., 0, | 0, 1, ..., 0, | 0, 0, 0, 1, .., 0 ] -> 1 + 24 + forecastHorizon * 24 elements (| is used as a seperator to make it readeable)

            Each sample :
            [ timestep_0, timestep_1, ..., timestep_n ]

            Overall(3D) :
            [ [ timestep_0, timestep_1, ..., timestep_n ], [ timestep_0, timestep_1, ..., timestep_n ] ... ]
            '''
            self.num_months         = 0
            self.num_days           = 0
            self.num_hours          = 0
            self.num_minutes        = 0
            self.num_extra_features = 0


            if dataType == 'train':
                ds = self.ds_train
            elif dataType == 'test':
                ds = self.ds_test
            elif dataType == 'realTest':
                ds = self.ds_realTest

            print('Preparing data')

            features = []
            if self.enableLoadData:
                for (columnName, columnData) in extraData.iteritems():
                    if columnName in ['DATETIME', 'SETTLEMENT_DATE', 'SETTLEMENT_PERIOD']:
                        continue

                    if self.enableSelectiveLoadFeatures:
                        # Load only selected ones...
                        if not columnName in self.selectiveLoadFeatureNames:
                            continue


                    print('\tAdding feature ' + columnName)
                    self.num_extra_features = self.num_extra_features + 1
                    tempScaler = MinMaxScaler(feature_range=(-1, 1))
                    features.append(tempScaler.fit_transform(numpy.array(columnData).reshape(-1, 1)).reshape(-1))


            Xa = numpy.array(Xa)
            if self.enableExcursionFeature:
                excursionData = numpy.concatenate((numpy.array([Xa[1]-Xa[0]]), Xa[1:] - Xa[:-1]))
                tempScaler = MinMaxScaler(feature_range=(-1,1))
                excursionData = tempScaler.fit_transform(excursionData.reshape(-1, 1))
                self.num_extra_features = self.num_extra_features + 1


            #binBoundaries = numpy.array([-0.080, -0.036, 0.036, 0.080]) + 50
            if self.binBoundaryType == 'StandardDeviation':
                binBoundaries = numpy.array([-3*0.06016975081160766, -2*0.06016975081160766, -0.06016975081160766, 0.06016975081160766, 2*0.06016975081160766, 3*0.06016975081160766]) + 50 # Standard Deviation
            elif self.binBoundaryType == 'Gaussian':
                binBoundaries = numpy.array([49.836000000000006, 49.888999999999996, 49.94, 50.06100000000001, 50.121, 50.183]) # Gaussian boundaries specific to data


            # Concat the bins so that the transform can know the output bin boundaries for the MinMaxScaler transformed version
            Xa = Xa[:ds]
            Xa = numpy.concatenate((Xa, binBoundaries))
            X = scaler.transform(Xa.reshape(Xa.shape[0],1)).reshape(Xa.shape[0])

            binBoundaries = X[ds:]
            X             = X[:ds]

            enableMonthofYear  = self.enableMonthofYear
            enableDayofWeek    = self.enableDayofWeek
            enableHourofDay    = self.enableHourofDay
            enableMinuteofHour = self.enableMinuteofHour
            enableTimeFeaturesForecastHorizon = self.enableTimeFeaturesForecastHorizon
            enableTimeFeaturesLookback = self.enableTimeFeaturesLookback
            enableExcursionFeature = self.enableExcursionFeature
            # Prepare features
            if enableMonthofYear:
                print('\tAdding feature month of year')
            if enableDayofWeek:
                print('\tAdding feature day of week')
            if enableHourofDay:
                print('\tAdding feature hour of day')
            if enableMinuteofHour:
                print('\tAdding feature minute of hour')

            all_hours   = [o.hour for o in date_time]
            all_days    = [o.weekday() for o in date_time]
            all_months  = [o.month for o in date_time]
            all_minutes = [o.minute for o in date_time]

            all_hours   = numpy.array(all_hours).reshape(len(all_hours), 1)
            all_days    = numpy.array(all_days).reshape(len(all_days), 1)
            all_months  = numpy.array(all_months).reshape(len(all_months), 1)
            all_minutes = numpy.array(all_minutes).reshape(len(all_minutes), 1)


            onehot_hour_of_day      = OneHotEncoder(sparse =False, categories ='auto')
            onehot_day_of_week      = OneHotEncoder(sparse =False, categories ='auto')
            onehot_month_of_year    = OneHotEncoder(sparse =False, categories ='auto')
            onehot_minute_of_hour   = OneHotEncoder(sparse =False, categories ='auto')


            encoded_hour_of_day      = onehot_hour_of_day.fit(numpy.arange(24).reshape(-1, 1))
            encoded_day_of_week      = onehot_day_of_week.fit(numpy.arange(7).reshape(-1, 1))
            encoded_month_of_year    = onehot_month_of_year.fit(numpy.arange(12).reshape(-1, 1))
            encoded_minute_of_hour   = onehot_minute_of_hour.fit(numpy.arange(60).reshape(-1, 1))

            encoded_hour_of_day      = onehot_hour_of_day.transform(all_hours)
            encoded_day_of_week      = onehot_day_of_week.transform(all_days)
            encoded_month_of_year    = onehot_month_of_year.transform(all_months)
            encoded_minute_of_hour   = onehot_minute_of_hour.transform(all_minutes)

            if enableMonthofYear:
                self.num_months  = onehot_month_of_year.get_feature_names().shape[0]
            if enableDayofWeek:
                self.num_days    = onehot_day_of_week.get_feature_names().shape[0]
            if enableHourofDay:
                self.num_hours   = onehot_hour_of_day.get_feature_names().shape[0]
            if enableMinuteofHour:
                self.num_minutes = onehot_minute_of_hour.get_feature_names().shape[0]


            lb = self.lb
            forecastHorizon = self.forecastHorizon
            customX = []
            # Every input is an element
            customY = []
            # i -> each sample, index of the samples from 0 to (ds - lb - forecastHorizon + 1)
            for i in range(ds - lb - forecastHorizon + 1):
                # -------------------- Samples
                customXsamples=[]

                # j -> each look back timestep, index of the timesteps from 0 to lb
                for j in range(lb):
                    # -------------------- Timesteps
                    # Add each OneHotEncoding as a feature
                    seriesOutputOneHot = []
                    # k -> each feature, index of the features from 0 to forecastHorizon (also additional features are added, this is only for future horizon times)
                    for k in range(forecastHorizon):
                        # -------------------- Features
                        # Add features based on the flags

                        if enableTimeFeaturesForecastHorizon:
                            # Add for the whole horizon
                            if enableMonthofYear:
                                seriesOutputOneHot.extend(encoded_month_of_year[i+lb+k])

                            # Add for the whole horizon
                            if enableDayofWeek:
                                seriesOutputOneHot.extend(encoded_day_of_week[i+lb+k])

                            # Add for the whole horizon
                            if enableHourofDay:
                                seriesOutputOneHot.extend(encoded_hour_of_day[i+lb+k])

                            # Add for the whole horizon
                            if enableMinuteofHour:
                                seriesOutputOneHot.extend(encoded_minute_of_hour[i+lb+k])



                    # Form input array
                    add_to_input_array = []
                    # Add features
                    if enableTimeFeaturesLookback:
                        if enableMonthofYear:
                            add_to_input_array.extend(list(encoded_month_of_year[i+j]))
                        if enableDayofWeek:
                            add_to_input_array.extend(list(encoded_day_of_week[i+j]))
                        if enableHourofDay:
                            add_to_input_array.extend(list(encoded_hour_of_day[i+j]))
                        if enableMinuteofHour:
                            add_to_input_array.extend(list(encoded_minute_of_hour[i+j]))

                    extraDataAddition = []
                    if not features == []:
                        # Round down to the half hour to get load values
                        roundedDatetime = date_time[i+j] - date_time[i+j].minute * datetime.timedelta(minutes=1) - date_time[i+j].second * datetime.timedelta(seconds=1)
                        roundedDatetime = roundedDatetime + datetime.timedelta(minutes=1) * (date_time[i+j].minute // 30) * 30
                        indices = extraData.index[extraData['DATETIME'] == roundedDatetime][0]

                        extraDataAddition = [feature[indices] for feature in features]

                    if enableExcursionFeature:
                        customXsamples.append([X[i+j]] + [excursionData[i+j][0]] + extraDataAddition + add_to_input_array + seriesOutputOneHot)
                    else:
                        customXsamples.append([X[i+j]] + extraDataAddition + add_to_input_array + seriesOutputOneHot)



                customX.append(customXsamples)

                # We may want to use one hot encoding at the end
                customY.append(X[i + lb : i + lb + forecastHorizon])

            customY        = numpy.array(customY)
            if self.enableClassification:
                oldShape       = customY.shape
                customY        = customY.reshape(-1, 1)
                customY        = numpy.digitize(customY, bins=binBoundaries)
                customYEncoder = OneHotEncoder(sparse=False, categories ='auto')
                customY        = customYEncoder.fit_transform(customY)
                customY.reshape(ds-lb-forecastHorizon+1, customYEncoder.get_feature_names().shape[0] * oldShape[1])

            print('Bin Boundaries : ' + str(binBoundaries))
            return numpy.array(customX), customY


        print('Reading data...')
        scaler = MinMaxScaler(feature_range =(-1,1))
        trainData, trainExtraLoadData, train_date_time, testData, testExtraLoadData, test_date_time, realTestData, realTestExtraLoadData, realTest_date_time = readData(scaler)

        print('Preprocessing data...')
        Xtrain, Ytrain = prepareData(trainData, trainExtraLoadData, train_date_time, 'train', scaler)
        Xtest,  Ytest  = prepareData(testData , testExtraLoadData, test_date_time, 'test', scaler)
        XrealTest,  YrealTest  = prepareData(realTestData , realTestExtraLoadData, realTest_date_time, 'realTest', scaler)

        if self.enableClassification:
            # Variables for plotting
            self.YtrainPlot = numpy.argmax(Ytrain, axis=1) - Ytrain.shape[1] // 2 # TODO - forecastHorizon>1 icin ne yapacagiz? straightforward mi?- sanmiyorum, cunku her sample icin farkli farkli gruplarin max. indexlerini bulmamiz gerekecek 3 boyutlu vektor olarak hesaplayip
            self.YtestPlot  = numpy.argmax(Ytest, axis=1) - Ytrain.shape[1] // 2
        else:
            self.YtrainPlot = scaler.inverse_transform(Ytrain)
            self.YtestPlot  = scaler.inverse_transform(Ytest)

        # Export to global variables
        self.scaler = scaler
        self.trainData = trainData
        self.trainExtraLoadData = trainExtraLoadData
        self.train_date_time = train_date_time
        self.testData = testData
        self.testExtraLoadData = testExtraLoadData
        self.test_date_time = test_date_time
        self.realTestData = realTestData
        self.realTestExtraLoadData = realTestExtraLoadData
        self.realTest_date_time = realTest_date_time
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.XrealTest = XrealTest
        self.YrealTest = YrealTest


    def saveData(self):
        '''
        Save preprocessed data, every variable into file in output directory
        '''
        print('Saving data...')
        if not self.enableSecondForecasting:
            # For secondForcasting data size is bigger than 4GB, fail to save it with pickle...
            featureListLen = [self.num_extra_features, self.enableTimeFeaturesLookback, self.forecastHorizon, self.enableTimeFeaturesForecastHorizon, self.num_months, self.num_days, self.num_hours, self.num_minutes]
            with open(self.directoryName + '/preprocessedData.pickle', 'wb') as f:
                pickle.dump([self.realTestData, self.realTestExtraLoadData, self.realTest_date_time, self.XrealTest, self.YrealTest, self.ds_realTest, self.ds_train, self.ds_test, featureListLen, self.YtrainPlot, self.YtestPlot, self.scaler, self.trainData, self.trainExtraLoadData, self.train_date_time, self.testData, self.testExtraLoadData, self.test_date_time, self.Xtrain, self.Ytrain, self.Xtest, self.Ytest, self.num_months, self.num_days, self.num_hours, self.num_minutes, self.num_extra_features] , f)
    def loadData(self, load_dir=None):
        '''
        Load the preprocessed data from output directory
        '''
        if load_dir is None:
            load_dir = self.directoryName

        with open(load_dir + '/preprocessedData.pickle', 'rb') as f:
            self.realTestData, self.realTestExtraLoadData, self.realTest_date_time, self.XrealTest, self.YrealTest, self.ds_realTest, self.ds_train, self.ds_test, featureListLen, self.YtrainPlot, self.YtestPlot, self.scaler, self.trainData, self.trainExtraLoadData, self.train_date_time, self.testData, self.testExtraLoadData, self.test_date_time, self.Xtrain, self.Ytrain, self.Xtest, self.Ytest, self.num_months, self.num_days, self.num_hours, self.num_minutes, self.num_extra_features = pickle.load(f)

        self.num_extra_features, self.enableTimeFeaturesLookback, self.forecastHorizon, self.enableTimeFeaturesForecastHorizon, self.num_months, self.num_days, self.num_hours, self.num_minutes = featureListLen

    def _buildModel(self):
        # Build model
        ## Loss function-metric definitions
        stdSonucu = numpy.std(self.Ytest)
        def nRMSE(y_true, y_pred):
            # BU HATALI! BUNUN HESAPLANMASINDA VERI TEK TEK VERILIYOR, O YUZDEN MAE ILE AYNI SEY CIKIYOR BU!
            # Elemanlari tek tek verip hesapladigi icin standard deviation hesaplarken havaya ucuyor sistem cunku elimizde 1 tane eleman var!
            return K.sqrt(K.mean(K.square(y_pred - y_true))) / stdSonucu
        def RMSE(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))
        def MAPE(y_true, y_pred):
            # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
            pass

        nn2 = Sequential()


        ## tensorboard
        logdir = self.directoryName + '/logs/scalars/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


        ## Model parameters
        if self.enableClassification:
            outputLayerDim = self.forecastHorizon * self.Ytrain.shape[1]
        else:
            outputLayerDim = self.forecastHorizon

        # Size of features - previous time + current time step features + horizon features
        num_features = 1 + (self.num_extra_features + (self.num_months + self.num_days + self.num_hours + self.num_minutes) * self.enableTimeFeaturesLookback) + self.forecastHorizon * self.enableTimeFeaturesForecastHorizon * (self.num_months + self.num_days + self.num_hours + self.num_minutes)

        ## Construct model
        for i in range(self.lnumber-1):
            nn2.add(self.forecastModel(self.nnum, input_shape=(self.lb, num_features), return_sequences=True, stateful=self.stateful, batch_size=self.batchSize, dropout=self.dropout))

        nn2.add(self.forecastModel(self.nnum, input_shape=(self.lb, num_features), stateful=self.stateful, batch_size=1, dropout=self.dropout))

        nn2.add(Dense(outputLayerDim))

        if self.enableClassification:
            nn2.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        else:
            nn2.compile(optimizer=self.optimizer, loss=self.loss, metrics=['mse', 'mae'])

        self.nn = nn2

    def train(self, epoch=1, trainMode='predictTrain'):
        '''
        Give number of epochs and train mode.
        Train mode : wheter to save results in each epoch or train continuously
            - directTrain  : mode.train(epoch=100)
            - predictTrain : save intermediate results
        '''

        # If not done yet, build the model
        if self.nn is None:
            self._buildModel()


        if trainMode == 'directTrain':
            pass
        elif trainMode == 'predictTrain':
            listStart = 250
            listEnd   = 450

            if len(self.errors['trainMAE']) == 0:
                # Test on train set and plot
                self.plot(0, self.YtrainPlot[listStart:listEnd], self.predict(self.Xtrain[listStart:listEnd]), customTag='train')
                # Test on test set and plot
                self.plot(0, self.YtestPlot[listStart:listEnd], self.predict(self.Xtest[listStart:listEnd]), customTag='validation')

            for i in range(len(self.errors['trainMAE']), epoch+len(self.errors['trainMAE'])):
                history_lstm2 = self.nn.fit(self.Xtrain, self.Ytrain, batch_size=self.batchSize, epochs=1, shuffle=False, verbose=1, callbacks=[self.tensorboard_callback], validation_data=(self.Xtest, self.Ytest))

                # Test on train set and plot
                self.plot(i+1, self.YtrainPlot[listStart:listEnd], self.predict(self.Xtrain[listStart:listEnd]), customTag='train')

                # Test on test set and plot
                self.plot(i+1, self.YtestPlot[listStart:listEnd], self.predict(self.Xtest[listStart:listEnd]), customTag='validation')

                # Calculate errors and log
                trainPredictions    = self.predict(self.Xtrain)
                testPredictions     = self.predict(self.Xtest )
                realTestPredictions = self.predict(self.XrealTest)
                trainCikti          = self.scaler.inverse_transform(numpy.array( self.Ytrain )).reshape(-1, 1)
                testCikti           = self.scaler.inverse_transform(numpy.array( self.Ytest )).reshape(-1, 1)
                realTestCikti       = self.scaler.inverse_transform(numpy.array( self.YrealTest)).reshape(-1, 1)

                ## MAE
                trainMAE                        = numpy.sum(numpy.abs(trainPredictions - trainCikti)) / trainCikti.shape[0]
                testMAE                         = numpy.sum(numpy.abs(testPredictions - testCikti )) / testCikti.shape[0]
                realTestMAE                     = numpy.sum(numpy.abs(realTestPredictions - realTestCikti)) / realTestCikti.shape[0]
                persistenceTrainMAE             = numpy.sum(numpy.abs(trainCikti[:-1] - trainCikti[1:])) / (trainCikti.shape[0] - 1)
                persistenceTestMAE              = numpy.sum(numpy.abs(testCikti[:-1] - testCikti[1:])) / (testCikti.shape[0] - 1)
                persistenceRealTestMAE          = numpy.sum(numpy.abs(realTestCikti[:-1] - realTestCikti[1:])) / (realTestCikti.shape[0] - 1)
                realTestMAXMAE                  = numpy.max(numpy.abs(realTestPredictions - realTestCikti))
                persistenceRealTestMAXMAE       = numpy.max(numpy.abs(realTestCikti[1:] - realTestCikti[:-1]))
                realTestNum15mHz                = numpy.sum(numpy.abs(realTestPredictions - realTestCikti) > 0.015)
                persistenceRealTestNum15mHz     = numpy.sum(numpy.abs(realTestCikti[1:] - realTestCikti[:-1]) > 0.015)
                realTestHistogramMAE            = realTestCikti - realTestPredictions
                persistenceRealTestHistogramMAE = realTestCikti[1:] - realTestCikti[:-1]

                self.errors['trainMAE'].append(trainMAE)
                self.errors['testMAE'].append(testMAE)
                self.errors['realTestMAE'].append(realTestMAE)
                self.errors['persistenceTrainMAE'].append(persistenceTrainMAE)
                self.errors['persistenceTestMAE'].append(persistenceTestMAE)
                self.errors['persistenceRealTestMAE'].append(persistenceRealTestMAE)
                self.errors['realTestMAXMAE'].append(realTestMAXMAE)
                self.errors['persistenceRealTestMAXMAE'].append(persistenceRealTestMAXMAE)
                self.errors['realTestNum15mHz'].append(realTestNum15mHz)
                self.errors['persistenceRealTestNum15mHz'].append(persistenceRealTestNum15mHz)
                self.errors['realTestHistogramMAE'].append(realTestHistogramMAE)
                self.errors['persistenceRealTestHistogramMAE'].append(persistenceRealTestHistogramMAE)

                ## MSE
                trainMSE               = numpy.sum((trainPredictions - trainCikti)**2) / trainCikti.shape[0]
                testMSE                = numpy.sum((testPredictions - testCikti )**2) / testCikti.shape[0]
                realTestMSE            = numpy.sum((realTestPredictions - realTestCikti )**2) / realTestCikti.shape[0]
                persistenceTrainMSE    = numpy.sum((trainCikti[:-1] - trainCikti[1:])**2) / (trainCikti.shape[0] - 1)
                persistenceTestMSE     = numpy.sum((testCikti[:-1] - testCikti[1:])**2) / (testCikti.shape[0] - 1)
                persistenceRealTestMSE = numpy.sum((realTestCikti[:-1] - realTestCikti[1:])**2) / (realTestCikti.shape[0] - 1)

                self.errors['trainMSE'].append(trainMSE)
                self.errors['testMSE'].append(testMSE)
                self.errors['realTestMSE'].append(realTestMSE)
                self.errors['persistenceTrainMSE'].append(persistenceTrainMSE)
                self.errors['persistenceTestMSE'].append(persistenceTestMSE)
                self.errors['persistenceRealTestMSE'].append(persistenceRealTestMSE)

                ## MAPE
                trainMAPE               = numpy.sum(numpy.abs(trainCikti - trainPredictions) / trainCikti * 100) / trainCikti.shape[0]
                testMAPE                = numpy.sum(numpy.abs(testCikti - testPredictions ) / testCikti * 100) / testCikti.shape[0]
                realTestMAPE            = numpy.sum(numpy.abs(realTestCikti - realTestPredictions ) / realTestCikti * 100) / realTestCikti.shape[0]
                persistenceTrainMAPE    = numpy.sum(numpy.abs(trainCikti[:-1] - trainCikti[1:]) / trainCikti[1:] * 100) / (trainCikti.shape[0] - 1)
                persistenceTestMAPE     = numpy.sum(numpy.abs(testCikti[:-1] - testCikti[1:]) / testCikti[1:] * 100) / (testCikti.shape[0] - 1)
                persistenceRealTestMAPE = numpy.sum(numpy.abs(realTestCikti[:-1] - realTestCikti[1:]) / realTestCikti[1:] * 100) / (realTestCikti.shape[0] - 1)

                self.errors['trainMAPE'].append(trainMAPE)
                self.errors['testMAPE'].append(testMAPE)
                self.errors['realTestMAPE'].append(realTestMAPE)
                self.errors['persistenceTrainMAPE'].append(persistenceTrainMAPE)
                self.errors['persistenceTestMAPE'].append(persistenceTestMAPE)
                self.errors['persistenceRealTestMAPE'].append(persistenceRealTestMAPE)





    def predict(self, data):
        if self.enableClassification:
            return (numpy.argmax(self.nn.predict(numpy.array( data )), axis=1)).reshape(-1, 1) - self.Ytrain.shape[1] // 2
        else:
            return self.scaler.inverse_transform(self.nn.predict(numpy.array( data ))).reshape(-1, 1)

    def saveCSV(self, saveCorrectValues=False, saveTrainValidation=False, saveNumber='', saveType='regular'):
        '''
        Save test set output to csv file as forecast values

        saveCorrectValues   : save also the correct output values for the forecast
        saveTrainValidation : also save train and validation test results and include the seperator date time information
        saveNumber          : extension to file name to save as
        saveType            : regular             -> save the date in first column, the data in the second column
                            : completePersistence -> save the date in first, forecast in second, real value in third, persistence in fourth
        '''
        if not saveTrainValidation:
            dates         = self.realTest_date_time[self.lb:]
            forecast      = self.predict(self.XrealTest).reshape(-1)
            correctOutput = self.scaler.inverse_transform(self.YrealTest).reshape(-1)
        else:
            dates         = self.train_date_time[self.lb:] + self.test_date_time[self.lb:] + self.realTest_date_time[self.lb:]
            forecast      = numpy.concatenate([self.predict(self.Xtrain).reshape(-1), self.predict(self.Xtest).reshape(-1), self.predict(self.XrealTest).reshape(-1)])
            correctOutput = numpy.concatenate([self.scaler.inverse_transform(self.Ytrain).reshape(-1), self.scaler.inverse_transform(self.Ytest).reshape(-1), self.scaler.inverse_transform(self.YrealTest).reshape(-1)])

        def regularSaveCSV():
            print('Saving forecast data to ' + self.directoryName + '/Forecast' + str(saveNumber) + '.csv')
            df = pd.DataFrame(data={
                            'dtm' : dates,
                            'f'   : forecast
                        })

            df.index = df['dtm']
            df = df.drop(['dtm'], axis=1)

            df.to_csv(self.directoryName + '/Forecast' + str(saveNumber) + '.csv')

            if saveTrainValidation:
                # Add comments
                with open(self.directoryName + '/Forecast' + str(saveNumber) + '.csv', 'r+') as f:
                    line = \
                    '# train set boundaries are from ' + str(self.train_date_time[self.lb]) + ' to ' + str(self.train_date_time[-1]) + \
                    ', validation set boundaries are from ' + str(self.test_date_time[self.lb]) + ' to ' + str(self.test_date_time[-1]) + \
                    ', test set boundaries are from ' + str(self.realTest_date_time[self.lb]) + ' to ' + str(self.realTest_date_time[-1])

                    content = f.read()
                    f.seek(0, 0)
                    f.write(line.rstrip('\r\n') + '\n' + content)

            if saveCorrectValues:
                print('Saving correct data to ' + self.directoryName + '/Forecast' + str(saveNumber) + '.csv')
                df = pd.DataFrame(data={
                                'dtm' : dates,
                                'f'   : correctOutput
                })

                df.index = df['dtm']
                df = df.drop(['dtm'], axis=1)

                df.to_csv(self.directoryName + '/CorrectOutput' + str(saveNumber) + '.csv')

                if saveTrainValidation:
                    # Add comments
                    with open(self.directoryName + '/CorrectOutput' + str(saveNumber) + '.csv', 'r+') as f:
                        line = \
                    '# train set boundaries are from ' + str(self.train_date_time[self.lb]) + ' to ' + str(self.train_date_time[-1]) + \
                    ', validation set boundaries are from ' + str(self.test_date_time[self.lb]) + ' to ' + str(self.test_date_time[-1]) + \
                    ', test set boundaries are from ' + str(self.realTest_date_time[self.lb]) + ' to ' + str(self.realTest_date_time[-1])

                        content = f.read()
                        f.seek(0, 0)
                        f.write(line.rstrip('\r\n') + '\n' + content)
        def completePersistenceSaveCSV():
            persistence = numpy.concatenate([self.trainData[self.lb-1:-1], self.testData[self.lb-1:-1], self.realTestData[self.lb-1:-1]])

            df = pd.DataFrame(data={
                            'Date Time'          : dates,
                            'Forecast Value'     : forecast,
                            'Ground Truth' : correctOutput,
                            'Persistence'  : persistence
                        })

            df.index = df['Date Time']
            df = df.drop(['Date Time'], axis=1)

            print('Saving forecast data to ' + self.directoryName + '/Complete' + str(saveNumber) + '.csv')
            df.to_csv(self.directoryName + '/Complete' + str(saveNumber) + '.csv')
            print('Saving forecast data to ' + self.directoryName + '/Complete' + str(saveNumber) + '.xlsx')
            df.to_excel(self.directoryName + '/Complete' + str(saveNumber) + '.xlsx', sheet_name='CompleteResults', float_format="%.4f")

            if saveTrainValidation:
                # Add comments
                with open(self.directoryName + '/Complete' + str(saveNumber) + '.csv', 'r+') as f:
                    line = \
                    '# train set boundaries are from ' + str(self.train_date_time[self.lb]) + ' to ' + str(self.train_date_time[-1]) + \
                    ', validation set boundaries are from ' + str(self.test_date_time[self.lb]) + ' to ' + str(self.test_date_time[-1]) + \
                    ', test set boundaries are from ' + str(self.realTest_date_time[self.lb]) + ' to ' + str(self.realTest_date_time[-1])

                    content = f.read()
                    f.seek(0, 0)
                    f.write(line.rstrip('\r\n') + '\n' + content)


        if saveType == 'regular':
            regularSaveCSV()
        elif saveType == 'completePersistence':
            completePersistenceSaveCSV()


    def plotError(self, bins=[50]):
        '''
        Plot error graph and other training - testing results.

        bins=[50] -> give number of bins for the histogram, iterable required
        '''
        # Choose the smallest MSE for validation
        selectedEpoch = numpy.argmin(self.errors['testMSE'])
        epochLen      = len(self.errors['testMSE'])
        print('Using ' + str(selectedEpoch) + '. epoch train for test set')

        xaxis = list(map(str, list(range(1,len(self.errors['trainMAE'])+1))))
        ### MAE
        plt.cla()
        plt.plot(xaxis, self.errors['trainMAE'])
        plt.plot(xaxis, self.errors['testMAE'])
        plt.plot([str(selectedEpoch)], [self.errors['realTestMAE'][selectedEpoch]], 'x', markersize=15)
        plt.plot(xaxis, self.errors['persistenceTrainMAE'])
        plt.plot(xaxis, self.errors['persistenceTestMAE'])
        plt.plot([str(selectedEpoch)], [self.errors['persistenceRealTestMAE'][0]], 'x', markersize=15)
        plt.title('MAE vs. Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend(['Train MAE', 'Validation MAE', 'Test MAE', 'Persistence Train MAE', 'Persistence Validation MAE', 'Persistence Test MAE'])
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.savefig(self.directoryName + '/MAEPlot.png')
        plt.cla()

        ### MSE
        plt.cla()
        plt.plot(xaxis, self.errors['trainMSE'])
        plt.plot(xaxis, self.errors['testMSE'])
        plt.plot([str(selectedEpoch)], [self.errors['realTestMSE'][selectedEpoch]], 'x', markersize=15)
        plt.text(selectedEpoch, self.errors['realTestMSE'][selectedEpoch], 'Test Error = ' + str(self.errors['realTestMSE'][selectedEpoch]))
        plt.plot(xaxis, self.errors['persistenceTrainMSE'])
        plt.plot(xaxis, self.errors['persistenceTestMSE'])
        plt.plot([str(selectedEpoch)], [self.errors['persistenceRealTestMSE'][0]], 'x', markersize=15)
        plt.text(selectedEpoch, self.errors['persistenceRealTestMSE'][0], 'Persistence Test Error = ' + str(self.errors['persistenceRealTestMSE'][0]))
        plt.text(selectedEpoch, self.errors['testMSE'][selectedEpoch], 'Min Validation Error = ' + str(numpy.min(self.errors['testMSE'])))
        plt.title('MSE vs. Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend(['Train MSE', 'Validation MSE', 'Test MSE', 'Persistence Train MSE', 'Persistence Validation MSE', 'Persistence Test MSE'])
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.savefig(self.directoryName + '/MSEPlot.png')
        plt.cla()

        ### MAPE
        plt.cla()
        plt.plot(xaxis, self.errors['trainMAPE'])
        plt.plot(xaxis, self.errors['testMAPE'])
        plt.plot([str(selectedEpoch)], [self.errors['realTestMAPE'][selectedEpoch]], 'x', markersize=15)
        plt.plot(xaxis, self.errors['persistenceTrainMAPE'])
        plt.plot(xaxis, self.errors['persistenceTestMAPE'])
        plt.plot([str(selectedEpoch)], [self.errors['persistenceRealTestMAPE'][0]], 'x', markersize=15)
        plt.title('MAPE vs. Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('MAPE')
        plt.legend(['Train MAPE', 'Validation MAPE', 'Test MAPE', 'Persistence Train MAPE', 'Persistence Validation MAPE', 'Persistence Test MAPE'])
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.savefig(self.directoryName + '/MAPEPlot.png')
        plt.cla()

        ### Error Histogram
        # Different bins depending on data...
        for bin_s in bins:
            ## Persistence
            plt.cla()
            plt.hist(self.errors['persistenceRealTestHistogramMAE'][selectedEpoch], bins=bin_s)
            plt.title('Persistence Model Test Set Error Histogram\n(Correct Values - Predictions)')
            plt.xlabel('Values')
            plt.ylabel('Occurences')
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.savefig(self.directoryName + '/'  + str(bin_s) + 'bins' + 'PersistenceErrorHistogram.png')
            plt.cla()
            ## LSTM Model
            plt.cla()
            plt.hist(self.errors['realTestHistogramMAE'][selectedEpoch], bins=bin_s)
            plt.title('LSTM Model Test Set Error Histogram\n(Correct Values - Predictions)')
            plt.xlabel('Values')
            plt.ylabel('Occurences')
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.savefig(self.directoryName + '/'  + str(bin_s) + 'bins' + 'LSTMModelErrorHistogram.png')
            plt.cla()


    def saveModel(self):
        # TODO -> eski loss degerleri...
        self.nn.save_weights(self.directoryName + '/weights')

        with open(self.directoryName + '/errors.pickle', 'wb') as f:
            pickle.dump(self.errors , f)


    def loadModel(self):
        if self.nn is None:
            self._buildModel()

        self.nn.load_weights(self.directoryName + '/weights')

        with open(self.directoryName + '/errors.pickle' , 'rb') as f:
            self.errors = pickle.load(f)

