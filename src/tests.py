from src.frequencyForecast import FrequencyForecaster
from tensorflow.keras.optimizers import Adam
import numpy as np

def persistenceValues():
    '''
    Print persistence metric results
    '''
    ff = FrequencyForecaster('output/LoadTest/persistenceValues')
    ff.readPrepareData()
    Ytrain = ff.Ytrain
    Ytest  = ff.Ytest
    trainData = ff.trainData
    testData = ff.testData

    # MinMax scaler values
    print('MinMax scaled values : ')
    trainMSE = np.sum((Ytrain[:-1] - Ytrain[1:])**2) / Ytrain[1:].shape[0]
    testMSE  = np.sum((Ytest[:-1] - Ytest[1:])**2) / Ytest[1:].shape[0]
    print('{0:40s} : {1:3f} / {2:3f}'.format('Persistence model MSE for continuous outputs (train / test)', trainMSE, testMSE))

    trainMAE = np.sum(np.abs(Ytrain[:-1] - Ytrain[1:])) / Ytrain[1:].shape[0]
    testMAE  = np.sum(np.abs(Ytest[:-1] - Ytest[1:])) / Ytest[1:].shape[0]
    print('{0:40s} : {1:3f} / {2:3f}'.format('Persistence model MAE for continuous outputs (train / test)', trainMAE, testMAE))

    trainMAPE = np.sum(np.abs((Ytrain[1:] - Ytrain[:-1])/Ytrain[1:])*100) / Ytrain[1:].shape[0]
    testMAPE  = np.sum(np.abs((Ytest[1:] - Ytest[:-1])/Ytest[1:])*100) / Ytest[1:].shape[0]
    print('{0:40s} : {1:3f} / {2:3f}'.format('Persistence model MAPE for continuous outputs (train / test)', trainMAPE, testMAPE))

    # Real values
    print('Real values : ')
    trainData = np.array(trainData)
    testData  = np.array(testData)
    trainMSE = np.sum((trainData[:-1] - trainData[1:])**2) / trainData[1:].shape[0]
    testMSE  = np.sum((testData[:-1] - testData[1:])**2) / testData[1:].shape[0]
    print('{0:40s} : {1:3f} / {2:3f}'.format('Persistence model MSE for continuous outputs (train / test)', trainMSE, testMSE))

    trainMAE = np.sum(np.abs(trainData[:-1] - trainData[1:])) / trainData[1:].shape[0]
    testMAE  = np.sum(np.abs(testData[:-1] - testData[1:])) / testData[1:].shape[0]
    print('{0:40s} : {1:3f} / {2:3f}'.format('Persistence model MAE for continuous outputs (train / test)', trainMAE, testMAE))

    trainMAPE = np.sum(np.abs((trainData[1:] - trainData[:-1])/trainData[1:])*100) / trainData[1:].shape[0]
    testMAPE  = np.sum(np.abs((testData[1:] - testData[:-1])/testData[1:])*100) / testData[1:].shape[0]
    print('{0:40s} : {1:3f} / {2:3f}'.format('Persistence model MAPE for continuous outputs (train / test)', trainMAPE, testMAPE))
    print('\n')

def createModel():
    print('Creating SampleModel')
    ff = FrequencyForecaster('output/SampleModel')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True

    if os.path.exists('output/SampleModel/checkpoint'):
        ff.loadData()
        ff.loadModel()
    else:
        ff.readPrepareData()
        ff.train(epoch=5)
        ff.saveData()
        ff.saveModel()

    return ff

### Tests, since different tests will be done, they can be easily called by it's name!
def test1():
    ff = FrequencyForecaster('output/LoadTest/Continuous')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True

    ff.readPrepareData()
    ff.train(epoch=30, trainMode='predictTrain')

def test2():
    ff = FrequencyForecaster('output/LoadTest/DiscreteSTD')
    ff.enableClassification = True
    ff.binBoundaryType = 'StandardDeviation'
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True

    ff.readPrepareData()
    ff.train(epoch=30, trainMode='predictTrain')

def test3():
    ff = FrequencyForecaster('output/LoadTest/DiscreteGaussian')
    ff.enableClassification = True
    ff.binBoundaryType = 'Gaussian'
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True

    ff.readPrepareData()
    ff.train(epoch=30, trainMode='predictTrain')

def test4():
    '''
    Continuous forecast with single load feature I014_TSD
    '''
    ff = FrequencyForecaster('output/LoadTest/Continuous_IO14_TSD')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True

    ff.readPrepareData()
    ff.train(epoch=30, trainMode='predictTrain')

def test5():
    ff = FrequencyForecaster('output/mock1')

    ff.readPrepareData()
    ff.train(epoch=30, trainMode='predictTrain')


'''
Below will be to beat persistence model
Features are certain :
    Hour of Day
    Day of Week
    Past Frequency Data
    I014_TSD
train set      : 2 month = 60 days = 60 * 24 * 60 minutes
validation set : 18514 minutes
test set       : 18514 minutes
respectively.

TrainData2.csv
TestData2.csv files...
'''

# Data starting from 4th month
def test6():
    '''
    Add dropout - hasanDene
    '''
    ff = FrequencyForecaster('output/LoadTest/Continuous_IO14_TSD_dropout')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.dropout = 0.2

    ff.readPrepareData()
    ff.train(epoch=30, trainMode='predictTrain')


def test7():
    '''
    Add 1 layer
    '''
    ff = FrequencyForecaster('output/LoadTest/Continuous_IO14_TSD_3layers')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 3

    ff.readPrepareData()
    ff.train(epoch=30, trainMode='predictTrain')


def test8():
    '''
    Subtract 1 layer
    '''
    ff = FrequencyForecaster('output/LoadTest/Continuous_IO14_TSD_1layer')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1

    ff.readPrepareData()
    ff.train(epoch=30, trainMode='predictTrain')


def test9():
    '''
    Double neuron number
    '''
    ff = FrequencyForecaster('output/LoadTest/Continuous_IO14_TSD_doubleNeuron')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.nnum = ff.nnum * 2

    ff.readPrepareData()
    ff.train(epoch=30, trainMode='predictTrain')

def test10():
    '''
    Decrease learning rate
    1 layer architecture
    '''
    ff = FrequencyForecaster('output/LoadTest/Continuous_IO14_TSD_learningRate0003')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.optimizer = Adam(learning_rate=0.0003)

    ff.loadData()
    # ff.readPrepareData()
    # ff.saveData()
    ff.train(epoch=30, trainMode='predictTrain')
    return ff

def test11():
    '''
    Decrease num neurons
    '''
    ff = FrequencyForecaster('output/LoadTest/Continuous_IO14_TSD_decreaseNeurons')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.nnum = 96

    # ff.readPrepareData()
    ff.loadData(load_dir='output/LoadTest/Continuous_IO14_TSD_learningRate0003')
    ff.train(epoch=30, trainMode='predictTrain')
    return ff

def test12():
    '''
    Dropout
    Less layer
    '''
    ff = FrequencyForecaster('output/LoadTest/Continuous_IO14_TSD_dropoutLessLayer')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.dropout = 0.1
    ff.lnumber = 1

    # ff.readPrepareData()
    ff.loadData(load_dir='output/LoadTest/Continuous_IO14_TSD_learningRate0003')
    ff.train(epoch=30, trainMode='predictTrain')
    return ff

def test13():
    '''
    1 layer
    96 neuron
    '''
    ff = FrequencyForecaster('output/LoadTest/Continuous_IO14_TSD_96Neuron1Layer')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 96

    # ff.readPrepareData()
    ff.loadData(load_dir='output/LoadTest/Continuous_IO14_TSD_learningRate0003')
    ff.train(epoch=20, trainMode='predictTrain')
    ff.plotError()
    return ff

def test14():
    '''
    1 layer
    96 neuron
    0.0003 learning rate
    '''
    ff = FrequencyForecaster('output/LoadTest/Continuous_IO14_TSD_96Neuron1LayerLearningRate0003')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 96
    ff.optimizer = Adam(learning_rate=0.0003)

    ff.readPrepareData()
    # ff.loadData(load_dir='output/LoadTest/Continuous_IO14_TSD_learningRate0003')
    ff.train(epoch=20, trainMode='predictTrain')
    ff.plotError()
    return ff

def test15():
    '''
    1 layer
    96 neuron
    0.0003 learning rate
    stateful = False
    '''
    ff = FrequencyForecaster('output/LoadTest/test15Continuous_IO14_TSD_96Neuron1LayerLearningRate0003statefulFalse')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 96
    ff.stateful = False
    ff.optimizer = Adam(learning_rate=0.0003)

    # ff.readPrepareData()
    ff.loadData(load_dir='output/LoadTest/Continuous_IO14_TSD_learningRate0003')
    ff.train(epoch=20, trainMode='predictTrain')
    ff.plotError()
    return ff


def test16():
    '''
    1 layer
    72 neuron
    0.0003 learning rate
    '''
    ff = FrequencyForecaster('output/LoadTest/test16Continuous_IO14_TSD_72Neuron1LayerLearningRate0003')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 72
    ff.optimizer = Adam(learning_rate=0.0003)

    # ff.readPrepareData()
    ff.loadData(load_dir='output/LoadTest/Continuous_IO14_TSD_learningRate0003')
    ff.train(epoch=20, trainMode='predictTrain')
    ff.plotError()
    return ff

def test17():
    '''
    1 layer
    96 neuron
    0.0003 learning rate
    6 lookback
    '''
    ff = FrequencyForecaster('output/LoadTest/test17Continuous_IO14_TSD_96Neuron1LayerLearningRate0003lookBack6')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 96
    ff.optimizer = Adam(learning_rate=0.0003)
    ff.lb = 6

    ff.readPrepareData()
    ff.train(epoch=20, trainMode='predictTrain')
    ff.plotError()
    return ff

def test18():
    '''
    1 layer
    96 neuron
    0.0003 learning rate
    30 lookback
    '''
    ff = FrequencyForecaster('output/LoadTest/test18Continuous_IO14_TSD_96Neuron1LayerLearningRate0003lookBack30')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 96
    ff.optimizer = Adam(learning_rate=0.0003)
    ff.lb = 30

    ff.readPrepareData()
    ff.train(epoch=20, trainMode='predictTrain')
    ff.plotError()
    return ff

def test19():
    '''
    1 layer
    96 neuron
    0.0003 learning rate
    3 lookback
    '''
    ff = FrequencyForecaster('output/LoadTest/test19Continuous_IO14_TSD_96Neuron1LayerLearningRate0003lookBack3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 96
    ff.optimizer = Adam(learning_rate=0.0003)
    ff.lb = 3

    ff.readPrepareData()
    ff.train(epoch=20, trainMode='predictTrain')
    ff.plotError()
    return ff

def test20(testMode=False):
    '''
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    '''
    ff = FrequencyForecaster('output/LoadTest/test20Continuous_IO14_TSD_48Neuron1LayerLearningRate0003')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 48
    ff.optimizer = Adam(learning_rate=0.0003)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.loadData(load_dir='output/LoadTest/Continuous_IO14_TSD_learningRate0003')
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff


def test21():
    '''
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test21Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    ff.readPrepareData()
    ff.saveData()
    ff.train(epoch=10)
    ff.saveModel()

    ff.plotError()
    return ff

def test21load():
    '''
    Load test21 completely

    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test21Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    ff.loadData()
    ff.loadModel()

    ff.plotError()
    return ff

def test21CSV():
    '''
    Load test21 completely

    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test21CSVContinuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)


    ff.readPrepareData()
    for i in range(20):
        ff.train(epoch=1, trainMode='predictTrain')
        ff.saveCSV(saveCorrectValues=True, saveTrainValidation=True, saveNumber=i, saveType='completePersistence')
        ff.plotError()

    # Get the lowest error
    targetEpoch = np.argmin(ff.errors['testMSE'])

    # Use the lowest error file as a result
    print('Choosing the best in epoch ' + str(targetEpoch) + ' and writing to ' + ff.directoryName + '/Complete' + '.csv')
    os.replace(ff.directoryName + '/Complete' + str(targetEpoch) + '.csv', ff.directoryName + '/Complete.csv')
    print('Choosing the best in epoch ' + str(targetEpoch) + ' and writing to ' + ff.directoryName + '/Complete' + '.xlsx')
    os.replace(ff.directoryName + '/Complete' + str(targetEpoch) + '.xlsx', ff.directoryName + '/Complete.xlsx')

    return ff


def test22(testMode=False):
    '''
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0001 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test22Continuous_IO14_TSD_48Neuron1LayerLearningRate0001Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0001)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.loadData(load_dir='output/LoadTest/test21Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff

def test23(testMode=False):
    '''
    Test mode = test with smaller data on local computer
    1 layer
    24 neuron
    0.0003 learning rate
    lb = 3
    '''
    ff = FrequencyForecaster('output/LoadTest/test23Continuous_IO14_TSD_24Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 24
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.loadData(load_dir='output/LoadTest/test21Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff


def test24(testMode=False):
    '''
    Test mode = test with smaller data on local computer
    1 layer
    24 neuron
    0.0003 learning rate
    lb = 1
    '''
    ff = FrequencyForecaster('output/LoadTest/test24Continuous_IO14_TSD_24Neuron1LayerLearningRate0003Lookback1')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 24
    ff.lb      = 1
    ff.optimizer = Adam(learning_rate=0.0003)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.readPrepareData()
        ff.saveData()
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff


def test25(testMode=False):
    '''
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb = 1
    '''
    ff = FrequencyForecaster('output/LoadTest/test25Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback1')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 1
    ff.optimizer = Adam(learning_rate=0.0003)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.loadData(load_dir='output/LoadTest/test24Continuous_IO14_TSD_24Neuron1LayerLearningRate0003Lookback1')
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff

def test26(testMode=False):
    '''
    Test mode = test with smaller data on local computer
    1 layer
    12 neuron
    0.0003 learning rate
    lb = 3
    '''
    ff = FrequencyForecaster('output/LoadTest/test26Continuous_IO14_TSD_12Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 12
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.loadData(load_dir='output/LoadTest/test21Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff


def test27(testMode=False):
    '''
    Test mode = test with smaller data on local computer
    1 layer
    6 neuron
    0.0003 learning rate
    lb = 3
    '''
    ff = FrequencyForecaster('output/LoadTest/test27Continuous_IO14_TSD_6Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 6
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.loadData(load_dir='output/LoadTest/test21Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff


def test28(testMode=False):
    '''
    2017 en buyuk error icin
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test28DATA2Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.dataDirectory = 'data2'
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.readPrepareData()
        ff.saveData()
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff

def test28CSV():
    '''
    2017 en buyuk error icin
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test28CSVDATA2Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.dataDirectory = 'data2'
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    ff.readPrepareData()
    for i in range(20):
        ff.train(epoch=1, trainMode='predictTrain')
        ff.saveCSV(saveCorrectValues=True, saveTrainValidation=True, saveNumber=i, saveType='completePersistence')
        ff.plotError()

    # Get the lowest error
    targetEpoch = np.argmin(ff.errors['testMSE'])

    # Use the lowest error file as a result
    print('Choosing the best in epoch ' + str(targetEpoch) + ' and writing to ' + ff.directoryName + '/Complete' + '.csv')
    os.replace(ff.directoryName + '/Complete' + str(targetEpoch) + '.csv', ff.directoryName + '/Complete.csv')
    print('Choosing the best in epoch ' + str(targetEpoch) + ' and writing to ' + ff.directoryName + '/Complete' + '.xlsx')
    os.replace(ff.directoryName + '/Complete' + str(targetEpoch) + '.xlsx', ff.directoryName + '/Complete.xlsx')

    return ff

def test29(testMode=False):
    '''
    2018 en buyuk error icin
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test29DATA3Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.dataDirectory = 'data3'
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.readPrepareData()
        ff.saveData()
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff

def test29CSV():
    '''
    2018 en buyuk error icin
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test29CSVDATA3Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.dataDirectory = 'data3'
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)


    ff.readPrepareData()
    for i in range(20):
        ff.train(epoch=1, trainMode='predictTrain')
        ff.saveCSV(saveCorrectValues=True, saveTrainValidation=True, saveNumber=i, saveType='completePersistence')
        ff.plotError()

    # Get the lowest error
    targetEpoch = np.argmin(ff.errors['testMSE'])

    # Use the lowest error file as a result
    print('Choosing the best in epoch ' + str(targetEpoch) + ' and writing to ' + ff.directoryName + '/Complete' + '.csv')
    os.replace(ff.directoryName + '/Complete' + str(targetEpoch) + '.csv', ff.directoryName + '/Complete.csv')
    print('Choosing the best in epoch ' + str(targetEpoch) + ' and writing to ' + ff.directoryName + '/Complete' + '.xlsx')
    os.replace(ff.directoryName + '/Complete' + str(targetEpoch) + '.xlsx', ff.directoryName + '/Complete.xlsx')

    return ff

def test30(testMode=False):
    '''
    2017(en yuksek test error) train - validation -> 2018(en yuksek test error) test
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test30DATA4Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.dataDirectory = 'data4'
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.readPrepareData()
        ff.saveData()
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff

def test30CSV():
    '''
    2017(en yuksek test error) train - validation -> 2018(en yuksek test error) test
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test30CSVDATA4Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.dataDirectory = 'data4'
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    ff.readPrepareData()
    for i in range(20):
        ff.train(epoch=1, trainMode='predictTrain')
        ff.saveCSV(saveCorrectValues=True, saveTrainValidation=True, saveNumber=i, saveType='completePersistence')
        ff.plotError()

    # Get the lowest error
    targetEpoch = np.argmin(ff.errors['testMSE'])

    # Use the lowest error file as a result
    print('Choosing the best in epoch ' + str(targetEpoch) + ' and writing to ' + ff.directoryName + '/Complete' + '.csv')
    os.replace(ff.directoryName + '/Complete' + str(targetEpoch) + '.csv', ff.directoryName + '/Complete.csv')
    print('Choosing the best in epoch ' + str(targetEpoch) + ' and writing to ' + ff.directoryName + '/Complete' + '.xlsx')
    os.replace(ff.directoryName + '/Complete' + str(targetEpoch) + '.xlsx', ff.directoryName + '/Complete.xlsx')

    return ff

def test31(testMode=False):
    '''
    test on greater data size (90 days, greatest test error is selected as well)
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test31DATA5Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.dataDirectory = 'data5'
    ff.optimizer = Adam(learning_rate=0.0003)

    if testMode:
        ff.readPrepareData()
        ff.train(epoch=5, trainMode='predictTrain')
    else:
        ff.readPrepareData()
        ff.saveData()
        ff.train(epoch=20, trainMode='predictTrain')

    ff.plotError()
    return ff

def test31CSV():
    '''
    test on greater data size (90 days, greatest test error is selected as well)
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test31CSVDATA5Continuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.dataDirectory = 'data5'
    ff.optimizer = Adam(learning_rate=0.0003)

    ff.readPrepareData()
    for i in range(20):
        ff.train(epoch=1, trainMode='predictTrain')
        ff.saveCSV(saveCorrectValues=True, saveTrainValidation=True, saveNumber=i, saveType='completePersistence')
        ff.plotError()

    # Get the lowest error
    targetEpoch = np.argmin(ff.errors['testMSE'])

    # Use the lowest error file as a result
    print('Choosing the best in epoch ' + str(targetEpoch) + ' and writing to ' + ff.directoryName + '/Complete' + '.csv')
    os.replace(ff.directoryName + '/Complete' + str(targetEpoch) + '.csv', ff.directoryName + '/Complete.csv')
    print('Choosing the best in epoch ' + str(targetEpoch) + ' and writing to ' + ff.directoryName + '/Complete' + '.xlsx')
    os.replace(ff.directoryName + '/Complete' + str(targetEpoch) + '.xlsx', ff.directoryName + '/Complete.xlsx')

    return ff


def test32(enabledTests, testMode=False):
    '''
    Random search and log errors

    Possible tests : learningRates
                   : neuronNumber
                   : lookback
    '''
    def createModel(neuronNum=48, lookback=3, lr=0.0003, testMode=False):
        ff = FrequencyForecaster('output/LoadTest/test32')
        ff.enableDayofWeek                   = True
        ff.enableHourofDay                   = True
        ff.enableTimeFeaturesForecastHorizon = True
        ff.enableTimeFeaturesLookback        = False
        ff.enableLoadData                    = True
        ff.enableSelectiveLoadFeatures       = True

        ff.lb        = lookback


        if neuronNum is None and lr is None:
            ff.readPrepareData()
            ff.saveData()
            return

        ff.lnumber   = 1
        ff.nnum      = neuronNum
        ff.optimizer = Adam(learning_rate=lr)



        if lr == 0.0003 and neuronNum == 48:
            ff.readPrepareData()
        else:
            ff.loadData()

        if testMode:
            ff.train(epoch=3, trainMode='predictTrain')
        else:
            ff.train(epoch=20, trainMode='predictTrain')

        return ff

    if enabledTests == []:
        print('No enabled tests, exiting...')
        return None

    for elem in enabledTests:
        if elem not in ['learningRates', 'neuronNumber', 'lookback']:
            print('Possible misspelling in argument enabledTests, exiting...')
            return None

    if testMode:
        learningRates = [0.0001, 0.001, 0.1]
        neuronNumber  = [6, 16, 32]
        lookback      = [1, 3, 9]

        numTrials     = 4 # for each test
    else:
        '''
        >>> plt.yscale('log'); plt.plot(np.sort(np.exp(np.random.uniform(np.log(0.0001), np.log(0.01), 1000))), 'x'); plt.show() is linear!
        '''
        # learningRates = np.exp(np.random.uniform(np.log(0.0001), np.log(0.01), 36))
        # neuronNumber  = np.random.uniform(1, 500, 36)
        # lookback      = range(1, 37)
        learningRates  = [3e-4] #[1e-4, np.sqrt(10)*1e-4, 1e-3, np.sqrt(10)*1e-3, 1e-2]
        neuronNumber   = [12]   #[6, 12, 24, 48, 72, 96, 128]
        lookback       = [1, 2, 3, 6, 12, 30]

        numTrials     = 7 # for each test


    # Initialize model for first two trials
    # createModel(None, 3, None, testMode)


    # Calculate errors, standard deviations etc.
    meanMSE = {
        'learningRates' : [],
        'neuronNumber'  : [],
        'lookback'      : []
    }
    meanMAE = {
        'learningRates' : [],
        'neuronNumber'  : [],
        'lookback'      : []
    }
    meanMAPE = {
        'learningRates' : [],
        'neuronNumber'  : [],
        'lookback'      : []
    }
    stdMSE = {
        'learningRates' : [],
        'neuronNumber'  : [],
        'lookback'      : []
    }
    stdMAE = {
        'learningRates' : [],
        'neuronNumber'  : [],
        'lookback'      : []
    }
    stdMAPE = {
        'learningRates' : [],
        'neuronNumber'  : [],
        'lookback'      : []
    }


    # Save models
    models = {
        'lookback'      : [],
        'learningRates' : [],
        'neuronNumber'  : []
    }
    # Save errors
    errors = {
        'lookback'      : [],
        'learningRates' : [],
        'neuronNumber'  : []
    }
    if 'learningRates' in enabledTests:
        for lrIndex, lr in enumerate(learningRates):
            print('Learning rate = ' + str(lr))

            modelsLoc = []
            errorsLoc = []
            bestMSEValues  = [] # store best MSE values for validation set best epoch
            bestMAEValues  = []
            bestMAPEValues = []
            for i in range(numTrials):
                print('\t' + str(lrIndex) + ' of learning rates, trial ' + str(i))
                ff = createModel(lr=lr, testMode=testMode)

                # modelsLoc.append(ff)
                errorsLoc.append(ff.errors)


                bestMSEValues.append(np.min(ff.errors['testMSE']))
                bestMAEValues.append(np.min(ff.errors['testMAE']))
                bestMAPEValues.append(np.min(ff.errors['testMAPE']))


            models['learningRates'].append(modelsLoc)
            errors['learningRates'].append(errorsLoc)

            meanMSE['learningRates'].append(np.mean(bestMSEValues))
            meanMAE['learningRates'].append(np.mean(bestMAEValues))
            meanMAPE['learningRates'].append(np.mean(bestMAPEValues))

            # TODO!
            stdMSE['learningRates'].append(np.std(bestMSEValues))
            stdMAE['learningRates'].append(np.std(bestMAEValues))
            stdMAPE['learningRates'].append(np.std(bestMAPEValues))

    if 'neuronNumber' in enabledTests:
        for nnum in neuronNumber:
            print('Neuron number = ' + str(nnum))

            modelsLoc = []
            errorsLoc = []
            bestMSEValues  = [] # store best MSE values for validation set best epoch
            bestMAEValues  = []
            bestMAPEValues = []
            for i in range(numTrials):
                print('\tTrial ' + str(i))
                ff = createModel(neuronNum=nnum, testMode=testMode)

                # modelsLoc.append(ff)
                errorsLoc.append(ff.errors)

                bestMSEValues.append(np.min(ff.errors['testMSE']))
                bestMAEValues.append(np.min(ff.errors['testMAE']))
                bestMAPEValues.append(np.min(ff.errors['testMAPE']))

            errors['neuronNumber'].append(errorsLoc)
            models['neuronNumber'].append(modelsLoc)

            meanMSE['neuronNumber'].append(np.mean(bestMSEValues))
            meanMAE['neuronNumber'].append(np.mean(bestMAEValues))
            meanMAPE['neuronNumber'].append(np.mean(bestMAPEValues))

            # TODO!
            stdMSE['neuronNumber'].append(np.std(bestMSEValues))
            stdMAE['neuronNumber'].append(np.std(bestMAEValues))
            stdMAPE['neuronNumber'].append(np.std(bestMAPEValues))

    if 'lookback' in enabledTests:
        for lb in lookback:
            print('Lookback = ' + str(lb))

            modelsLoc = []
            errorsLoc = []
            bestMSEValues  = [] # store best MSE values for validation set best epoch
            bestMAEValues  = []
            bestMAPEValues = []
            for i in range(numTrials):
                print('\tTrial ' + str(i))
                ff = createModel(lookback=lb, testMode=testMode)

                # modelsLoc.append(ff)
                errorsLoc.append(ff.errors)

                bestMSEValues.append(np.min(ff.errors['testMSE']))
                bestMAEValues.append(np.min(ff.errors['testMAE']))
                bestMAPEValues.append(np.min(ff.errors['testMAPE']))

            errors['lookback'].append(errorsLoc)
            models['lookback'].append(modelsLoc)

            meanMSE['lookback'].append(np.mean(bestMSEValues))
            meanMAE['lookback'].append(np.mean(bestMAEValues))
            meanMAPE['lookback'].append(np.mean(bestMAPEValues))

            # TODO!
            stdMSE['lookback'].append(np.std(bestMSEValues))
            stdMAE['lookback'].append(np.std(bestMAEValues))
            stdMAPE['lookback'].append(np.std(bestMAPEValues))



    return {
        'errors' : errors,
        'models' : models,
        'parameters' : {
            'learningRates' : learningRates,
            'neuronNumber'  : neuronNumber,
            'lookback'      : lookback
        },
        'mean' : {
            'MSE'  : meanMSE,
            'MAE'  : meanMAE,
            'MAPE' : meanMAPE
        },
        'std' : {
            'MSE'  : stdMSE,
            'MAE'  : stdMAE,
            'MAPE' : stdMAPE
        }
    }


# ---------------------------------------------------------------------------------------------------- After 31 May 2020
def test33():
    '''
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    ff = FrequencyForecaster('output/LoadTest/test31GRUContinuous_IO14_TSD_48Neuron1LayerLearningRate0003Lookback3', forecastModel='GRU')
    ff.enableDayofWeek = True
    ff.enableHourofDay = True
    ff.enableTimeFeaturesForecastHorizon = True
    ff.enableTimeFeaturesLookback = False
    ff.enableLoadData = True
    ff.enableSelectiveLoadFeatures = True
    ff.dataDirectory = 'data'
    ff.lnumber = 1
    ff.nnum    = 48
    ff.lb      = 3
    ff.optimizer = Adam(learning_rate=0.0003)

    ff.readPrepareData()
    ff.saveData()
    ff.train(epoch=10)
    ff.saveModel()

    ff.plotError()
    return ff


ff = test33()
