import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
import sys

# Global variables...
lookback = 3

# 2017 data...
train2017 = pd.read_csv('data/data1/TrainData.csv')
test2017  = pd.read_csv('data/data1/TestData.csv')
trainData1      = np.array(train2017['f'])
trainData1Dates = list(pd.to_datetime(train2017['dtm']))
# validationData1 = np.array(pd.read_csv('data2/ValidationData.csv')['f'])
testData1       = np.array(test2017['f'])
testData1Dates  = list(pd.to_datetime(test2017['dtm']))

# 2018 (only test data)
test2018 = pd.read_csv('data/data2/TestData.csv')
testData2      = np.array(test2018['f'])
testData2Dates = list(pd.to_datetime(test2018['dtm']))

def persistance(data):
    predictions = data[lookback:-1]
    groundTruth = data[lookback + 1:]

    # MSE
    mse = np.sum((groundTruth - predictions)**2) / groundTruth.shape[0]

    # MAE
    mae = np.sum(np.abs(groundTruth - predictions)) / groundTruth.shape[0]

    # MAPE
    mape = np.sum(np.abs(groundTruth - predictions) / groundTruth * 100) / groundTruth.shape[0]

    return {
        'MSE'  : mse,
        'MAE'  : mae,
        'MAPE' : mape
    }

def statisticalMeanModel(trainData, trainDataDates, testData, testDataDates):
    ### 1) First, let's calculate average
    totalFrequency = np.zeros((7, 24))
    totalPoint     = np.zeros((7, 24))

    if not (len(trainData) == len(trainData) and len(testData) == len(testDataDates)):
        raise 'Failed to find correspondences...'

    for index in range(len(trainDataDates)):
        day_of_week = trainDataDates[index].weekday()
        hour_of_day = trainDataDates[index].hour

        totalFrequency[day_of_week][hour_of_day] = totalFrequency[day_of_week][hour_of_day] + trainData[index]
        totalPoint[day_of_week][hour_of_day]     = totalPoint[day_of_week][hour_of_day] + 1

    ## Find the average
    averageFrequencies = totalFrequency / totalPoint

    ### 2) Now, calculate the error
    total_mse  = 0
    total_mae  = 0
    total_mape = 0

    count_error = len(testData)

    for index in range(len(testData)):
        day_of_week = testDataDates[index].weekday()
        hour_of_day = testDataDates[index].hour

        prediction  = averageFrequencies[day_of_week][hour_of_day]
        groundTruth = testData[index]

        total_mse  = total_mse + (groundTruth - prediction)**2
        total_mae  = total_mae + np.abs(groundTruth - prediction)
        total_mape = total_mape + np.abs(groundTruth - prediction) / groundTruth * 100

    return {
        'MSE'  : total_mse / count_error,
        'MAE'  : total_mae / count_error,
        'MAPE' : total_mape / count_error
    }

# Persistance test data
persistanceTestDataset1 = persistance(testData1)
persistanceTestDataset2 = persistance(testData2)

# Statistical mean model
statisticalMeanModelDataset1 = statisticalMeanModel(trainData1, trainData1Dates, testData1, testData1Dates)
statisticalMeanModelDataset2 = statisticalMeanModel(trainData1, trainData1Dates, testData2, testData2Dates)

# LSTM model
lstmDataset1 = {'MSE' : 0.0012050999648141297, 'MAE' : 0.026109216592171473, 'MAPE' : 0.05221703659847376}
lstmDataset2 = {'MSE' : 0.0015051246825703336, 'MAE' : 0.029088190348774802, 'MAPE' : 0.05812498465654654}


print('persistence test 1', persistanceTestDataset1)
print('persistence test 2', persistanceTestDataset2)
print('stat. mean 1', statisticalMeanModelDataset1)
print('stat. mean 2', statisticalMeanModelDataset2)
print('lstm 1', lstmDataset1)
print('lstm 2', lstmDataset2)
# Model test data
# modelTestDataset1 =
