from src.frequencyForecast import FrequencyForecaster
from tensorflow.keras.optimizers import Adam
import numpy as np
from os.path import join
import time
import pdb
import pandas as pd
import pickle
from multiprocessing import Pool
import multiprocessing, logging
import itertools
import sys

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

mpl = multiprocessing.log_to_stderr()
mpl.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------------------------------- After 31 May 2020
def test33():
    '''
    Test mode = test with smaller data on local computer
    1 layer
    48 neuron
    0.0003 learning rate
    lb=3
    '''
    def inner_test(folder, foreMod):
        ff = FrequencyForecaster(folder, forecastModel=foreMod)
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

    inner_test('output/test33/GRU', 'GRU')
    inner_test('output/test33/LSTM', 'LSTM')
    inner_test('output/test33/SimpleRNN', 'SimpleRNN')

def test34():
    '''
    Best epoch test set errors. Configuration in test33.

    This function calculates the statistics STD, Mean for GRU, LSTM, SimpleRNN
    with many forecasted samples.

    The result is simply,

    std([forecasted values vector] - [real values vector])
    mean([forecasted values vector] - [real values vector])


    Output is saved to test34_all_errors.pickle. The output format is as follows:
    dictionary[forecast_model][data_set][epoch] ...
        ... ['test'][error_metric][std or mean] -> gives a float
        ... ['validation'] -> epoch validation calculated error inside FrequencyForecaster (MSE only)
        ... ['test_old_calculation'] -> epoch test calculated error inside FrequencyForecaster (MSE only)
    '''
    lookback  = 3
    max_epoch = 20
    def createModel(folder, foreMod, data_set):
        ff = FrequencyForecaster(folder, forecastModel=foreMod)
        ff.enableDayofWeek = True
        ff.enableHourofDay = True
        ff.enableTimeFeaturesForecastHorizon = True
        ff.enableTimeFeaturesLookback = False
        ff.enableLoadData = True
        ff.enableSelectiveLoadFeatures = True
        ff.dataDirectory = data_set
        ff.lnumber = 1
        ff.nnum    = 48
        ff.lb      = lookback
        ff.optimizer = Adam(learning_rate=0.0003)

        # Change to read preapare save if there is no load data present...
        # ff.readPrepareData()
        # ff.saveData()
        ff.loadData()

        return ff



    ### Global parameters...
    forecast_model_names = ['GRU', 'LSTM', 'SimpleRNN']
    data_set_names = ['data/data1', 'data/data2']

    # Calculate errors, standard deviations etc.
    dic_MSE = {
        'GRU'      : {}, # for 2 different data sets
        'LSTM'     : {},
        'SimpleRNN': {},
    }
    dic_MAE = {
        'GRU'      : {}, # for 2 different data sets
        'LSTM'     : {},
        'SimpleRNN': {},
    }
    dic_MAPE = {
        'GRU'      : {}, # for 2 different data sets
        'LSTM'     : {},
        'SimpleRNN': {},
    }
    min_indeces = {
        'GRU'      : {}, # for 2 different data sets
        'LSTM'     : {},
        'SimpleRNN': {},
    }

    run_times = {}
    all_data_calculated = {}
    for forecast_model in (forecast_model_names + ['persistence', 'statistical_mean']):
        run_times[forecast_model]           = {}
        all_data_calculated[forecast_model] = {}
        for data_set in data_set_names:
            run_times[forecast_model][data_set]           = {'train_time': 0, 'predict_time': 0}

            all_data_calculated[forecast_model][data_set] = [] # all errors inside

    for forecast_model in forecast_model_names:
        for data_set in data_set_names:
            ff = createModel(join('output/test34', forecast_model, data_set), forecast_model, data_set)

            allSTD  = {
                'MAE' : [],
                'MAPE' : [],
                'MSE' : []
            }
            allMEAN = {
                'MAE' : [],
                'MAPE' : [],
                'MSE' : []
            }
            train_time   = []
            predict_time = []
            for epoch_counter in range(max_epoch):

                # ff.train(epoch=1, trainMode='directTrain')
                train_time.append(ff.train(epoch=1))



                time_start = time.time()
                predictedOutput = ff.predict(ff.XrealTest)
                time_end = time.time()
                predict_time.append(time_end - time_start)

                realOutput = ff.scaler.inverse_transform(np.array(ff.YrealTest)).reshape(-1, 1)

                difference = np.abs(predictedOutput - realOutput)
                # MAE
                allSTD['MAE'].append(np.std(difference))
                allMEAN['MAE'].append(np.mean(difference))

                # MSE
                allSTD['MSE'].append(np.std(difference**2))
                allMEAN['MSE'].append(np.mean(difference**2))

                # MAPE
                allSTD['MAPE'].append(np.std(difference / realOutput * 100))
                allMEAN['MAPE'].append(np.mean(difference / realOutput * 100))

                all_data_calculated[forecast_model][data_set].append({
                    'test':{
                        'MAE':{
                            'std': allSTD['MAE'][-1],
                            'mean': allMEAN['MAE'][-1]
                        },
                        'MSE':{
                            'std': allSTD['MSE'][-1],
                            'mean': allMEAN['MSE'][-1]
                        },
                        'MAPE':{
                            'std': allSTD['MAPE'][-1],
                            'mean': allMEAN['MAPE'][-1]
                        }
                    },
                    'validation': ff.errors['testMSE'][epoch_counter],
                    'test_old_calculation': ff.errors['realTestMSE'][epoch_counter],
                    'min_val_error_index' : np.argmin(ff.errors['testMSE'])
                })


                # pdb.set_trace()

            min_val_error_index = np.argmin(ff.errors['testMSE'])

            min_indeces[forecast_model][data_set] = min_val_error_index


            dic_MAE [forecast_model][data_set] = {'std' : allSTD['MAE'] [min_val_error_index], 'mean' : allMEAN['MAE'] [min_val_error_index]}
            dic_MSE [forecast_model][data_set] = {'std' : allSTD['MSE'] [min_val_error_index], 'mean' : allMEAN['MSE'] [min_val_error_index]}
            dic_MAPE[forecast_model][data_set] = {'std' : allSTD['MAPE'][min_val_error_index], 'mean' : allMEAN['MAPE'][min_val_error_index]}

            run_times[forecast_model][data_set]['train_time']   = sum(train_time  [:min_val_error_index+1])
            run_times[forecast_model][data_set]['predict_time'] = predict_time[min_val_error_index]

            del ff
            # For correction...
            # print('Forecast model', forecast_model, 'data set', data_set, 'MSE', dic_MSE [forecast_model][data_set]['mean'], 'MAE', dic_MAE [forecast_model][data_set]['mean'],'MAPE', dic_MAPE [forecast_model][data_set]['mean'], 'train_time', sum(train_time  [:min_val_error_index+1]), 'predict_time', predict_time[min_val_error_index])


    # Statistical mean and persistence results
    for forecast_model in ['persistence', 'statistical_mean']:
        # Initialize
        dic_MAE[forecast_model] = {}
        dic_MSE[forecast_model] = {}
        dic_MAPE[forecast_model] = {}

        for data_set in data_set_names:
            test_data       = pd.read_csv(join(data_set, 'TestData.csv'))
            test_data_dates = list(pd.to_datetime(test_data['dtm']))
            test_data       = np.array(test_data['f'])

            train_data       = pd.read_csv(join(data_set, 'TrainData.csv'))
            train_data_dates = list(pd.to_datetime(train_data['dtm']))
            train_data       = np.array(train_data['f'])

            if forecast_model == 'persistence':
                prediction  = test_data[lookback:-1]
                groundTruth = test_data[lookback + 1:]

            elif forecast_model == 'statistical_mean':
                ### 1) First, let's calculate average
                totalFrequency = np.zeros((7, 24))
                totalPoint     = np.zeros((7, 24))


                for index in range(len(train_data_dates)):
                    day_of_week = train_data_dates[index].weekday()
                    hour_of_day = train_data_dates[index].hour

                    totalFrequency[day_of_week][hour_of_day] = totalFrequency[day_of_week][hour_of_day] + train_data[index]
                    totalPoint[day_of_week][hour_of_day]     = totalPoint[day_of_week][hour_of_day] + 1

                ## Find the average
                averageFrequencies = totalFrequency / totalPoint

                ### 2) Now, calculate the error
                groundTruth = []
                prediction  = []
                for index in range(len(test_data)):
                    day_of_week = test_data_dates[index].weekday()
                    hour_of_day = test_data_dates[index].hour

                    prediction.append(averageFrequencies[day_of_week][hour_of_day])
                    groundTruth.append(test_data[index])



            difference = np.abs(np.array(prediction) - np.array(groundTruth))



            # MAE
            dic_MAE[forecast_model][data_set] = {'std': np.std(difference), 'mean': np.mean(difference)}

            # MSE
            dic_MSE[forecast_model][data_set] = {'std': np.std(difference**2), 'mean': np.mean(difference**2)}

            # MAPE
            dic_MAPE[forecast_model][data_set] = {'std': np.std(difference / groundTruth * 100), 'mean': np.mean(difference / groundTruth * 100)}

            all_data_calculated[forecast_model][data_set] = {
                'MAE':{
                    'std': dic_MAE[forecast_model][data_set]['std'],
                    'mead': dic_MAE[forecast_model][data_set]['mean']
                },
                'MSE':{
                    'std': dic_MSE[forecast_model][data_set]['std'],
                    'mead': dic_MSE[forecast_model][data_set]['mean']
                },
                'MAPE':{
                    'std': dic_MAPE[forecast_model][data_set]['std'],
                    'mead': dic_MAPE[forecast_model][data_set]['mean']
                },
            }

    explanation_comment = """
    Output is saved to test34_all_errors.pickle. The output format is as follows:
    dictionary[forecast_model][data_set][epoch] ...
        ... ['test'][error_metric][std or mean] -> gives a float
        ... ['validation'] -> epoch validation calculated error inside FrequencyForecaster (MSE only)
        ... ['test_old_calculation'] -> epoch test calculated error inside FrequencyForecaster (MSE only)
        ... ['min_val_error_index'] -> epoch that minimum error has been seen
            """
    pickle.dump([all_data_calculated, explanation_comment], open('test34_all_errors.pickle', 'wb'))




    print('----------------------Results----------------------')
    for forecast_model in (forecast_model_names + ['persistence', 'statistical_mean']):
        print('Forecast model', forecast_model)
        for data_set in data_set_names:
            print('\tData set', data_set)

            print('\t\tStandard Deviation')
            print('\t\t\tMAE  : ', str(dic_MAE [forecast_model][data_set]['std']))
            print('\t\t\tMSE  : ', str(dic_MSE [forecast_model][data_set]['std']))
            print('\t\t\tMAPE : ', str(dic_MAPE[forecast_model][data_set]['std']))

            print('\t\tMean')
            print('\t\t\tMAE  : ', str(dic_MAE [forecast_model][data_set]['mean']))
            print('\t\t\tMSE  : ', str(dic_MSE [forecast_model][data_set]['mean']))
            print('\t\t\tMAPE : ', str(dic_MAPE[forecast_model][data_set]['mean']))

            if forecast_model in forecast_model_names:
                print('\t\tMinimum Error Epoch', str(min_indeces[forecast_model][data_set]))

                print('\t\tTimes')
                print('\t\t\tTrain Time =', run_times[forecast_model][data_set]['train_time'])
                print('\t\t\tPredict Time =', run_times[forecast_model][data_set]['predict_time'])

            print()

def test35_async_model_create(argss):
    forecast_model, data_set, test_case_number = argss

    lookback  = 3

    def createModel(folder, foreMod, data_set, test_case_number):
        ff = FrequencyForecaster(folder, forecastModel=foreMod)
        ff.enableDayofWeek = True
        ff.enableHourofDay = True
        ff.enableTimeFeaturesForecastHorizon = True
        ff.enableTimeFeaturesLookback = False
        ff.enableLoadData = True
        ff.enableSelectiveLoadFeatures = True
        ff.dataDirectory = data_set
        ff.lnumber = 1
        ff.nnum    = 48
        ff.lb      = lookback
        ff.optimizer = Adam(learning_rate=0.0003)

        # Change to read preapare save if there is no load data present...
        if not ff.loadData():
            if test_case_number == 1:
                ff.readPrepareData()
                backup_train_data = ff.trainData
                backup_train_X    = ff.Xtrain
                backup_train_Y    = ff.Ytrain

                backup_val_data = ff.testData
                backup_val_X    = ff.Xtest
                backup_val_Y    = ff.Ytest

                ff.readPrepareData(input_mask={'name': 'normal', 'sigma': 0.005/3})
                ff.trainData = np.concatenate((backup_train_data, ff.trainData))
                ff.Xtrain    = np.concatenate((backup_train_X, ff.Xtrain))
                ff.Ytrain    = np.concatenate((backup_train_Y, ff.Ytrain))

                ff.testData = backup_val_data
                ff.Xtest    = backup_val_X
                ff.Ytest    = backup_val_Y

                ff.saveData()

            elif test_case_number == 2:
                ff.readPrepareData(input_mask={'name': 'normal', 'sigma': 0.005/3})

                ff.saveData()

            else:
                ff.readPrepareData()
                backup_train_data = ff.trainData
                backup_train_X    = ff.Xtrain
                backup_train_Y    = ff.Ytrain

                backup_val_data = ff.testData
                backup_val_X    = ff.Xtest
                backup_val_Y    = ff.Ytest

                ff.readPrepareData(input_mask={'name': 'normal', 'sigma': 0.005/3})
                ff.trainData = backup_train_data
                ff.Xtrain    = backup_train_X
                ff.Ytrain    = backup_train_Y

                ff.testData = backup_val_data
                ff.Xtest    = backup_val_X
                ff.Ytest    = backup_val_Y

                ff.saveData()

        return ff


    ff = createModel(join('output/test35', forecast_model, data_set, str(test_case_number)), forecast_model, data_set, test_case_number)

    return ff

def test35_async_calls(argss):
    forecast_model, data_set, test_case_number, ff, max_epoch = argss

    allSTD  = {
        'MAE' : [],
        'MAPE' : [],
        'MSE' : []
    }
    allMEAN = {
        'MAE' : [],
        'MAPE' : [],
        'MSE' : []
    }
    train_time   = []
    predict_time = []
    all_data_calculated = []
    for epoch_counter in range(max_epoch):

        # ff.train(epoch=1, trainMode='directTrain')
        train_time.append(ff.train(epoch=1))



        time_start = time.time()
        predictedOutput = ff.predict(ff.XrealTest)
        time_end = time.time()
        predict_time.append(time_end - time_start)

        realOutput = ff.scaler.inverse_transform(np.array(ff.YrealTest)).reshape(-1, 1)

        difference = np.abs(predictedOutput - realOutput)
        # MAE
        allSTD['MAE'].append(np.std(difference))
        allMEAN['MAE'].append(np.mean(difference))

        # MSE
        allSTD['MSE'].append(np.std(difference**2))
        allMEAN['MSE'].append(np.mean(difference**2))

        # MAPE
        allSTD['MAPE'].append(np.std(difference / realOutput * 100))
        allMEAN['MAPE'].append(np.mean(difference / realOutput * 100))

        all_data_calculated.append({
            'test':{
                'MAE':{
                    'std': allSTD['MAE'][-1],
                    'mean': allMEAN['MAE'][-1]
                },
                'MSE':{
                    'std': allSTD['MSE'][-1],
                    'mean': allMEAN['MSE'][-1]
                },
                'MAPE':{
                    'std': allSTD['MAPE'][-1],
                    'mean': allMEAN['MAPE'][-1]
                }
            },
            'validation': ff.errors['testMSE'][epoch_counter],
            'test_old_calculation': ff.errors['realTestMSE'][epoch_counter],
            'min_val_error_index' : np.argmin(ff.errors['testMSE'])
        })


        # pdb.set_trace()

    min_val_error_index = np.argmin(ff.errors['testMSE'])


    del ff



    return {
        'forecast_model': forecast_model,
        'data_set': data_set,
        'all_data_calculated': all_data_calculated,
        'min_indeces': min_val_error_index,
        'dic_MAE': {'std' : allSTD['MAE'] [min_val_error_index], 'mean' : allMEAN['MAE'] [min_val_error_index]},
        'dic_MSE': {'std' : allSTD['MSE'] [min_val_error_index], 'mean' : allMEAN['MSE'] [min_val_error_index]},
        'dic_MAPE': {'std' : allSTD['MAPE'][min_val_error_index], 'mean' : allMEAN['MAPE'][min_val_error_index]},
        'run_times_train': sum(train_time  [:min_val_error_index+1]),
        'run_times_predict': predict_time[min_val_error_index]
    }
def test35():
    '''
    Best epoch test set errors. Configuration in test33. Gaussian noise trials
    done, noise is added to train, test input data. Following 3 tests will be
    done:
        1)
            - Train : [correct_train_data, noisy_train_data] -> [correct_train_labels]
            - Test  : [noisy_test_data]                      -> [correct_test_labels]
        2)
            - Train : [noisy_train_data] -> [correct_train_labels]
            - Test  : [noisy_test_data]  -> [correct_test_labels]
        3)
            - Train : [correct_train_data] -> [correct_train_labels]
            - Test  : [noisy_test_data]    -> [correct_test_labels]
        4)
            - Train : [correct_train_data] -> [correct_train_labels]
            - Test  : [correct_test_data] -> [correct_test_labels]
            # note this is a check test




    This function calculates the statistics STD, Mean for GRU, LSTM, SimpleRNN
    with many forecasted samples.

    The result is simply,

    std([forecasted values vector] - [real values vector])
    mean([forecasted values vector] - [real values vector])


    Output is saved to test35_all_errors.pickle. More information is in the
    pickle object.
    '''
    ### Global parameters...
    max_epoch = 1
    forecast_model_names = ['GRU', 'LSTM', 'SimpleRNN']
    data_set_names = ['data/data1', 'data/data2']
    all_test_numbers = [1,2,3]


    run_times           = {}
    all_data_calculated = {}
    dic_MSE             = {}
    dic_MAE             = {}
    dic_MAPE            = {}
    min_indeces         = {}
    for test_number in all_test_numbers:
        run_times[test_number]           = {}
        all_data_calculated[test_number] = {}
        dic_MSE[test_number]             = {}
        dic_MAE[test_number]             = {}
        dic_MAPE[test_number]            = {}
        min_indeces[test_number]         = {}
        for forecast_model in (forecast_model_names + ['persistence', 'statistical_mean']):
            run_times[test_number][forecast_model]           = {}
            all_data_calculated[test_number][forecast_model] = {}
            dic_MSE[test_number][forecast_model]             = {}
            dic_MAE[test_number][forecast_model]             = {}
            dic_MAPE[test_number][forecast_model]            = {}
            min_indeces[test_number][forecast_model]         = {}


    # test35_async_calls('LSTM', 'data/data1', 1)
    p = Pool(len(forecast_model_names)*len(data_set_names))

    input_argss = [(model_data[1], model_data[2], model_data[0]) for model_data in itertools.product([1,2,3], forecast_model_names, data_set_names)]
    all_models = p.map(test35_async_model_create, input_argss)

    input_argss = [(model_data[1], model_data[2], model_data[0], all_models[counter], max_epoch) for counter, model_data in enumerate(itertools.product(all_test_numbers, forecast_model_names, data_set_names))]
    all_included = p.map(test35_async_calls, input_argss)

    pdb.set_trace()

    list_counter = 0
    for test_number in all_test_numbers:
        for forecast_model in forecast_model_names + ['persistence', 'statistical_mean']:
            for data_set in data_set_names:
                if forecast_model in ['persistence', 'statistical_mean']:
                    test_data       = all_models[list_counter].realTestData
                    test_data_dates = all_models[list_counter].realTest_date_time

                    train_data       = all_models[list_counter].trainData
                    train_data_dates = all_models[list_counter].train_date_time

                    if forecast_model == 'persistence':
                        prediction  = test_data[lookback:-1]
                        groundTruth = test_data[lookback + 1:]

                    elif forecast_model == 'statistical_mean':
                        ### 1) First, let's calculate average
                        totalFrequency = np.zeros((7, 24))
                        totalPoint     = np.zeros((7, 24))


                        for index in range(len(train_data_dates)):
                            day_of_week = train_data_dates[index].weekday()
                            hour_of_day = train_data_dates[index].hour

                            totalFrequency[day_of_week][hour_of_day] = totalFrequency[day_of_week][hour_of_day] + train_data[index]
                            totalPoint[day_of_week][hour_of_day]     = totalPoint[day_of_week][hour_of_day] + 1

                        ## Find the average
                        averageFrequencies = totalFrequency / totalPoint

                        ### 2) Now, calculate the error
                        groundTruth = []
                        prediction  = []
                        for index in range(len(test_data)):
                            day_of_week = test_data_dates[index].weekday()
                            hour_of_day = test_data_dates[index].hour

                            prediction.append(averageFrequencies[day_of_week][hour_of_day])
                            groundTruth.append(test_data[index])



                    difference = np.abs(np.array(prediction) - np.array(groundTruth))



                    # MAE
                    dic_MAE[test_number][forecast_model][data_set] = {'std': np.std(difference), 'mean': np.mean(difference)}

                    # MSE
                    dic_MSE[test_number][forecast_model][data_set] = {'std': np.std(difference**2), 'mean': np.mean(difference**2)}

                    # MAPE
                    dic_MAPE[test_number][forecast_model][data_set] = {'std': np.std(difference / groundTruth * 100), 'mean': np.mean(difference / groundTruth * 100)}

                    all_data_calculated[test_number][forecast_model][data_set] = [{
                        'MAE':{
                            'std': dic_MAE[all_data_calculated, explanation_comment][forecast_model][data_set]['std'],
                            'mead': dic_MAE[all_data_calculated, explanation_comment][forecast_model][data_set]['mean']
                        },
                        'MSE':{
                            'std': dic_MSE[all_data_calculated, explanation_comment][forecast_model][data_set]['std'],
                            'mead': dic_MSE[all_data_calculated, explanation_comment][forecast_model][data_set]['mean']
                        },
                        'MAPE':{
                            'std': dic_MAPE[all_data_calculated, explanation_comment][forecast_model][data_set]['std'],
                            'mead': dic_MAPE[all_data_calculated, explanation_comment][forecast_model][data_set]['mean']
                        },
                        'validation': -1,
                        'test_old_calculation': -1,
                        'min_val_error_index' : 0
                    }] * max_epoch


                else:
                    elem = all_included[list_counter]
                    list_counter = list_counter + 1


                    all_data_calculated[test_number][forecast_model][data_set] = elem['all_data_calculated']

                    min_indeces[test_number][forecast_model][data_set] = elem['min_indeces']

                    dic_MAE[test_number][forecast_model][data_set]  = elem['dic_MAE']
                    dic_MSE[test_number][forecast_model][data_set]  = elem['dic_MSE']
                    dic_MAPE[test_number][forecast_model][data_set] = elem['dic_MAPE']

                    run_times[test_number][forecast_model][data_set] = {'train_time': elem['run_times_train'], 'predict_time': elem['run_times_predict']}









    print('----------------------Results----------------------')
    for test_number in all_test_numbers:
        print('Test ', test_number)
        for forecast_model in (forecast_model_names + ['persistence', 'statistical_mean']):
            print('\tForecast model', forecast_model)
            for data_set in data_set_names:
                print('\t\tData set', data_set)

                print('\t\t\tStandard Deviation')
                print('\t\t\t\tMAE  : ', str(dic_MAE [test_number][forecast_model][data_set]['std']))
                print('\t\t\t\tMSE  : ', str(dic_MSE [test_number][forecast_model][data_set]['std']))
                print('\t\t\t\tMAPE : ', str(dic_MAPE[test_number][forecast_model][data_set]['std']))

                print('\t\t\tMean')
                print('\t\t\t\tMAE  : ', str(dic_MAE [test_number][forecast_model][data_set]['mean']))
                print('\t\t\t\tMSE  : ', str(dic_MSE [test_number][forecast_model][data_set]['mean']))
                print('\t\t\t\tMAPE : ', str(dic_MAPE[test_number][forecast_model][data_set]['mean']))

                if forecast_model in forecast_model_names:
                    print('\t\t\tMinimum Error Epoch', str(min_indeces[test_number][forecast_model][data_set]))

                    print('\t\t\tTimes')
                    print('\t\t\t\tTrain Time =', run_times[test_number][forecast_model][data_set]['train_time'])
                    print('\t\t\t\tPredict Time =', run_times[test_number][forecast_model][data_set]['predict_time'])

                print()

    explanation_comment = """
    Output is saved to test35_all_errors.pickle. The output format is as follows:
    dictionary[test_number][forecast_model][data_set][epoch] ...
        ... ['test'][error_metric][std or mean] -> gives a float
        ... ['validation'] -> epoch validation calculated error inside FrequencyForecaster (MSE only)
        ... ['test_old_calculation'] -> epoch test calculated error inside FrequencyForecaster (MSE only)
        ... ['min_val_error_index'] -> epoch that minimum error has been seen
            """


    pickle.dump([all_data_calculated, explanation_comment], open('test35_all_errors.pickle', 'wb'))



ff = test35()
