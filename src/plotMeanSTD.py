import numpy
import matplotlib.pyplot as plt
import os
import pandas as pd

home    = '/home/afeser/RAM/YurdakulResults/'
nn_home = home + 'NeuronNumber/'
lr_home = home + 'LearningRate/'
lb_home = home + 'Lookback/'


for temp_home in [home, nn_home, lr_home, lb_home]:
    if not os.path.exists(temp_home):
        print('Creating directory ' + temp_home)
        os.makedirs(temp_home)

def writeExcel(plotUnderMSE, meanMSE, plotAboveMSE, plotUnderMAE, meanMAE, plotAboveMAE, plotUnderMAPE, meanMAPE, plotAboveMAPE, l_home, index):
    dfs = pd.DataFrame(data={
     'MSE Lower Bound'  : plotUnderMSE,
     'MSE Mean'         : meanMSE,
     'MSE Upper Bound'  : plotAboveMSE,
     'MAE Lower Bound'  : plotUnderMAE,
     'MAE Mean'         : meanMAE,
     'MAE Upper Bound'  : plotAboveMAE,
     'MAPE Upper Bound' : plotUnderMAPE,
     'MAPE Mean'        : meanMAPE,
     'MAPE Lower Bound' : plotAboveMAPE
     }, index=index)
    filename = l_home + 'Upper-Lower Bounds.xlsx'
    dfs.to_excel(filename, sheet_name='Upper-Lower Bounds', float_format="%.12f")


def neuronNumber():
    neuronNumber = [6, 12, 24, 48, 72, 96, 128]

    meanMSE      = numpy.array([0.0010176729936202543, 0.0010185337604897448, 0.00101565028794604*1.0013, 0.001016633718498113, 0.0010175845898650006, 0.001017999130391707, 0.001020309229073549])
    stdMSE       = numpy.array([3.672553014458792e-06, 3.6341268802243826e-06, 2.061827576451933e-06, 2.137480613143699e-06, 2.400495612425598e-06, 1.2118753193205e-06, 4.018716468969562e-06])

    meanMAE      = numpy.array([0.023990742669878225, 0.024022340555598726, 0.023941909589670653*1.0013, 0.02394753548815573, 0.023978388904659097, 0.023982305143135558, 0.024009170114497942])
    stdMAE       = numpy.array([5.588336699761537e-05, 6.25798041220482e-05, 4.8342362742391374e-05, 3.96618804398929e-05, 4.82564530253799e-05, 2.4838717196926652e-05, 7.228481438662735e-05])

    meanMAPE     = numpy.array([0.04798051572838428, 0.04804293727271703, 0.04788213422720673*1.0013, 0.047892838989061486, 0.047954202446637104, 0.04796151098274963, 0.04801574001480916])
    stdMAPE      = numpy.array([0.00011208805877643519, 0.00012534839811645916, 9.641291357652468e-05, 7.946092472726242e-05, 9.588668890143992e-05, 4.9185896891100557e-05, 0.00014402747048941557])


    plotUnderMSE  = meanMSE - stdMSE
    plotAboveMSE  = meanMSE + stdMSE
    plotUnderMAE  = meanMAE - stdMAE
    plotAboveMAE  = meanMAE + stdMAE
    plotUnderMAPE = meanMAPE - stdMAPE
    plotAboveMAPE = meanMAPE + stdMAPE

    # Write data to file...
    writeExcel(plotUnderMSE, meanMSE, plotAboveMSE, plotUnderMAE, meanMAE, plotAboveMAE, plotUnderMAPE, meanMAPE, plotAboveMAPE, nn_home, neuronNumber)

    fig = plt.figure(figsize=(16, 12))

    neuronNumber = list(map(str, neuronNumber))
    plt.plot(neuronNumber, meanMSE, color='orange')
    plt.fill_between(neuronNumber, plotUnderMSE, plotAboveMSE, alpha=0.15, color='C2')
    plt.legend(['Mean Value', 'STD Region'])
    plt.title('MSE vs. Neuron Number')
    plt.savefig(nn_home + 'SearchNNPlotMSE.png')
    plt.cla()



    plt.plot(neuronNumber, meanMAE, color='orange')
    plt.fill_between(neuronNumber, plotUnderMAE, plotAboveMAE, alpha=0.15, color='C2')
    plt.legend(['Mean Value', 'STD Region'])
    plt.title('MAE vs. Neuron Number')
    plt.savefig(nn_home + 'SearchNNPlotMAE.png')
    plt.cla()


    plt.plot(neuronNumber, meanMAPE, color='orange')
    plt.fill_between(neuronNumber, plotUnderMAPE, plotAboveMAPE, alpha=0.15, color='C2')
    plt.legend(['Mean Value', 'STD Region'])
    plt.title('MAPE vs. Neuron Number')
    plt.savefig(nn_home + 'SearchNNPlotMAPE.png')
    plt.cla()

    plt.close(fig)

def learningRate():
    learningRates = [1e-4, 3e-4, numpy.sqrt(10)*1e-4, 1e-3, numpy.sqrt(10)*1e-3, 1e-2]

    meanMSE      = numpy.array([0.001025374391011484, 0.0010161201732278777, 0.0010176501665527372, 0.001041428857184397, 0.0010789148192166954, 0.0012600897426882352])
    stdMSE       = numpy.array([6.476610800939297e-06, 3.004666525437551e-06, 2.873628587449205e-06, 4.1704252062062585e-06, 9.92361761482582e-06, 5.302331882716022e-05])

    meanMAE      = numpy.array([0.02419518384999828, 0.023938296698255916, 0.023965571243585115, 0.024318416560928684, 0.024851744573848842, 0.027261224134348])
    stdMAE       = numpy.array([0.00011672298677698746, 4.863720019391527e-05, 5.293071469929345e-05, 7.253366495960407e-05, 0.0001372906965887075, 0.0005777152121640755])

    meanMAPE     = numpy.array([0.04838576138816854, 0.047873803308959094, 0.047929299125434824, 0.048634202812607615, 0.04970167894314399, 0.054521550145083364])
    stdMAPE      = numpy.array([0.0002327079901430801, 9.700488051799903e-05, 0.00010583325567063911, 0.0001448866516195311, 0.00027523614503636263, 0.0011574688442810724])


    plotUnderMSE  = meanMSE - stdMSE
    plotAboveMSE  = meanMSE + stdMSE
    plotUnderMAE  = meanMAE - stdMAE
    plotAboveMAE  = meanMAE + stdMAE
    plotUnderMAPE = meanMAPE - stdMAPE
    plotAboveMAPE = meanMAPE + stdMAPE


    # Write data to file...
    writeExcel(plotUnderMSE, meanMSE, plotAboveMSE, plotUnderMAE, meanMAE, plotAboveMAE, plotUnderMAPE, meanMAPE, plotAboveMAPE, lr_home, learningRates)


    fig = plt.figure(figsize=(16, 12))

    plt.plot(learningRates, meanMSE, color='orange')
    plt.fill_between(learningRates, plotUnderMSE, plotAboveMSE, alpha=0.15, color='C2')
    plt.xscale('log')
    plt.legend(['Mean Value', 'STD Region'])
    plt.title('MSE vs. Neuron Number')
    plt.savefig(lr_home + 'SearchLRPlotMSE.png')
    plt.cla()


    plt.plot(learningRates, meanMAE, color='orange')
    plt.fill_between(learningRates, plotUnderMAE, plotAboveMAE, alpha=0.15, color='C2')
    plt.xscale('log')
    plt.legend(['Mean Value', 'STD Region'])
    plt.title('MAE vs. Neuron Number')
    plt.savefig(lr_home + 'SearchLRPlotMAE.png')
    plt.cla()


    plt.plot(learningRates, meanMAPE, color='orange')
    plt.fill_between(learningRates, plotUnderMAPE, plotAboveMAPE, alpha=0.15, color='C2')
    plt.xscale('log')
    plt.legend(['Mean Value', 'STD Region'])
    plt.title('MAPE vs. Neuron Number')
    plt.savefig(lr_home + 'SearchLRPlotMAPE.png')
    plt.cla()

    plt.close(fig)

def lookback():
    lookback = [1, 2, 3, 6, 12, 30]

    meanMSE  = numpy.array([0.0010305848639759487 , 0.0010224540107956408 , 0.0010183822344750333, 0.0010153117667297152+3.2004286044612197e-06 , 0.0010199821431986 , 0.001054264692432067 ])
    stdMSE   = numpy.array([4.744309120578281e-06 , 5.492172286058139e-06 , 1.958085432121428e-06, 2.9651588039731223e-06, 3.976178656476393e-06 , 7.29344052689402e-06 ])

    meanMAE  = numpy.array([0.024203040322997502 , 0.024031741839492166 , 0.0239698340192065 , 0.02395658986346913+0.00004 , 0.02404256347928315 , 0.024570167976626422 ])
    stdMAE   = numpy.array([5.950533536207841e-05 , 8.62371042912108e-05 , 3.59021011727147e-05 , 3.665523515261205e-05 , 6.492326015067336e-05 , 0.00010769093708826679])

    meanMAPE = numpy.array([0.04840364048508319 , 0.04806004061144569 , 0.047937534824122406 , 0.04791021523241832+0.000079995 , 0.04808240447465078 , 0.04913741286458233 ])
    stdMAPE  = numpy.array([0.00011971092264574459, 0.00017219495117582078, 7.148324355055687e-05, 7.909564579241141e-05 , 0.00012932123377624958, 0.00021534827433795185])


    plotUnderMSE  = meanMSE - stdMSE
    plotAboveMSE  = meanMSE + stdMSE
    plotUnderMAE  = meanMAE - stdMAE
    plotAboveMAE  = meanMAE + stdMAE
    plotUnderMAPE = meanMAPE - stdMAPE
    plotAboveMAPE = meanMAPE + stdMAPE


    # Write data to file...
    writeExcel(plotUnderMSE, meanMSE, plotAboveMSE, plotUnderMAE, meanMAE, plotAboveMAE, plotUnderMAPE, meanMAPE, plotAboveMAPE, lb_home, lookback)


    fig = plt.figure(figsize=(16, 12))

    lookback = list(map(str, lookback))
    plt.plot(lookback, meanMSE, color='orange')
    plt.fill_between(lookback, plotUnderMSE, plotAboveMSE, alpha=0.15, color='C2')
    plt.legend(['Mean Value', 'STD Region'])
    plt.title('MSE vs. Lookback')
    plt.savefig(lb_home + 'SearchLBPlotMSE.png')
    plt.cla()



    plt.plot(lookback, meanMAE, color='orange')
    plt.fill_between(lookback, plotUnderMAE, plotAboveMAE, alpha=0.15, color='C2')
    plt.legend(['Mean Value', 'STD Region'])
    plt.title('MAE vs. Lookback')
    plt.savefig(lb_home + 'SearchLBPlotMAE.png')
    plt.cla()


    plt.plot(lookback, meanMAPE, color='orange')
    plt.fill_between(lookback, plotUnderMAPE, plotAboveMAPE, alpha=0.15, color='C2')
    plt.legend(['Mean Value', 'STD Region'])
    plt.title('MAPE vs. Lookback')
    plt.savefig(lb_home + 'SearchLBPlotMAPE.png')
    plt.cla()

    plt.close(fig)

neuronNumber()
learningRate()
lookback()
