import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def autoCorrelation():
    series = np.array(pd.read_csv('trainData.csv')['total_consumption'])

    firstDay = series[:24]

    correlation = []
    for i in range(240):
        correlation.append(np.corrcoef(firstDay, series[i:i+24])[0][1])


    plt.plot(correlation)
    plt.title('Correlation for Time Window 24')
    plt.xlabel('Time (hours)')
    plt.ylabel('Correlation Coefficient')
    plt.show()


    plt.plot(series)
    plt.title('Data Themselves')
    plt.xlabel('Time (hours)')
    plt.ylabel('Consumption (kW)')
    plt.show()


autoCorrelation()
