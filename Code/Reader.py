# Project file specific reader functions
# to read in the provided data set files


import csv
import numpy as np

def read_gas_data(path, file):
    '''
    Reads in the gas data and returns it
    as a float type numpy array.
    Formats are in csv and the first two
    columns (row ID and date) are removed
    '''
    data_train= []
    with open(path+file+'.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            try:
                data_train.append([float(x) for x in row[2:]])
            except: 
                header = row 
                print('The header: ')
                print(header[2:]) # print out the header to see features
                ValueError
    return np.array(data_train)


def read_elec_data(path, file):
    '''
    Reads in the data and returns it
    as a float type numpy array
    '''
    data_train= []
    with open(path+file) as file:
        reader = file.readlines()
        for row in reader:
            row = row.split()
            data_train.append([float(x) for x in row])
    return np.array(data_train)