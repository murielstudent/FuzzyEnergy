# FuzzyEnergy
Code for forecasting and anomaly detection using a Fuzzy Inference System

### Dependencies:

* [Skicit-learn](http://scikit-learn.org/) - Machine Learning in Python

Basics: numpy, os, sys, matplotlib, itertools, csv, random, python 3.4

## How to use

For examples, see the DEMOS directory

## tuning
you can tune the system by giving the plot=True argument to cluster(). You will see for each feature, 
a plot of the approximate data with the cluster centroids assigned: 

```
Ncentroids = cluster(data, target_col, number_of_centroids, plot=True)
```

## training:

### specify parameters

```
### the target column of your data
target_col = 0

### specify overlap of your sets, or the variance of the gaussian

overlap = 0.2	

### Gaussian, triangle or trapezoid

mf = 'Gaussian'

### specify for each feature the amount of centroids (can also be an integer if same for all)

Ncentroids = [7,7,5,5,7,7,7,7,7,7,7,7,7,7,7,7,7,5,5,7,7,7]

###the algorithm you want to use, WM or COR

algorithm = 'WM' 
```

### train the FIS with your specified parameters and a numpy array holding the data

```
RB, target_centroids , feature_centroids = FIS.train(data, target_col, mf, Ncentroids, overlap, method = algorithm, iterations = 1)
```

### write a FIS file

```
with open("name.FIS", "w") as fis_file:
    FIS.write(fis_file, algorithm, mf, overlap, target_centroids, feature_centroids, RB)
```

## testing:

### Read in the FIS file

```
method, mf, overlap, target_centroids, feature_centroids, RB = FIS.read(fis_file_path)
```

### test with numpy array test set

```
target_col = 0
RMSE, MAE = FIS.test(data, mf, overlap, target_centroids, feature_centroids, RB, target_col)
```


















