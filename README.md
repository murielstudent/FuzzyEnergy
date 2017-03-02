# FuzzyEnergy
Code for the forecasting and anomaly detection using Fuzzy Logic

### Dependencies:

* [Skicit-learn](http://scikit-learn.org/) - Machine Learning in Python

numpy, itertools, csv, random, python 3.4

## How to use

## tuning
you can tune the system by giving the plot=True argument to cluster(). You will see for each feature, 
a plot of the approximate data with the cluster centroids assigned: 

```
Ncentroids = cluster(data, target_col, Ncentroids, plot=True)
```

## training:
### specify parameters

```
### what is the target column of your data


target_col = 0

### specify overlap of your sets, or the variance of the gaussian

overlap = 0.2	

### Gaussian, triangle or trapezoid

mf = 'Gaussian'

### specify for eaach feature the amount of centroids (can also be an integer if same for all)

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


















