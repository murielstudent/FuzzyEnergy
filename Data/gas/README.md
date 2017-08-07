### Background info:

This dataset originates from:

De Nadai M, van Someren M. (2015) Short-term anomaly detection in gas consumption
through arima and artificial neural network forecast. Procs. 2015 ieee workshop on environmental,
energy, and structural monitoring systems (EESMS 2015). 250–255. New
York: IEEE. doi: 10.1109/EESMS.2015.7175886

The target column contains gas consumption of the Nicolaes Tulphuis (NTH) building of de University of Amsterdam:
Hogeschool van Amsterdam: http://www.hva.nl/over-de-hva/locaties/locaties.html

The weather data is retrieved from the KLM weather site:
Koninklijk Nederlands Meteorologisch Instituut: http://projects.knmi.nl/klimatologie/uurgegevens/selectie.cgi

### The features:

gas: The gas consumption in m3

before1: The gas consumption one hour before

before2: The gas consumption two hours before

peak5: The highest gas consumption measures in the pas 5 hours

sum5: The total gas consumption is the pas 5 hours

peak24: The highest gas consumption measures in the pas 24 hours

sum24: The total gas consumption is the pas 24 hours

mean15: The mean gas consumption in the pas 15 days

hour: The hour of the day 

FH: Wind speed

T: Temperature

Q: Radiation

U: Humidity

peak5T: The highest measured temperature in the past 5 hours

diffT: The difference in temperature with the preveous gas measure

std_day: Seasonal Trend Decomposition residuals, when your trend is within one day

std_day: Seasonal Trend Decomposition residuals, when your trend is within one year

kwh: The KwH consumption

kwhpeak5: The highest measured energy consumption in the pas 5 hours

day_year: The day of the year

day_week: The day of the week (note that holidays are annoted as sundays)

next_day_week: The day after