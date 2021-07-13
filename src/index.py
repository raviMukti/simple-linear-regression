# author Ravi Mukti
# created 12-07-2021
# to install this library please run this command
# pip3 install numpy
# pip3 install matplotlib
# pip3 install pandas
# pylint: disable=unsubscriptable-object
# pylint: disable=no-member

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import os


# Import Data Set
BASEDIR = os.getcwd()
dataset = pd.read_csv(os.path.join(BASEDIR, 'resource/produksi_jagung.csv'))

print("### 5 Data Teratas ###")
print(dataset.head())
print("#######################")

var_x = dataset['Kilogram'].values.reshape(-1, 1)
var_y = dataset['Kwintal'].values.reshape(-1, 1)

plt.title("Pengaruh Pupuk Terhadap Hasil Panen")
plt.xlabel('Jumlah Pupuk(Kilogram)')
plt.ylabel('Hasil Panen(Kwintal)')
plt.scatter(dataset["Kilogram"], dataset["Kwintal"], c="red", marker="+")

reg = linear_model.LinearRegression()
reg.fit(dataset[['Kilogram']], dataset.Kwintal)

coeff = reg.coef_
intercept = reg.intercept_

print()
print(f"Intercept : {intercept}")
print(f"Koefisien : {coeff}")
print()

# y = mx + b
# m = coeffisien
# b = intercept
print("#######################")
print("Masukan Jumlah Pupuk dalam Kg")
inputPupuk = input()
 
try:
    inputPupuk = int(inputPupuk)
    pred = reg.predict([[inputPupuk]])
except:
    print("Input anda invalid")
    exit()

print()
print("Y = mx + b")
print(f"Y = {coeff}*{inputPupuk}+{intercept}")
print(f"Y = {pred}")
print()
print(f"Prediksi Hasil Panen : {pred}")

plt.plot(dataset.Kilogram, reg.predict(dataset[['Kilogram']]), color="blue")

print("Tampilkan Grafis Regresi Linear?(Y/n)")
tampilkan = input()
if tampilkan == "Y" or tampilkan == "y":
    plt.show()
elif tampilkan == "N" or tampilkan == "n":
    exit()
else:
    print("Input anda invalid")
    exit()