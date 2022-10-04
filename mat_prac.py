import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""=========================== pandas ============="""

#to read data from a csv file
data = pd.read_csv("MELBOURNE_HOUSE_PRICES_LESS.csv")
#for accesing data use .fieldname operator or ["fieldname"]
#print(data.Address)
#print(data["Address"])

#used to caluculate statistics
#mean
print("mean",data["Price"].mean())
#mode
print("mode",data["Rooms"].mode())
#median
print("median",data["Price"].median())
#standard deviation
print("standard deviation",data["Price"].std())
#value counts
print(dict(data["Type"].value_counts()))

"""========================*******====================="""

"""=========================== matplotlib   ============="""
#line plot
#plt.plot(xpoints,ypoints) #it takes single as ypoints and x=[1,2..goes]
plt.figure(figsize=(5,5))
plt.title("sample line plot")
plt.plot(data["Price"],label="price",marker=".")
plt.legend()
#plt.xlabel()
#plt.ylabel()
plt.savefig("line_plot.png",dpi=300)
plt.show() # to show the plot

#histogram = frequency distributions
#simply it is a graph between value_counts
plt.hist(data["Price"],bins=5,edgecolor="red")
plt.show()

