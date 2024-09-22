from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv(r"C:\Users\pukam\Downloads\Mumbai1.csv")
data = data.drop(['Gas_Connection'], axis=1)
data = data.drop(['Intercom'], axis=1)
data = data.drop(['Swimming_Pool'], axis=1)
data = data.drop(['Indoor_Games'], axis=1)
data = data.drop(['Unnamed: 0'], axis=1)
data = data.drop(['Location'], axis=1)


def home(request):
    return render(request,"home.html")

def predict(request):
    return render(request,"predict.html")

def result(request):
    X = data.drop(['Price'], axis=1)  # input (all except price)
    Y = data['Price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    b=float(request.GET['n2'])
    c=float(request.GET['n3'])
    d=float(request.GET['h1'])
    f=float(request.GET['a1'])
    g=float(request.GET['a2'])
    h=float(request.GET['a3'])
    i=float(request.GET['a4'])
    j=float(request.GET['a5'])
    k=float(request.GET['a6'])
    l=float(request.GET['a7'])
    m=float(request.GET['a8'])
    n=float(request.GET['a9'])

    pred = model.predict(np.array([b,c,d,f,g,h,i,j,k,l,m,n]).reshape(1,-1))
    pred=round(pred[0])

    price="The predicted price is Rs."+str(pred)
    return render(request,"predict.html", {"result2":price})


def Graph (request):
    #Acc to area
    '''plt.figure(figsize=(16, 8))
    plt.xlabel("Area ($)")
    plt.ylabel("Price ($)")
    plt.scatter(data['Area'], data['Price'], color='maroon')
    plt.savefig('static/my_plot.png')
    plt.clf()
    #Acc to betrooms
    x = data['No. of Bedrooms']
    y = data['Price']
    plt.bar(x, y, color='maroon')
    plt.xlabel("No. of Bedrooms ($)")
    plt.ylabel("price ($)")
    plt.savefig('static/my_plot2.png')
    plt.clf()
    #Acc to Location

    #acc to new
    x = data['New/Resale']
    y = data['Price']
    plt.bar(x, y, color='maroon', width=0.3)
    plt.xlabel("Status ($)")
    plt.ylabel("Price ($)")
    plt.savefig('static/my_plot3.png')
    plt.clf()
    #heatmap
    plt.figure(figsize=(16, 16))
    sns.heatmap(data.corr(), annot=True)
    plt.savefig('static/my_plot4.png')
    plt.clf()'''
    return render(request,"Graph.html")

