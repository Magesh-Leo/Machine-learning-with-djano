from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn import pipeline,preprocessing,metrics,model_selection,ensemble
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LinearRegression
import joblib


#Data Analysis

df=pd.read_csv(r'C:\Users\ELCOT\auto_mpg.csv')
mapper=DataFrameMapper([
                        (['cylinders', 'displacement','weight','acceleration', 'model year'],preprocessing.StandardScaler()),
                        (['origin'],preprocessing.OneHotEncoder())
])

pipeline_obj=pipeline.Pipeline([
                                ('mapper',mapper),
                                ('model',ensemble.RandomForestRegressor())

])

X=['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'model year', 'origin']
y=['mpg']

pipeline_obj.fit(df[X],df[y])
print(pipeline_obj.predict(df[X]))

joblib.dump(pipeline_obj,'RFMautompg.pkl')

model_reload=joblib.load('RFMautompg.pkl')

print(model_reload.score(df[X],df[y]))
# Create your views here.

def index(request):
    return render(request,'Mpgpredict.html')

def predictMPG(request):
    if request.method=="POST":
        temp={}
        temp['cylinders']=int(request.POST.get('cylinders'))
        temp['displacement']=int(request.POST.get('displacement'))
        temp['horsepower']=int(request.POST.get('horsepower'))
        temp['weight']=int(request.POST.get('weight'))
        temp['acceleration']=int(request.POST.get('acceleration'))
        temp['model year']=int(request.POST.get('model year'))
        temp['origin']=int(request.POST.get('origin'))

        testdata=pd.DataFrame({'x':temp}).transpose()
        score_val=model_reload.predict(testdata)[0]
        context={'score_val':score_val}
        return render(request,'Mpgpredict.html',context)
    else:
        return render(request,'Mpgpredict.html')