from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from math import*
from matplotlib import pyplot as plt   
from matplotlib import style
import numpy as np
from scipy import stats as st
import random
import statistics as stat
import pandas as pd
import seaborn as sns
import matplotlib
import random

import scipy
from scipy.optimize import curve_fit 
from numpy import array, exp
from statistics import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import curve_fit
import itertools
from sklearn.cluster import KMeans
import seaborn.objects as so
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib import cbook
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score


matplotlib.use('TkAgg')

win = tk.Tk()
win.title("ТАГАНСКОЕ ЗАПАД")
win.geometry("1000x1300")
icon = tk.PhotoImage(file="C:\\Users\\79284\\Pictures\\бентонит1.png")
win.iconphoto(True, icon)

##scale_widget_5 = tk.Scale(win, orient="horizontal", resolution=1, from_=5267300, to=5267850)
##label_1 = tk.Label(win, text="X")
##label_1.grid(row=30, column=0)
##scale_widget_5.grid(row=31, column=0)
##
##scale_widget_6 = tk.Scale(win, orient="horizontal", resolution=1, from_=14716100, to=14717400)
##label_1 = tk.Label(win, text="Y")
##label_1.grid(row=32, column=0)
##scale_widget_6.grid(row=33, column=0)
##
##
##scale_widget_3 = tk.Scale(win, orient="horizontal", resolution=1, from_=14716100, to=14717400)
##label_1 = tk.Label(win, text="Y min")
##label_1.grid(row=30, column=1)
##scale_widget_3.grid(row=31, column=1)
##
##scale_widget_4 = tk.Scale(win, orient="horizontal", resolution=1, from_=14716100, to=14717400)
##label_1 = tk.Label(win, text="Y max")
##label_1.grid(row=32, column=1)
##scale_widget_4.grid(row=33, column=1)
##
##scale_widget_1 = tk.Scale(win, orient="horizontal", resolution=1, from_=5267300, to=5267850)
##label_1 = tk.Label(win, text="X min")
##label_1.grid(row=30, column=9)
##scale_widget_1.grid(row=31, column=9)
##
##scale_widget_2 = tk.Scale(win, orient="horizontal", resolution=1, from_=5267300, to=5267850)
##label_1 = tk.Label(win, text="X max")
##label_1.grid(row=32, column=9)
##scale_widget_2.grid(row=33, column=9)



res101 = tk.Label(win, text = " ")
res101.grid(row = 5, column = 0)
res102 = tk.Label(win, text = " ")
res102.grid(row = 6, column = 0)
res103 = tk.Label(win, text = " ")
res103.grid(row = 7, column = 0)
res104 = tk.Label(win, text = " ")
res104.grid(row = 8, column = 0)
res105 = tk.Label(win, text = " ")
res105.grid(row = 9, column = 0)
res = tk.Label(win, text = " ")
res.grid(row = 11, column = 0)
#res1 = tk.Label(win, text = "-")
##res1.grid(row = 11, column = 10)
res2 = tk.Label(win, text = "")
res2.grid(row = 12, column = 0)
res3 = tk.Label(win, text = " ", fg='red', font = ("Arial Bold", 9))
res3.grid(row = 13, column = 10)
##res22 = tk.Label(win, text = " ")
##res22.grid(row = 14, column = 0)
lbl = Label(win, text="1 - Глубина залегания, м", font=("Arial Bold", 12), fg='blue')  
lbl.grid(column=0, row=14)  
num0 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num0.grid(row = 15, column = 0)
lbl = Label(win, text="2 - Влажность глины, %", font=("Arial Bold", 12), fg='blue')  
lbl.grid(column=0, row=16)  
num1 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num1.grid(row = 17, column = 0)
lbl = Label(win, text="3 - Песок, %", font=("Arial Bold", 12), fg='blue')  
lbl.grid(column=0, row=18)  
num2 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num2.grid(row = 19, column = 0)
lbl = Label(win, text="4 - Индекс набухания, мл/2г", font=("Arial Bold", 12), fg='blue')  
lbl.grid(column=0, row=20)  
num3 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num3.grid(row = 21, column = 0)
lbl = Label(win, text="5 - Электропроводность, мкСм/см", font=("Arial Bold", 12), fg='blue')  
lbl.grid(column=0, row=22)  
num4 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num4.grid(row = 23, column = 0)
##lbl = Label(win, text="5 - Электропроводность, мкСм", font=("Arial Bold", 12))  
##lbl.grid(column=0, row=24)  
##num5 = tk.Entry(win, width=20, justify='center', font=("Arial Bold", 12))
##num5.grid(row = 25, column = 0)
##lbl = Label(win, text="6 - Индекс набухания, мл/2г", font=("Arial Bold", 12))  
##lbl.grid(column=0, row=26)  
##num5 = tk.Entry(win, width=20, justify='center', font=("Arial Bold", 12))
##num5.grid(row = 27, column = 0)

##lbl = Label(win, text="ВВЕСТИ УРОВЕНЬ СТАТИСТИЧЕСКОЙ ЗНАЧИМОСТИ (percent)", font=("Arial Bold", 10), fg='orange')
##lbl.grid(column=0, row=4)  
##num4 = tk.Entry(win, width=15, fg='orange', justify='center')
##num4.grid(row = 5, column = 0)
##lbl = Label(win, text="ВВЕСТИ ГИПОТЕЗУ О СРЕДНЕМ КВП ПЕРВОЙ СОВОКУПНОСТИ КЕРНОВ", font=("Arial Bold", 10), fg='blue')
##lbl.grid(column=0, row=6)  
##num5 = tk.Entry(win, width=20, fg='blue', justify='center')
##num5.grid(row = 7, column = 0)
lbl = Label(win, text="                                       6 - КОЕ, мг-экв/100г                                         ", font=("Arial Bold", 12), fg='blue')
lbl.grid(column=0, row=24)  
num10 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num10.grid(row = 25, column = 0)
lbl = Label(win, text="                                       7 - Монтмориллонит, %                                       ", font=("Arial Bold", 12), fg='blue')
lbl.grid(column=0, row=26)  
num11 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num11.grid(row = 27, column = 0)
lbl = Label(win, text="Целевой параметр", font=("Arial Bold", 12), fg='green')
lbl.grid(column=0, row=28)  
num111 = tk.Entry(win, width=20, fg='green', justify='center', font=("Arial Bold", 12))
num111.grid(row = 29, column = 0)
lbl = Label(win, text="Число кластеров", font=("Arial Bold", 10), fg='brown')
lbl.grid(column=9, row=17)  
num222 = tk.Entry(win, width=15, fg='brown', justify='center', font=("Arial Bold", 10))
num222.grid(row = 18, column = 9)
##lbl = Label(win, text="Координаты X - Y - (H)", font=("Arial Bold", 12), fg='brown')
##lbl.grid(column=0, row=33)  
##num333 = tk.Entry(win, width=30, fg='brown', justify='center', font=("Arial Bold", 11))
##num333.grid(row = 34, column = 0)
lbl = Label(win, text="Введите глубину Z, м", font=("Arial Bold", 12), fg='brown')
lbl.grid(column=9, row=27)  
num444 = tk.Entry(win, width=10, fg='brown', justify='center', font=("Arial Bold", 11))
num444.grid(row = 28, column = 9)
lbl = Label(win, text="Показывать линейную графику", font=("Arial Bold", 10), fg='black')
lbl.grid(column=9, row=8)  
num555 = tk.Entry(win, width=6, fg='black', justify='center', font=("Arial Bold", 10))
num555.grid(row = 9, column = 9)
##res10 = tk.Label(win, text = "-")
##res10.grid(row = 15, column = 0)
##res11 = tk.Label(win, text = "-")
##res11.grid(row = 16, column = 0)
##res12 = tk.Label(win, text = "-")
##res12.grid(row = 17, column = 0)
##
res0 = tk.Label(win, text = " ", fg='red', font = ("Arial Bold", 10))
res0.grid(row = 15, column = 9)
res1 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 10))
res1.grid(row = 13, column = 9)
res2 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 10))
res2.grid(row = 14, column = 9)
res3 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 12))
res3.grid(row = 22, column = 9)
res28 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 12))
res28.grid(row = 29, column = 9)
res26 = tk.Label(win, text = " ", fg='black', font = ("Arial Bold", 10))
res26.grid(row = 24, column = 9)
resf = tk.Label(win, text = " ", fg='red', font = ("Arial Bold", 10))
resf.grid(row = 26, column = 9)
res30 = tk.Label(win, text = " ", fg='black', font = ("Arial Bold", 10))
res30.grid(row = 33, column = 0)
res31 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 10))
res31.grid(row = 34, column = 0)
res32 = tk.Label(win, text = " ", fg='blue', font = ("Arial Bold", 12))
res32.grid(row = 34, column = 9)

####
####res18 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 12))
####res18.grid(row = 31, column = 0)
####res19 = tk.Label(win, text = " ", fg='red', font = ("Arial Bold", 10))
####res19.grid(row = 32, column = 0)
##
##res21 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 12))
##res21.grid(row = 14, column = 9)
##res22 = tk.Label(win, text = "-", fg='black', font = ("Arial Bold", 10))
##res22.grid(row = 32, column = 0)
##res23 = tk.Label(win, text = "-", fg='blue', font = ("Arial Bold", 10))
##res23.grid(row = 35, column = 0)
##res24 = tk.Label(win, text = "-", fg='blue', font = ("Arial Bold", 10))
##res24.grid(row = 36, column = 0)
##res25 = tk.Label(win, text = "-", fg='blue', font = ("Arial Bold", 10))
##res25.grid(row = 37, column = 0)
##res26 = tk.Label(win, text = "-", fg='red', font = ("Arial Bold", 9))
##res26.grid(row = 35, column = 3)
##res27 = tk.Label(win, text = "-", fg='red', font = ("Arial Bold", 9))
##res27.grid(row = 37, column = 3)

frame1 = tk.LabelFrame(win, text="Excel Data")
frame1.place(height=200, width=440)

# The file/file path text
label_file = tk.Label(win, text="No File Selected")
label_file.place(rely=0, relx=0)


#Treeview Widget
tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).

treescrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview) # command means update the yaxis view of the widget
treescrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview) # command means update the xaxis view of the widget
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget
treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget

def File_dialog():
    """This Function will open the file explorer and assign the chosen file path to label_file"""
    filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetype=(("xlsx files", "*.xlsx"),("All Files", "*.*")))
    label_file["text"] = filename
    return None

def Load_excel_data():
    
    global aaa
    global bbb
    global df
    for i in tv1.get_children():
        tv1.delete(i)     # clearing screen (table tv1)
    """If the file selected is valid this will load the file into the Treeview"""
    file_path = label_file["text"]
    try:
        excel_filename = r"{}".format(file_path)
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)
    

    except ValueError:
        tk.messagebox.showerror("Information", "The file you have chosen is invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"No such file as {file_path}")
        return None
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) # let the column heading = column name

    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tv1.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert   

    
def df():
    res3.configure(text = "  ")
    global df
    def sd(x):
        x1 = []
        for i in x:
            i = str(i)
            x1.append(i.replace(",", "."))
        return x1
    t = list(df.columns)
    
    a = list(df[t[0]])
    a = sd(a)
    a = [float(item) for item in a]
    #a = [x for x in a if str(x) != 'nan']
    b = list(df[t[1]])
    b = sd(b)
    b = [float(item) for item in b]
    c = list(df[t[2]])
    c = sd(c)
    c = [float(item) for item in c]
    d = list(df[t[3]])
    d = sd(d)
    d = [float(item) for item in d]
    e = list(df[t[4]])
    e = sd(e)
    e = [float(item) for item in e]
    f = list(df[t[5]])
    f = sd(f)
    f = [float(item) for item in f]
    g = list(df[t[6]])
    g = sd(g)
    g = [float(item) for item in g]

    df_r = df.copy()

    df = pd.DataFrame({t[0]:a, t[1]:b, t[2]:c, t[3]:d, t[4]:e, t[5]:f, t[6]:g})
    t = list(df.columns)
        

##        one = list(num1.get().split())
##        aa = [float(item) for item in one]
##        two = list(num2.get().split())
##        bb = [float(item) for item in two]
##        three = list(num3.get().split())
##        cc = [float(item) for item in three]
##        four = list(num4.get().split())
##        dd = [float(item) for item in four]
##        five = list(num10.get().split())
##        ee = [float(item) for item in five]
##        six = list(num11.get().split())
##        ff = [float(item) for item in six]

        
    dff = df.copy()
    if list(num111.get().split()):
        seven = list(num111.get().split())
        m = [float(item) for item in seven]
        for i in range(1,7):
            if i == m[0]:
                col = dff.pop(t[i-1])
                dff.insert((len(t)-1), col.name, col)
    
    t = list(dff.columns)
    dfv = dff.query('Электропроводность > 0')
    dfv['Глубина'].corr(dfv['Электропроводность'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Глубина']
    values_y = dfv['Электропроводность']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Электропроводность'].isna(), 'Электропроводность'] = dff.loc[dff['Электропроводность'].isna(), 'Глубина']*args[0]+args[1]
    
    dfv = dff.query('Монтмориллонит > 0')
    dfv['Глубина'].corr(dfv['Монтмориллонит'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Глубина']
    values_y = dfv['Монтмориллонит']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Монтмориллонит'].isna(), 'Монтмориллонит'] = dff.loc[dff['Монтмориллонит'].isna(), 'Глубина']*args[0]+args[1]

    dfv = dff.query('КОЕ > 0')
    dfv['Монтмориллонит'].corr(dfv['КОЕ'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['КОЕ']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['КОЕ'].isna(), 'КОЕ'] = dff.loc[dff['КОЕ'].isna(), 'Монтмориллонит']*args[0]+args[1]

    dfv = dff.query('Влажность > 0')
    dfv['Монтмориллонит'].corr(dfv['Влажность'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Влажность']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Влажность'].isna(), 'Влажность'] = dff.loc[dff['Влажность'].isna(), 'Монтмориллонит']*args[0]+args[1]

    dfv = dff.query('Песок > 0')
    dfv['Монтмориллонит'].corr(dfv['Песок'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Песок']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Песок'].isna(), 'Песок'] = dff.loc[dff['Песок'].isna(), 'Монтмориллонит']*args[0]+args[1]
    
    dfv = dff.query('Индекс > 0')
    dfv['Монтмориллонит'].corr(dfv['Индекс'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Индекс']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Индекс'].isna(), 'Индекс'] = dff.loc[dff['Индекс'].isna(), 'Монтмориллонит']*args[0]+args[1]

    y = dff[t[-1]]
    X = dff.drop([t[-1]], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_st = scaler.fit_transform(X_train)
    X_test_st = scaler.transform(X_test)

    models = [
    [Lasso(), 'Линейная регрессия Lasso'],
    [Ridge(), 'Линейная регрессия Ridge'],
    [RandomForestRegressor(n_estimators = 200, random_state = 0), 'Случайный лес'],
    [GradientBoostingRegressor(n_estimators = 200, random_state = 0), 'Градиентный бустинг'],
    [DecisionTreeRegressor(random_state = 0), 'Дерево решений']
    ]

    def metrics(y_true, y_pred, title):
        print(title)
        print('MAE: {:.2f}'.format(mean_absolute_error(y_true,y_pred)))
        print('MSE: {:.2f}'.format(mean_squared_error(y_true,y_pred)))
        print('R2: {:.2f}'.format(r2_score(y_true,y_pred)))

    def prediction(mod, X_train, y_train, X_test, y_test, name):
        model = mod
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics(y_test, y_pred, name)

    c = dff.corr()
    plt.figure(figsize = (8,8))
    sns.heatmap(c, annot = True, square = True)
    plt.title('Матрица корреляции признаков');
    plt.show()

    for i in models:
        prediction(i[0],X_train_st,y_train,X_test_st,y_test,i[1])

    p = []
    w = []
    for i in models:
        model = i[0]
        model.fit(X_train_st, y_train)
        y_pred = model.predict(X_test_st)
        p.append(r2_score(y_test,y_pred))
        w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)
    if max(p) == p[0]:
        res1.configure(text = "МAX R2-CORE - LASSO: %s" % round(p[0],2))
        #print('Максимальная метрика R2-CORE у регрессии LASSO: ', p[0])
    elif max(p) == p[1]:
        res1.configure(text = "МAX R2-CORE - RIDGE: %s" % round(p[1],2))
        #print('Максимальная метрика R2-CORE у регрессии Ridge: ', p[1])
    elif max(p) == p[2]:
        res1.configure(text = "МAX R2-CORE - RandomForest: %s" % round(p[2],2))
        #print('Максимальная метрика R2-CORE у регрессии RandomForest: ', p[2])
    elif max(p) == p[3]:
        res1.configure(text = "МAX R2-CORE - GradientBoosting: %s" % round(p[3],2))
        #print('Максимальная метрика R2-CORE у регрессии GradientBoosting: ', p[3])  
    elif max(p) == p[4]:
        res1.configure(text = "МAX R2-CORE - TreeDecision: %s" % round(p[4],2))
        #print('Максимальная метрика R2-CORE у регрессии TreeDecision: ', p[4])      
    if min(w) == w[0]:
        res2.configure(text = "МIN mean(MSE + MAE) - LASSO: %s" % round(w[0],2)) 
        #print('Минимальная средняя метрика у регрессии LASSO: ', w[0])
    elif min(w) == w[1]:
        res2.configure(text = "МIN mean(MSE + MAE) - RIDGE: %s" % round(w[1],2))
        #print('Минимальная средняя метрика у регрессии Ridge: ', w[1])
    elif min(w) == w[2]:
        res2.configure(text = "МIN mean(MSE + MAE) - RandomForest: %s" % round(w[2],2))
        #print('Минимальная средняя метрика у регрессии RandomForest: ', w[2])
    elif min(w) == w[3]:
        res2.configure(text = "МIN mean(MSE + MAE) - GradientBoosting: %s" % round(w[3],2))
        #print('Минимальная средняя метрика у : регрессии GradientBoosting', w[3])
    elif min(w) == w[4]:
        res2.configure(text = "МIN mean(MSE + MAE) - TreeDecision: %s" % round(w[4],2))
        #print('Минимальная средняя метрика у : регрессии TreeDecision', w[4])   
  

    def graph(model, f):
        u = t.copy()
        dft = dff.copy()
        dft = dft.drop(t[-1], axis = 1)
        for i in dft.columns:
            dft[i] = round(dft[i].mean(),2)
        del u[-1]
        for i in u:
            dfr = dft.copy()
            for j in range(len(dfr[i])):
                mi = dff[i].min()
                ma = dff[i].max()
                n = (ma-mi)/(len(dft[i])-1)
                dfr[i][j] = mi + j*n
            X_st = scaler.transform(dfr)
            y_pred = model.predict(X_st)
            dfr[t[-1]] = y_pred
            dg = dfr.groupby(i).agg({t[-1]:'mean'})
    
            def mop(values_x,a,b,c,d):
                return a * values_x**3 + b * values_x**2 + c * values_x + d
            def mop1(values_x,a,b,c):
                return a * values_x**2 + b * values_x + c 
            def mop2(values_x,a,b):
                return a * values_x + b 
            values_x = dfr[i]
            values_y = dfr[t[-1]]
        
            args, covar = curve_fit(mop, values_x, values_y)
            y_pred_1 = mop(values_x, *args)
            r2_1 = r2_score(values_y, y_pred_1)
            args1, covar1 = curve_fit(mop1, values_x, values_y)
            y_pred_2 = mop1(values_x, *args1)
            r2_2 = r2_score(values_y, y_pred_2)
            args2, covar2 = curve_fit(mop2, values_x, values_y)
            y_pred_3 = mop2(values_x, *args2)
            r2_3 = r2_score(values_y, y_pred_3)
            r2 = [r2_1,r2_2,r2_3]
            y = []
            if max(r2) == r2_1:
                for j in range(len(dfr[i])):
                    y.append(args[0] * dfr[i][j]**3 + args[1] * dfr[i][j]**2 + args[2] * dfr[i][j] + args[3])
            elif max(r2) == r2_2:
                for j in range(len(dfr[i])):
                    y.append(args1[0] * dfr[i][j]**2 + args1[1] * dfr[i][j] + args1[2])    
            elif max(r2) == r2_3:
                for j in range(len(dfr[i])):
                    y.append(args2[0] * dfr[i][j] + args2[1]) 
            dgr = pd.DataFrame({'x':list(dfr[i]), 'y':y})
        
            fig, axes = plt.subplots(1, 1, figsize=(8, 5))
            sns.set_style('whitegrid')
            sns.set_palette('bright')
        
            sns.lineplot(dgr.pivot_table(index = 'x', values = 'y', aggfunc = 'mean'), color = 'red')
        
            sns.scatterplot(data=dg)
            #sns.set(rc={'figure.figsize':(8,5)})
            axes.set(xlabel= 'Переменный параметр <{}>'.format(i),
            ylabel='Функция отклика <{}>'.format(t[-1]),
            title ='Усредненная параметрическая диаграмма при переменном факторе {} ({})'.format(i, f))
            plt.legend(title = '{}'.format(dft.iloc[[0]]), loc=2, bbox_to_anchor=(1, 1), fontsize = 10)
            plt.xticks(rotation = 10)    
            plt.show()

    if max(p) == p[0]:
        graph(models[0][0], models[0][1])
    elif max(p) == p[1]:
        graph(models[1][0], models[1][1])
    elif max(p) == p[2]:
        graph(models[2][0], models[2][1])
    elif max(p) == p[3]:
        graph(models[3][0], models[3][1])
    elif max(p) == p[4]:
        graph(models[4][0], models[4][1])

    df = df_r
       
def dff():
    res1.configure(text = "  ")
    res2.configure(text = "  ")
    global df
    def sd(x):
        x1 = []
        for i in x:
            i = str(i)
            x1.append(i.replace(",", "."))
        return x1
    t = list(df.columns)
    
    a = list(df[t[0]])
    a = sd(a)
    a = [float(item) for item in a]
    #a = [x for x in a if str(x) != 'nan']
    b = list(df[t[1]])
    b = sd(b)
    b = [float(item) for item in b]
    c = list(df[t[2]])
    c = sd(c)
    c = [float(item) for item in c]
    d = list(df[t[3]])
    d = sd(d)
    d = [float(item) for item in d]
    e = list(df[t[4]])
    e = sd(e)
    e = [float(item) for item in e]
    f = list(df[t[5]])
    f = sd(f)
    f = [float(item) for item in f]
    g = list(df[t[6]])
    g = sd(g)
    g = [float(item) for item in g]

    df_u = df.copy()

    df = pd.DataFrame({t[0]:a, t[1]:b, t[2]:c, t[3]:d, t[4]:e, t[5]:f, t[6]:g})
    t = list(df.columns)  

    one = list(num1.get().split())
    aa = [float(item) for item in one]
    two = list(num2.get().split())
    bb = [float(item) for item in two]
    three = list(num3.get().split())
    cc = [float(item) for item in three]
    four = list(num4.get().split())
    dd = [float(item) for item in four]
    five = list(num10.get().split())
    ee = [float(item) for item in five]
    six = list(num11.get().split())
    ff = [float(item) for item in six]
    zer = list(num0.get().split())
    zz = [float(item) for item in zer]

    dfx = pd.DataFrame({t[0]:zz, t[1]:aa, t[2]:bb, t[3]:cc, t[4]:dd, t[5]:ee, t[6]:ff})
    dx = dfx.copy()
    dff = df.copy()
    
    if list(num111.get().split()):
        seven = list(num111.get().split())
        m = [float(item) for item in seven]
        for i in range(1,8):
            if i == m[0]:
                col = dff.pop(t[i-1])
                dff.insert((len(t)-1), col.name, col)

                col2 = dx.pop(t[i-1])
                dx.insert((len(t)-1), col2.name, col2)
    print(dff)
    print(dx)
##    dff.rename(columns = {'Электропроводность, мкСм/см':'Электропроводность', 'Индекс набух., мл/2г':'Индекс'}, inplace = True )
##    dx.rename(columns = {'Электропроводность, мкСм/см':'Электропроводность', 'Индекс набух., мл/2г':'Индекс'}, inplace = True )
    t = list(dff.columns)
    dfv = dff.query('Электропроводность > 0')
    dfv['Глубина'].corr(dfv['Электропроводность'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Глубина']
    values_y = dfv['Электропроводность']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Электропроводность'].isna(), 'Электропроводность'] = dff.loc[dff['Электропроводность'].isna(), 'Глубина']*args[0]+args[1]

    dfv = dff.query('Монтмориллонит > 0')
    dfv['Глубина'].corr(dfv['Монтмориллонит'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Глубина']
    values_y = dfv['Монтмориллонит']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Монтмориллонит'].isna(), 'Монтмориллонит'] = dff.loc[dff['Монтмориллонит'].isna(), 'Глубина']*args[0]+args[1]

    dfv = dff.query('КОЕ > 0')
    dfv['Монтмориллонит'].corr(dfv['КОЕ'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['КОЕ']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['КОЕ'].isna(), 'КОЕ'] = dff.loc[dff['КОЕ'].isna(), 'Монтмориллонит']*args[0]+args[1]

    dfv = dff.query('Влажность > 0')
    dfv['Монтмориллонит'].corr(dfv['Влажность'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Влажность']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Влажность'].isna(), 'Влажность'] = dff.loc[dff['Влажность'].isna(), 'Монтмориллонит']*args[0]+args[1]

    dfv = dff.query('Песок > 0')
    dfv['Монтмориллонит'].corr(dfv['Песок'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Песок']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Песок'].isna(), 'Песок'] = dff.loc[dff['Песок'].isna(), 'Монтмориллонит']*args[0]+args[1]
    
    dfv = dff.query('Индекс > 0')
    dfv['Монтмориллонит'].corr(dfv['Индекс'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Индекс']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Индекс'].isna(), 'Индекс'] = dff.loc[dff['Индекс'].isna(), 'Монтмориллонит']*args[0]+args[1]

    y = dff[t[-1]]
    X = dff.drop([t[-1]], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_st = scaler.fit_transform(X_train)
    X_test_st = scaler.transform(X_test)

    models = [
    [Lasso(), 'Линейная регрессия Lasso'],
    [Ridge(), 'Линейная регрессия Ridge'],
    [RandomForestRegressor(n_estimators = 200, random_state = 0), 'Случайный лес'],
    [GradientBoostingRegressor(n_estimators = 200, random_state = 0), 'Градиентный бустинг'],
    [DecisionTreeRegressor(random_state = 0), 'Дерево решений']
    ]

    def metrics(y_true, y_pred, title):
        print(title)
        print('MAE: {:.2f}'.format(mean_absolute_error(y_true,y_pred)))
        print('MSE: {:.2f}'.format(mean_squared_error(y_true,y_pred)))
        print('R2: {:.2f}'.format(r2_score(y_true,y_pred)))

    def prediction(mod, X_train, y_train, X_test, y_test, name):
        model = mod
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics(y_test, y_pred, name)

##    c = dff.corr()
##    plt.figure(figsize = (8,8))
##    sns.heatmap(c, annot = True, square = True)
##    plt.title('Матрица корреляции признаков');
##    plt.show()

    for i in models:
        prediction(i[0],X_train_st,y_train,X_test_st,y_test,i[1])

    p = []
    w = []
    for i in models:
        model = i[0]
        model.fit(X_train_st, y_train)
        y_pred = model.predict(X_test_st)
        p.append(r2_score(y_test,y_pred))
        w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)
    if max(p) == p[0]:
        res1.configure(text = "МAX R2-CORE - LASSO: %s" % round(p[0],2))
        #print('Максимальная метрика R2-CORE у регрессии LASSO: ', p[0])
    elif max(p) == p[1]:
        res1.configure(text = "МAX R2-CORE - RIDGE: %s" % round(p[1],2))
        #print('Максимальная метрика R2-CORE у регрессии Ridge: ', p[1])
    elif max(p) == p[2]:
        res1.configure(text = "МAX R2-CORE - RandomForest: %s" % round(p[2],2))
        #print('Максимальная метрика R2-CORE у регрессии RandomForest: ', p[2])
    elif max(p) == p[3]:
        res1.configure(text = "МAX R2-CORE - GradientBoosting: %s" % round(p[3],2))
        #print('Максимальная метрика R2-CORE у регрессии GradientBoosting: ', p[3])  
    elif max(p) == p[4]:
        res1.configure(text = "МAX R2-CORE - TreeDecision: %s" % round(p[4],2))
        #print('Максимальная метрика R2-CORE у регрессии TreeDecision: ', p[4])      
    if min(w) == w[0]:
        res2.configure(text = "МIN mean(MSE + MAE) - LASSO: %s" % round(w[0],2)) 
        #print('Минимальная средняя метрика у регрессии LASSO: ', w[0])
    elif min(w) == w[1]:
        res2.configure(text = "МIN mean(MSE + MAE) - RIDGE: %s" % round(w[1],2))
        #print('Минимальная средняя метрика у регрессии Ridge: ', w[1])
    elif min(w) == w[2]:
        res2.configure(text = "МIN mean(MSE + MAE) - RandomForest: %s" % round(w[2],2))
        #print('Минимальная средняя метрика у регрессии RandomForest: ', w[2])
    elif min(w) == w[3]:
        res2.configure(text = "МIN mean(MSE + MAE) - GradientBoosting: %s" % round(w[3],2))
        #print('Минимальная средняя метрика у : регрессии GradientBoosting', w[3])
    elif min(w) == w[4]:
        res2.configure(text = "МIN mean(MSE + MAE) - TreeDecision: %s" % round(w[4],2))
        #print('Минимальная средняя метрика у : регрессии TreeDecision', w[4])   

    Xn = dx.drop([t[-1]], axis = 1)
    Xn_st = scaler.transform(Xn)

    if max(p) == p[0]:
        y_pred = models[0][0].predict(Xn_st)
        print('Регрессия произведена с использованием модели', models[0][1], ', имеющей максимальную метрику R2-SCORE ', round(p[0],2))
##        Xn[t[-1]] = y_pred  
##        print(Xn)
        res3.configure(text = "Расчетное значение параметра %s равно: %s" % (t[-1], round(y_pred[0],2)))
    elif max(p) == p[1]:
        y_pred = models[1][0].predict(Xn_st)
        print('Регрессия произведена с использованием модели', models[1][1], ', имеющей максимальную метрику R2-SCORE ', round(p[1],2))
##        Xn[t[-1]] = y_pred  
##        print(Xn)
        res3.configure(text = "Расчетное значение параметра %s равно: %s" % (t[-1], round(y_pred[0],2)))
    elif max(p) == p[2]:
        y_pred = models[2][0].predict(Xn_st)
        print('Регрессия произведена с использованием модели', models[2][1], ', имеющей максимальную метрику R2-SCORE ', round(p[2],2))
##        Xn[t[-1]] = y_pred  
##        print(Xn)
        res3.configure(text = "Расчетное значение параметра %s равно: %s" % (t[-1], round(y_pred[0],2)))
    elif max(p) == p[3]:
        y_pred = models[3][0].predict(Xn_st)
        print('Регрессия произведена с использованием модели', models[3][1], ', имеющей максимальную метрику R2-SCORE ', round(p[3],2))
##        Xn[t[-1]] = y_pred  
##        print(Xn)
        res3.configure(text = "Расчетное значение параметра %s равно: %s" % (t[-1], round(y_pred[0],2)))
    elif max(p) == p[4]:
        y_pred = models[4][0].predict(Xn_st)
        print('Регрессия произведена с использованием модели', models[4][1], ', имеющей максимальную метрику R2-SCORE ', round(p[3],2))
##        Xn[t[-1]] = y_pred  
##        print(Xn)
        res3.configure(text = "Расчетное значение параметра %s равно: %s" % (t[-1], round(y_pred[0],2)))

    df = df_u

def dfff():
    res1.configure(text = "  ")
    res2.configure(text = "  ")
    global df
    def sd(x):
        x1 = []
        for i in x:
            i = str(i)
            x1.append(i.replace(",", "."))
        return x1
    t = list(df.columns)
    
    a = list(df[t[0]])
    a = sd(a)
    a = [float(item) for item in a]
    #a = [x for x in a if str(x) != 'nan']
    b = list(df[t[1]])
    b = sd(b)
    b = [float(item) for item in b]
    c = list(df[t[2]])
    c = sd(c)
    c = [float(item) for item in c]
    d = list(df[t[3]])
    d = sd(d)
    d = [float(item) for item in d]
    e = list(df[t[4]])
    e = sd(e)
    e = [float(item) for item in e]
    f = list(df[t[5]])
    f = sd(f)
    f = [float(item) for item in f]
    g = list(df[t[6]])
    g = sd(g)
    g = [float(item) for item in g]

    df_r = df.copy()

    df = pd.DataFrame({t[0]:a, t[1]:b, t[2]:c, t[3]:d, t[4]:e, t[5]:f, t[6]:g})
    t = list(df.columns)  

    one = list(num1.get().split())
    aa = [float(item) for item in one]
    two = list(num2.get().split())
    bb = [float(item) for item in two]
    three = list(num3.get().split())
    cc = [float(item) for item in three]
    four = list(num4.get().split())
    dd = [float(item) for item in four]
    five = list(num10.get().split())
    ee = [float(item) for item in five]
    six = list(num11.get().split())
    ff = [float(item) for item in six]
    zer = list(num0.get().split())
    zz = [float(item) for item in zer]

    dfx = pd.DataFrame({t[0]:zz, t[1]:aa, t[2]:bb, t[3]:cc, t[4]:dd, t[5]:ee, t[6]:ff})
    dx = dfx.copy()
    dff = df.copy()
    
##    if list(num111.get().split()):
##        seven = list(num111.get().split())
##        m = [float(item) for item in seven]
##        for i in range(1,7):
##            if i == m[0]:
##                col = dff.pop(t[i-1])
##                dff.insert((len(t)-1), col.name, col)
##
##                col2 = dx.pop(t[i-1])
##                dx.insert((len(t)-1), col2.name, col2)
    clast = list(num222.get().split())
    clast = [int(item) for item in clast]

    t = list(dff.columns)
    dfv = dff.query('Электропроводность > 0')
    dfv['Глубина'].corr(dfv['Электропроводность'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Глубина']
    values_y = dfv['Электропроводность']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Электропроводность'].isna(), 'Электропроводность'] = dff.loc[dff['Электропроводность'].isna(), 'Глубина']*args[0]+args[1]
    
    dfv = dff.query('Монтмориллонит > 0')
    dfv['Глубина'].corr(dfv['Монтмориллонит'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Глубина']
    values_y = dfv['Монтмориллонит']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Монтмориллонит'].isna(), 'Монтмориллонит'] = dff.loc[dff['Монтмориллонит'].isna(), 'Глубина']*args[0]+args[1]

    dfv = dff.query('КОЕ > 0')
    dfv['Монтмориллонит'].corr(dfv['КОЕ'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['КОЕ']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['КОЕ'].isna(), 'КОЕ'] = dff.loc[dff['КОЕ'].isna(), 'Монтмориллонит']*args[0]+args[1]

    dfv = dff.query('Влажность > 0')
    dfv['Монтмориллонит'].corr(dfv['Влажность'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Влажность']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Влажность'].isna(), 'Влажность'] = dff.loc[dff['Влажность'].isna(), 'Монтмориллонит']*args[0]+args[1]

    dfv = dff.query('Песок > 0')
    dfv['Монтмориллонит'].corr(dfv['Песок'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Песок']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Песок'].isna(), 'Песок'] = dff.loc[dff['Песок'].isna(), 'Монтмориллонит']*args[0]+args[1]
    
    dfv = dff.query('Индекс > 0')
    dfv['Монтмориллонит'].corr(dfv['Индекс'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Индекс']
    args, covar = curve_fit(mop, values_x, values_y)
    dff.loc[dff['Индекс'].isna(), 'Индекс'] = dff.loc[dff['Индекс'].isna(), 'Монтмориллонит']*args[0]+args[1]
    
    X = dff
    scaler = StandardScaler()
    X_st = scaler.fit_transform(X)
    mod_clusters = KMeans(n_clusters=clast[0], random_state=0)
    m_clusters = mod_clusters.fit_predict(X_st)
    X['cluster'] = m_clusters
    
    c = list(df.columns)
    X_count = X.groupby('cluster', as_index = False).agg({c[0]:'count'})
    plt.figure(figsize=(7, 5))
    sns.set_style('whitegrid')
    color = ["green", "Red", "Yellow"] 
    sns.set_palette(color) 
    sns.barplot(x = 'cluster', y = c[0], hue = 'cluster', data = X_count, palette="deep")
    plt.title('Количество экземпляров в сформированных кластерах')
    plt.ylabel('Число точек')
    plt.xlabel('Кластеры')
    plt.show()
    X_mean = X.groupby('cluster', as_index = False).agg('mean')
    for i in X_mean.drop('cluster', axis = 1).columns:
        plt.figure(figsize=(7, 5))
        sns.set_style('whitegrid')
        color = ["green", "Red", "Yellow"] 
        sns.set_palette(color)        
        sns.barplot(x = 'cluster', y = i, hue = 'cluster', data = X_mean, palette="deep")
        plt.title('Распредедение средних значений признака "{}" в сформированных кластерах'.format(i))
        plt.show()
 
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
##    color = ["green", "Red", "Yellow"] 
    sns.set_palette(color)
    X_m = X.groupby(['Глубина', 'cluster'], as_index = False).agg({'Индекс':'mean'})
    def d(r):
        if r['cluster'] == 0:
            return r['Индекс']
        else:
            return 0
    def d1(r):
        if r['cluster'] == 1:
            return r['Индекс']
        else:
            return 0
    def d2(r):
        if r['cluster'] == 2:
            return r['Индекс']
        else:
            return 0
    def d3(r):
        if r['cluster'] == 3:
            return r['Индекс']
        else:
            return 0
    def d4(r):
        if r['cluster'] == 4:
            return r['Индекс']
        else:
            return 0
    def d5(r):
        if r['cluster'] == 5:
            return r['Индекс']
        else:
            return 0
    def d6(r):
        if r['cluster'] == 6:
            return r['Индекс']
        else:
            return 0
    def d7(r):
        if r['cluster'] == 7:
            return r['Индекс']
        else:
            return 0
    def d8(r):
        if r['cluster'] == 8:
            return r['Индекс']
        else:
            return 0
    def d9(r):
        if r['cluster'] == 9:
            return r['Индекс']
        else:
            return 0
    
    X_m['ind_0'] = X_m.apply(d, axis = 1)
    X_m['ind_1'] = X_m.apply(d1, axis = 1)
    X_m['ind_2'] = X_m.apply(d2, axis = 1)
    X_m['ind_3'] = X_m.apply(d3, axis = 1)
    X_m['ind_4'] = X_m.apply(d4, axis = 1)
    X_m['ind_5'] = X_m.apply(d5, axis = 1)
    X_m['ind_6'] = X_m.apply(d6, axis = 1)
    X_m['ind_7'] = X_m.apply(d7, axis = 1)
    X_m['ind_8'] = X_m.apply(d8, axis = 1)
    X_m['ind_9'] = X_m.apply(d9, axis = 1)
    
    #sns.lineplot(y = 'Глубина', x = 'Индекс', hue = 'cluster', data = X_m, palette="deep", orient='y')
    if len(X['cluster'].unique()) == 1:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], alpha=0.5)
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.legend(['0'], title = 'кластеры')
        plt.title('Изменение среднего индекса набухания в различных кластерах с глубиной')
        plt.ylabel('Средний индекс набухания, мл/2г')
        plt.xlabel('Глубина, м')
        plt.show()       
    elif len(X['cluster'].unique()) == 2:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], alpha=0.5)
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.legend(['0', '1'], title = 'кластеры')
        plt.title('Изменение среднего индекса набухания в различных кластерах с глубиной')
        plt.ylabel('Средний индекс набухания, мл/2г')
        plt.xlabel('Глубина, м')
        plt.show()   
    elif len(X['cluster'].unique()) == 3:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], alpha=0.5)
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.legend(['0', '1', '2'], title = 'кластеры')
        plt.title('Изменение среднего индекса набухания в различных кластерах с глубиной')
        plt.ylabel('Средний индекс набухания, мл/2г')
        plt.xlabel('Глубина, м')
        plt.show()   

    elif len(X['cluster'].unique()) == 4:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], X_m['ind_3'], alpha=0.5, colors =['r', 'c', 'g', 'y'])
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_3"], alpha=0.3)
        plt.legend(['0', '1', '2', '3'], title = 'кластеры')
        plt.title('Изменение среднего индекса набухания в различных кластерах с глубиной')
        plt.ylabel('Средний индекс набухания, мл/2г')
        plt.xlabel('Глубина, м')
        plt.show()
    elif len(X['cluster'].unique()) == 5:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], X_m['ind_3'], X_m['ind_4'], alpha=0.5, colors =['r', 'c', 'g', 'y', 'k'])
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_3"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_4"], alpha=0.3)
        plt.legend(['0', '1', '2', '3', '4'], title = 'кластеры')
        plt.title('Изменение среднего индекса набухания в различных кластерах с глубиной')
        plt.ylabel('Средний индекс набухания, мл/2г')
        plt.xlabel('Глубина, м')
        plt.show()
    elif len(X['cluster'].unique()) == 6:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], X_m['ind_3'], X_m['ind_4'], X_m['ind_5'], alpha=0.5, colors =['r', 'c', 'g', 'y', 'k', 'm']) 
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_3"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_4"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_5"], alpha=0.3)
        plt.legend(['0', '1', '2', '3', '4', '5'], title = 'кластеры')
        plt.title('Изменение среднего индекса набухания в различных кластерах с глубиной')
        plt.ylabel('Средний индекс набухания, мл/2г')
        plt.xlabel('Глубина, м')
        plt.show()
    elif len(X['cluster'].unique()) == 7:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], X_m['ind_3'], X_m['ind_4'], X_m['ind_5'], X_m['ind_6'], alpha=0.5, colors =['r', 'c', 'g', 'y', 'k', 'm', 'blue'])
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_3"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_4"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_5"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_6"], alpha=0.3)
        plt.legend(['0', '1', '2', '3', '4', '5', '6'], title = 'кластеры')
        plt.title('Изменение среднего индекса набухания в различных кластерах с глубиной')
        plt.ylabel('Средний индекс набухания, мл/2г')
        plt.xlabel('Глубина, м')
        plt.show()
    
    
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
##    color = ["green", "Red", "Yellow"] 
    sns.set_palette(color)
    X_m = X.groupby(['Глубина', 'cluster'], as_index = False).agg({'Монтмориллонит':'mean'})
    def d(r):
        if r['cluster'] == 0:
            return r['Монтмориллонит']
        else:
            return 0
    def d1(r):
        if r['cluster'] == 1:
            return r['Монтмориллонит']
        else:
            return 0
    def d2(r):
        if r['cluster'] == 2:
            return r['Монтмориллонит']
        else:
            return 0
    def d3(r):
        if r['cluster'] == 3:
            return r['Монтмориллонит']
        else:
            return 0
    def d4(r):
        if r['cluster'] == 4:
            return r['Монтмориллонит']
        else:
            return 0
    def d5(r):
        if r['cluster'] == 5:
            return r['Монтмориллонит']
        else:
            return 0
    def d6(r):
        if r['cluster'] == 6:
            return r['Монтмориллонит']
        else:
            return 0
    def d7(r):
        if r['cluster'] == 7:
            return r['Монтмориллонит']
        else:
            return 0
    def d8(r):
        if r['cluster'] == 8:
            return r['Монтмориллонит']
        else:
            return 0
    
    
    X_m['ind_0'] = X_m.apply(d, axis = 1)
    X_m['ind_1'] = X_m.apply(d1, axis = 1)
    X_m['ind_2'] = X_m.apply(d2, axis = 1)
    X_m['ind_3'] = X_m.apply(d3, axis = 1)
    X_m['ind_4'] = X_m.apply(d4, axis = 1)
    X_m['ind_5'] = X_m.apply(d5, axis = 1)
    X_m['ind_6'] = X_m.apply(d6, axis = 1)
    X_m['ind_7'] = X_m.apply(d7, axis = 1)
    X_m['ind_8'] = X_m.apply(d8, axis = 1)
    
    if len(X['cluster'].unique()) == 1:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], alpha=0.5)
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.legend(['0'], title = 'кластеры')
        plt.title('Изменение средней концентрации монтмориллонита в различных кластерах с глубиной')
        plt.ylabel('Средняя концентрация монтмориллонита, %')
        plt.xlabel('Глубина, м')
        plt.show()       
    elif len(X['cluster'].unique()) == 2:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], alpha=0.5)
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.legend(['0', '1'], title = 'кластеры')
        plt.title('Изменение средней концентрации монтмориллонита в различных кластерах с глубиной')
        plt.ylabel('Средняя концентрация монтмориллонита, %')
        plt.xlabel('Глубина, м')
        plt.show()   
    elif len(X['cluster'].unique()) == 3:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], alpha=0.5)
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.legend(['0', '1', '2'], title = 'кластеры')
        plt.title('Изменение средней концентрации монтмориллонита в различных кластерах с глубиной')
        plt.ylabel('Средняя концентрация монтмориллонита, %')
        plt.xlabel('Глубина, м')
        plt.show()   

    elif len(X['cluster'].unique()) == 4:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], X_m['ind_3'], alpha=0.5, colors =['r', 'c', 'g', 'y'])
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_3"], alpha=0.3)
        plt.legend(['0', '1', '2', '3'], title = 'кластеры')
        plt.title('Изменение средней концентрации монтмориллонита в различных кластерах с глубиной')
        plt.ylabel('Средняя концентрация монтмориллонита, %')
        plt.xlabel('Глубина, м')
        plt.show()
    elif len(X['cluster'].unique()) == 5:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], X_m['ind_3'], X_m['ind_4'], alpha=0.5, colors =['r', 'c', 'g', 'y', 'k'])
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_3"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_4"], alpha=0.3)
        plt.legend(['0', '1', '2', '3', '4'], title = 'кластеры')
        plt.title('Изменение средней концентрации монтмориллонита в различных кластерах с глубиной')
        plt.ylabel('Средняя концентрация монтмориллонита, %')
        plt.xlabel('Глубина, м')
        plt.show()
    elif len(X['cluster'].unique()) == 6:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], X_m['ind_3'], X_m['ind_4'], X_m['ind_5'], alpha=0.5, colors =['r', 'c', 'g', 'y', 'k', 'm'])
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_3"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_4"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_5"], alpha=0.3)
        plt.legend(['0', '1', '2', '3', '4', '5'], title = 'кластеры')
        plt.title('Изменение средней концентрации монтмориллонита в различных кластерах с глубиной')
        plt.ylabel('Средняя концентрация монтмориллонита, %')
        plt.xlabel('Глубина, м')
        plt.show()
    elif len(X['cluster'].unique()) == 7:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], X_m['ind_3'], X_m['ind_4'], X_m['ind_5'], X_m['ind_6'], alpha=0.5, colors =['r', 'c', 'g', 'y', 'k', 'm', 'blue'])
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_3"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_4"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_5"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_6"], alpha=0.3)
        plt.legend(['0', '1', '2', '3', '4', '5', '6'], title = 'кластеры')
        plt.title('Изменение средней концентрации монтмориллонита в различных кластерах с глубиной')
        plt.ylabel('Средняя концентрация монтмориллонита, %')
        plt.xlabel('Глубина, м')
        plt.show()
    elif len(X['cluster'].unique()) == 8:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], X_m['ind_3'], X_m['ind_4'], X_m['ind_5'], X_m['ind_6'], X_m['ind_7'], alpha=0.5)
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_3"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_4"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_5"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_6"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_7"], alpha=0.3)
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7'], title = 'кластеры')
        plt.title('Изменение средней концентрации монтмориллонита в различных кластерах с глубиной')
        plt.ylabel('Средняя концентрация монтмориллонита, %')
        plt.xlabel('Глубина, м')
        plt.show()
    elif len(X['cluster'].unique()) == 9:
        plt.stackplot(X_m["Глубина"], X_m['ind_0'], X_m['ind_1'], X_m['ind_2'], X_m['ind_3'], X_m['ind_4'], X_m['ind_5'], X_m['ind_6'], X_m['ind_7'], X_m['ind_8'], alpha=0.5, colors =['r', 'c', 'g', 'y', 'k', 'm', 'blue'])
        plt.fill_between(X_m["Глубина"],X_m["ind_0"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_1"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_2"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_3"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_4"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_5"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_6"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_7"], alpha=0.3)
        plt.fill_between(X_m["Глубина"],X_m["ind_8"], alpha=0.3)
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8'], title = 'кластеры')
        plt.title('Изменение средней концентрации монтмориллонита в различных кластерах с глубиной')
        plt.ylabel('Средняя концентрация монтмориллонита, %')
        plt.xlabel('Глубина, м')
        plt.show()

    df = df_r
    


def dffff():
    try:
        
        global df

        if list(num444.get().split()):
            at_ = [at[0], at[1], at[2], at[3]]
            bt_ = [bt[0], bt[1], bt[2], bt[3]]

            ab = at_+bt_
            x_ = []
            y_ = []
            for i in range(len(ab)):
                x_.append(5267188.06 + 1.41*ab[i][1])
                y_.append(14715755.24 + 1.426*ab[i][0])
            Xmin = min(x_) 
            Xmax = max(x_) 
            Ymin = min(y_) 
            Ymax = max(y_)

                    
            Xt = 5267188.06 + 1.41*bt[-1][1]
            Yt = 14715755.24 + 1.426*bt[-1][0]
            
            
            res30.configure(text = "Xmin-Xmax-Ymin-Ymax: %s - %s - %s - %s " % (round(Xmin,1), round(Xmax,1), round(Ymin,1), round(Ymax,1)))
            res31.configure(text = "Координаты точки X-Y: %s - %s" % (round(Xt,1), round(Yt,1)))   

            S = (Xmax - Xmin)*(Ymax - Ymin)

            print(Xmin)
            print(Xmax)
        
        
        def sd(x):
            x1 = []
            for i in x:
                i = str(i)
                x1.append(i.replace(",", "."))
            return x1
        t = list(df.columns)
        a = list(df[t[0]])
        a = sd(a)
        a = [float(item) for item in a]
        #a = [x for x in a if str(x) != 'nan']
        b = list(df[t[1]])
        b = sd(b)
        b = [float(item) for item in b]
        c = list(df[t[2]])
        c = sd(c)
        c = [float(item) for item in c]
        d = list(df[t[3]])
        d = sd(d)
        d = [float(item) for item in d]
        e = list(df[t[4]])
        e = sd(e)
        e = [float(item) for item in e]
        f = list(df[t[5]])
        f = sd(f)
        f = [float(item) for item in f]
        g = list(df[t[6]])
        g = sd(g)
        g = [float(item) for item in g]
        xx = list(df[t[7]])
        xx = sd(xx)
        xx = [float(item) for item in xx]
        yy = list(df[t[8]])
        yy = sd(yy)
        yy = [float(item) for item in yy]

        df_r = df.copy()

        df = pd.DataFrame({t[0]:a, t[1]:b, t[2]:c, t[3]:d, t[4]:e, t[5]:f, t[6]:g, t[7]:xx, t[8]:yy})
        t = list(df.columns)
        tt = t.copy()

    ##        one = list(num1.get().split())
    ##        aa = [float(item) for item in one]
    ##        two = list(num2.get().split())
    ##        bb = [float(item) for item in two]
    ##        three = list(num3.get().split())
    ##        cc = [float(item) for item in three]
    ##        four = list(num4.get().split())
    ##        dd = [float(item) for item in four]
    ##        five = list(num10.get().split())
    ##        ee = [float(item) for item in five]
    ##        six = list(num11.get().split())
    ##        ff = [float(item) for item in six]
    ##        zer = list(num0.get().split())
    ##        zz = [float(item) for item in zer]
    ##
    ##        dfx = pd.DataFrame({t[0]:zz, t[1]:aa, t[2]:bb, t[3]:cc, t[4]:dd, t[5]:ee, t[6]:ff})
    ##        dx = dfx.copy()
        dff = df.copy()

        dfv = df.query('Электропроводность > 0')
        dfv['Глубина'].corr(dfv['Электропроводность'])
        def mop(values_x,a,b):
            return a * values_x + b 
        values_x = dfv['Глубина']
        values_y = dfv['Электропроводность']
        args, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['Электропроводность'].isna(), 'Электропроводность'] = df.loc[df['Электропроводность'].isna(), 'Глубина']*args[0]+args[1]

        dfv = df.query('Монтмориллонит > 0')
        def mop(values_x,a,b):
            return a * values_x + b 
        values_x = dfv['Глубина']
        values_y = dfv['Монтмориллонит']
        args_1, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['Монтмориллонит'].isna(), 'Монтмориллонит'] = df.loc[df['Монтмориллонит'].isna(), 'Глубина']*args_1[0]+args_1[1]

        dfv = df.query('Песок > 0')
        def mop(values_x,a,b):
            return a * values_x + b 
        values_x = dfv['Монтмориллонит']
        values_y = dfv['Песок']
        args_2, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['Песок'].isna(), 'Песок'] = df.loc[df['Песок'].isna(), 'Монтмориллонит']*args_2[0]+args_2[1]

        dfv = df.query('КОЕ > 0')
        def mop(values_x,a,b):
            return a * values_x + b 
        values_x = dfv['Монтмориллонит']
        values_y = dfv['КОЕ']
        args_3, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['КОЕ'].isna(), 'КОЕ'] = df.loc[df['КОЕ'].isna(), 'Монтмориллонит']*args_3[0]+args_3[1]

        dfv = df.query('Влажность > 0')
        dfv['Монтмориллонит'].corr(dfv['Влажность'])
        def mop(values_x,a,b,c,d):
            return a * values_x + b 
        values_x = dfv['Монтмориллонит']
        values_y = dfv['Влажность']
        args, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['Влажность'].isna(), 'Влажность'] = df.loc[df['Влажность'].isna(), 'Монтмориллонит']*args[0]+args[1]
        
        dfv = df.query('Индекс > 0')
        dfv['Монтмориллонит'].corr(dfv['Индекс'])
        def mop(values_x,a,b,c,d):
            return a * values_x + b 
        values_x = dfv['Монтмориллонит']
        values_y = dfv['Индекс']
        args, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['Индекс'].isna(), 'Индекс'] = df.loc[dff['Индекс'].isna(), 'Монтмориллонит']*args[0]+args[1]

        dfg = df.copy()


        t = ['Глубина', 'X', 'Y', 'Монтмориллонит']
        df = df[t]

        df_u = df.copy()

        if list(num111.get().split()):
            seven = list(num111.get().split())
            m = [int(item) for item in seven]
            for i in range(1,8):
                if i == m[0]:
                    df = df.drop(['Монтмориллонит'], axis = 1)
                    hh = tt[i-1]
                    df = pd.concat([df, dfg[[hh]]], axis = 1)

        
        t = list(df.columns)

        def table(u, Xmin, Xmax, Ymin, Ymax):
            global df
            t = df.columns
            df = df.drop([t[-1]], axis = 1)
            hh = tt[u-1]
            df = pd.concat([df, dfg[[hh]]], axis = 1)
            t = df.columns
            df = df.loc[(df['X'] >= Xmin) & (df['X'] <= Xmax) & (df['Y'] >= Ymin) & (df['Y'] <= Ymax)]
            y = df[t[-1]]
            X = df.drop([t[-1]], axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
            scaler = StandardScaler()
            X_train_st = scaler.fit_transform(X_train)
            X_test_st = scaler.transform(X_test)

            models = [
             [Lasso(), 'Линейная регрессия Lasso'],
             [Ridge(), 'Линейная регрессия Ridge'],
             [RandomForestRegressor(n_estimators = 200, random_state = 42), 'Случайный лес'],
             [GradientBoostingRegressor(n_estimators = 200, random_state = 42), 'Градиентный бустинг'],
             [DecisionTreeRegressor(random_state = 42), 'Дерево решений']
            ]

            def prediction(mod, X_train, y_train, X_test, y_test, name):
                model = mod
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics(y_test, y_pred, name)
            p = []
            w = []
            S = (Xmax - Xmin)*(Ymax - Ymin)
            for i in models:
                model = i[0]
                model.fit(X_train_st, y_train)
                y_pred = model.predict(X_test_st)
                p.append(r2_score(y_test,y_pred))
                w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)

            if list(num444.get().split()):
                X_n = [Xmin+20*i for i in range(0,round((Xmax - Xmin)/20))]
                Y_n = [Ymin+20*i for i in range(0,round((Ymax - Ymin)/20))]

                S_1 = S/(len(X_n)*len(Y_n))
            else:
                X_n = [Xmin+20*i for i in range(0,round((Xmax - Xmin)/20))]
                Y_n = [Ymin+20*i for i in range(0,round((Ymax - Ymin)/20))]

                S_1 = S/(len(X_n)*len(Y_n))
            
            Глубина_n = [0+1*i for i in range(0,36)]

            Y_N = []
            for i in range(len(X_n)):
                Y_N.append(Y_n)
            X_N = []
            for i in X_n:
                for j in range(len(Y_n)):
                    X_N.append(i)
            Y_NN = []
            for i in Y_N:
                Y_NN += i
            X_NNN = []
            for i in X_N:
                for j in range(len(Глубина_n)):
                    X_NNN.append(i)
            Y_NNN = []
            for i in Y_NN:
                for j in range(len(Глубина_n)):
                    Y_NNN.append(i)
            H = []
            for i in range(len(Y_NN)):
                H += Глубина_n
            X_F = pd.DataFrame({'Глубина':H, 'X':X_NNN, 'Y':Y_NNN})
            X_F_st = scaler.transform(X_F)
            for i in range(len(p)):
                if p[i] == max(p):
                    model = models[i][0]
                    model.fit(X_train_st, y_train)
                    y_pred_f = model.predict(X_F_st)
            X_F[t[-1]] = y_pred_f
            print(X_F[:10])
            return X_F

        if list(num444.get().split()):
    ##            try:
            resf.configure(text = "  ")
            q = list(num444.get().split())
            Q = [float(item) for item in q]
        
            df = df.loc[(df['X'] >= Xmin) & (df['X'] <= Xmax) & (df['Y'] >= Ymin) & (df['Y'] <= Ymax)]


            y = df[t[-1]]
            X = df.drop([t[-1]], axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
            scaler = StandardScaler()
            X_train_st = scaler.fit_transform(X_train)
            X_test_st = scaler.transform(X_test)

            models = [
             [Lasso(), 'Линейная регрессия Lasso'],
             [Ridge(), 'Линейная регрессия Ridge'],
             [RandomForestRegressor(n_estimators = 200, random_state = 42), 'Случайный лес'],
             [GradientBoostingRegressor(n_estimators = 200, random_state = 42), 'Градиентный бустинг'],
             [DecisionTreeRegressor(random_state = 42), 'Дерево решений']
            ]

            def metrics(y_true, y_pred, title):
                print(title)
                print('MAE: {:.2f}'.format(mean_absolute_error(y_true,y_pred)))
                print('MSE: {:.2f}'.format(mean_squared_error(y_true,y_pred)))
                print('R2: {:.2f}'.format(r2_score(y_true,y_pred)))

            def prediction(mod, X_train, y_train, X_test, y_test, name):
                model = mod
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics(y_test, y_pred, name)
            p = []
            w = []
            for i in models:
                model = i[0]
                model.fit(X_train_st, y_train)
                y_pred = model.predict(X_test_st)
                p.append(r2_score(y_test,y_pred))
                w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)

            res26.configure(text = "r2_score: %s" % round(max(p),2))

            if max(p) == p[0]:
                print('Максимальная метрика R2-CORE у регрессии LASSO: ', p[0])
            elif max(p) == p[1]:
                print('Максимальная метрика R2-CORE у регрессии Ridge: ', p[1])
            elif max(p) == p[2]:
                print('Максимальная метрика R2-CORE у регрессии RandomForest: ', p[2])
            elif max(p) == p[3]:
                print('Максимальная метрика R2-CORE у регрессии GradientBoosting: ', p[3])  
            elif max(p) == p[4]:
                print('Максимальная метрика R2-CORE у регрессии TreeDecision: ', p[4])  
     

            fig = plt.figure(figsize = (12, 7))
                #ax = plt.axes(projection ="3d")
            ax = fig.add_subplot(111, projection='3d')

            color_map = plt.get_cmap('spring')

            if list(num111.get().split()):
                if m[0] == 7:
                    if list(num11.get().split()):
                        mont = list(num11.get().split())
                        mont = [float(item) for item in mont]
                        df_ = df.loc[(df[t[-1]] <= mont[1]) & (df[t[-1]] >= mont[0])]
                    else:
                        df_ = df.copy()
                if m[0] == 6:
                    if list(num10.get().split()):
                        koe = list(num10.get().split())
                        koe = [float(item) for item in koe]
                        df_ = df.loc[(df[t[-1]] <= koe[1]) & (df[t[-1]] >= koe[0])]
                    else:
                        df_ = df.copy()
                if m[0] == 5:
                    if list(num4.get().split()):
                        el = list(num4.get().split())
                        el = [float(item) for item in el]
                        df_ = df.loc[(df[t[-1]] <= el[1]) & (df[t[-1]] >= el[0])]
                    else:
                        df_ = df.copy()
                if m[0] == 4:
                    if list(num3.get().split()):
                        ind = list(num3.get().split())
                        ind = [float(item) for item in ind]
                        df_ = df.loc[(df[t[-1]] <= ind[1]) & (df[t[-1]] >= ind[0])]
                    else:
                        df_ = df.copy()
                if m[0] == 3:
                    if list(num2.get().split()):
                        pes = list(num2.get().split())
                        pes = [float(item) for item in pes]
                        df_ = df.loc[(df[t[-1]] <= pes[1]) & (df[t[-1]] >= pes[0])]
                    else:
                        df_ = df.copy()
                if m[0] == 2:
                    if list(num1.get().split()):
                        vl = list(num1.get().split())
                        vl = [float(item) for item in vl]
                        df_ = df.loc[(df[t[-1]] <= vl[1]) & (df[t[-1]] >= vl[0])]
                    else:
                        df_ = df.copy()
                x = df_['X']
                y = df_['Y']
                z = df_['Глубина']
                
            else:
                df_ = df.copy()
                x = df_['X']
                y = df_['Y']
                z = df_['Глубина']

            scatter_plot = ax.scatter3D(x, y, z, c = df_[t[-1]], cmap = color_map)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Глубина')
            plt.colorbar(scatter_plot)
            plt.title('3d-диаграмма фактического распределения фактора {}'.format(t[-1]))
            plt.show()
        
            X_n = [Xmin+20*i for i in range(0,round((Xmax - Xmin)/20))]
            Y_n = [Ymin+20*i for i in range(0,round((Ymax - Ymin)/20))]

            S_1 = S/(len(X_n)*len(Y_n))
            
            Глубина_n = [0+1*i for i in range(0,36)]

            Y_N = []
            for i in range(len(X_n)):
                Y_N.append(Y_n)
            X_N = []
            for i in X_n:
                for j in range(len(Y_n)):
                    X_N.append(i)
            Y_NN = []
            for i in Y_N:
                Y_NN += i
            X_NNN = []
            for i in X_N:
                for j in range(len(Глубина_n)):
                    X_NNN.append(i)
            Y_NNN = []
            for i in Y_NN:
                for j in range(len(Глубина_n)):
                    Y_NNN.append(i)
            H = []
            for i in range(len(Y_NN)):
                H += Глубина_n
            X_F = pd.DataFrame({'Глубина':H, 'X':X_NNN, 'Y':Y_NNN})
            X_F_st = scaler.transform(X_F)
            for i in range(len(p)):
                if p[i] == max(p):
                    model = models[i][0]
                    model.fit(X_train_st, y_train)
                    y_pred_f = model.predict(X_F_st)
            X_F[t[-1]] = y_pred_f

            V_o = S_1 * len(X_F[t[-1]])
            
            colorlist = ["darkorange", "gold", "lawngreen", "lightseagreen"]
            newcmp = LinearSegmentedColormap.from_list("testCmap", colors=colorlist, N=256)

            if list(num111.get().split()):
                if m[0] == 7:
                    if list(num11.get().split()):
                        mont = list(num11.get().split())
                        mont = [float(item) for item in mont]
                        X_F = X_F.loc[(X_F[t[-1]] <= mont[1]) & (X_F[t[-1]] >= mont[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                if m[0] == 6:
                    if list(num10.get().split()):
                        koe = list(num10.get().split())
                        koe = [float(item) for item in koe]
                        X_F = X_F.loc[(X_F[t[-1]] <= koe[1]) & (X_F[t[-1]] >= koe[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                if m[0] == 5:
                    if list(num4.get().split()):
                        el = list(num4.get().split())
                        el = [float(item) for item in el]
                        X_F = X_F.loc[(X_F[t[-1]] <= el[1]) & (X_F[t[-1]] >= el[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                if m[0] == 4:
                    if list(num3.get().split()):
                        ind = list(num3.get().split())
                        ind = [float(item) for item in ind]
                        X_F = X_F.loc[(X_F[t[-1]] <= ind[1]) & (X_F[t[-1]] >= ind[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                if m[0] == 3:
                    if list(num2.get().split()):
                        pes = list(num2.get().split())
                        pes = [float(item) for item in pes]
                        X_F = X_F.loc[(X_F[t[-1]] <= pes[1]) & (X_F[t[-1]] >= pes[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                if m[0] == 2:
                    if list(num1.get().split()):
                        vl = list(num1.get().split())
                        vl = [float(item) for item in vl]
                        X_F = X_F.loc[(X_F[t[-1]] <= vl[1]) & (X_F[t[-1]] >= vl[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                if len(m) > 1:
                    
                    X_F_1 = table(2, Xmin, Xmax, Ymin, Ymax)
                    t11 = X_F_1.columns
                    if list(num1.get().split()):
                        vl = list(num1.get().split())
                        vl = [float(item) for item in vl]
                    else:
                        vl = [min(X_F_1['Влажность']), max(X_F_1['Влажность'])]
                    X_F_2 = table(3, Xmin, Xmax, Ymin, Ymax)
                    t22 = X_F_2.columns    
                    if list(num2.get().split()):
                        pes = list(num2.get().split())
                        pes = [float(item) for item in pes]
                    else:
                        pes = [min(X_F_2['Песок']), max(X_F_2['Песок'])]
                    X_F_3 = table(4, Xmin, Xmax, Ymin, Ymax)
                    t33 = X_F_3.columns    
                    if list(num3.get().split()):
                        ind = list(num3.get().split())
                        ind = [float(item) for item in ind]
                    else:
                        ind = [min(X_F_3['Индекс']), max(X_F_3['Индекс'])]
                    X_F_4 = table(5, Xmin, Xmax, Ymin, Ymax)
                    t44 = X_F_4.columns       
                    if list(num4.get().split()):
                        el = list(num4.get().split())
                        el = [float(item) for item in el]
                    else:
                        el = [min(X_F_4['Электропроводность']), max(X_F_4['Электропроводность'])]
                    X_F_5 = table(7, Xmin, Xmax, Ymin, Ymax)
                    t55 = X_F_5.columns     
                    if list(num11.get().split()):
                        mont = list(num11.get().split())
                        mont = [float(item) for item in mont]
                    else:
                        mont = [min(X_F_5['Монтмориллонит']), max(X_F_5['Монтмориллонит'])]
                    X_F_6 = table(6, Xmin, Xmax, Ymin, Ymax)
                    t66 = X_F_6.columns
                    if list(num10.get().split()):
                        koe = list(num10.get().split())
                        koe = [float(item) for item in koe]
                    else:
                        koe = [min(X_F_6['КОЕ']), max(X_F_6['КОЕ'])]
                    
                    X_F = X_F.drop([t[-1]], axis = 1)
                    X_F['Влажность'] = X_F_1[t11[-1]]
                    X_F['Песок'] = X_F_2[t22[-1]]
                    X_F['Индекс'] = X_F_3[t33[-1]]
                    X_F['Электропроводность'] = X_F_4[t44[-1]]
                    X_F['КОЕ'] = X_F_6[t66[-1]]
                    X_F['Монтмориллонит'] = X_F_5[t55[-1]]
                    X_F = X_F.loc[(X_F['Влажность'] <= vl[1]) & (X_F['Влажность'] >= vl[0])]
                    X_F = X_F.loc[(X_F['Песок'] <= pes[1]) & (X_F['Песок'] >= pes[0])]
                    X_F = X_F.loc[(X_F['Индекс'] <= ind[1]) & (X_F['Индекс'] >= ind[0])]
                    X_F = X_F.loc[(X_F['Электропроводность'] <= el[1]) & (X_F['Электропроводность'] >= el[0])]
                    X_F = X_F.loc[(X_F['Монтмориллонит'] <= mont[1]) & (X_F['Монтмориллонит'] >= mont[0])]
                    X_F = X_F.loc[(X_F['КОЕ'] <= koe[1]) & (X_F['КОЕ'] >= koe[0])]
                                  
                    X_F = X_F[['X', 'Y', 'Глубина', t[-1]]]
        
                    V_o = S_1 * len(X_F[t[-1]])
            
            res32.configure(text = "Запасы сырья, м3: %s" % round(V_o,1))


                
            print('запас', V_o)
            x = X_F['X']
            y = X_F['Y']
            z = X_F['Глубина']

            fig = plt.figure(figsize = (12, 7))
            #ax = plt.axes(projection ="3d")
            ax = fig.add_subplot(111, projection='3d')

            color_map = plt.get_cmap('spring')
            scatter_plot = ax.scatter3D(x, y, z, c = X_F[t[-1]], cmap = 'coolwarm')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Глубина')
            plt.colorbar(scatter_plot)
            plt.title('3d-модель распределения фактора {}'.format(t[-1]))
            plt.show()
        
            df = df_r
    ##        except:
    ##            resf.configure(text = "Неверное положение виджетов")
    ##        finally:
    ##            df = df_r  
        else:
            resf.configure(text = "  ")
            y = df[t[-1]]
            X = df.drop([t[-1]], axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
            scaler = StandardScaler()
            X_train_st = scaler.fit_transform(X_train)
            X_test_st = scaler.transform(X_test)

            models = [
                 [Lasso(), 'Линейная регрессия Lasso'],
                 [Ridge(), 'Линейная регрессия Ridge'],
                 [RandomForestRegressor(n_estimators = 200, random_state = 42), 'Случайный лес'],
                 [GradientBoostingRegressor(n_estimators = 200, random_state = 42), 'Градиентный бустинг'],
                 [DecisionTreeRegressor(random_state = 42), 'Дерево решений']
            ]

            def metrics(y_true, y_pred, title):
                print(title)
                print('MAE: {:.2f}'.format(mean_absolute_error(y_true,y_pred)))
                print('MSE: {:.2f}'.format(mean_squared_error(y_true,y_pred)))
                print('R2: {:.2f}'.format(r2_score(y_true,y_pred)))

            def prediction(mod, X_train, y_train, X_test, y_test, name):
                model = mod
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics(y_test, y_pred, name)
            if list(num555.get().split()):
                c = df.corr()
                plt.figure(figsize = (8,8))
                sns.heatmap(c, annot = True, square = True)
                plt.title('Матрица корреляции признаков')
                plt.show()

            for i in models:
                prediction(i[0],X_train_st,y_train,X_test_st,y_test,i[1])

            p = []
            w = []
            for i in models:
                model = i[0]
                model.fit(X_train_st, y_train)
                y_pred = model.predict(X_test_st)
                p.append(r2_score(y_test,y_pred))
                w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)

            res26.configure(text = "r2_score: %s" % round(max(p),2))
        
            if max(p) == p[0]:
                print('Максимальная метрика R2-CORE у регрессии LASSO: ', p[0])
            elif max(p) == p[1]:
                print('Максимальная метрика R2-CORE у регрессии Ridge: ', p[1])
            elif max(p) == p[2]:
                print('Максимальная метрика R2-CORE у регрессии RandomForest: ', p[2])
            elif max(p) == p[3]:
                print('Максимальная метрика R2-CORE у регрессии GradientBoosting: ', p[3])  
            elif max(p) == p[4]:
                print('Максимальная метрика R2-CORE у регрессии TreeDecision: ', p[4])      
            if min(w) == w[0]:
                print('Минимальная средняя метрика у регрессии LASSO: ', w[0])
            elif min(w) == w[1]:
                print('Минимальная средняя метрика у регрессии Ridge: ', w[1])
            elif min(w) == w[2]:
                print('Минимальная средняя метрика у регрессии RandomForest: ', w[2])
            elif min(w) == w[3]:
                print('Минимальная средняя метрика у : регрессии GradientBoosting', w[3])
            elif min(w) == w[4]:
                print('Минимальная средняя метрика у : регрессии TreeDecision', w[4])

            def graph(model, f):
                u = t.copy()
                dft = df.copy()
                dft = dft.drop(t[-1], axis = 1)
                for i in dft.columns:
                    dft[i] = round(dft[i].mean(),2)
                del u[-1]
                for i in u:
                    dfr = dft.copy()
                    for j in range(len(dfr[i])):
                        mi = df[i].min()
                        ma = df[i].max()
                        n = (ma-mi)/(len(dft[i])-1)
                        dfr[i][j] = mi + j*n
                    X_st = scaler.transform(dfr)
                    y_pred = model.predict(X_st)
                    dfr[t[-1]] = y_pred
                    dg = dfr.groupby(i).agg({t[-1]:'mean'})
        
                    def mop(values_x,a,b,c,d):
                        return a * values_x**3 + b * values_x**2 + c * values_x + d
                    def mop1(values_x,a,b,c):
                        return a * values_x**2 + b * values_x + c 
                    def mop2(values_x,a,b):
                        return a * values_x + b 
                    values_x = dfr[i]
                    values_y = dfr[t[-1]]
            
                    args, covar = curve_fit(mop, values_x, values_y)
                    y_pred_1 = mop(values_x, *args)
                    r2_1 = r2_score(values_y, y_pred_1)
                    args1, covar1 = curve_fit(mop1, values_x, values_y)
                    y_pred_2 = mop1(values_x, *args1)
                    r2_2 = r2_score(values_y, y_pred_2)
                    args2, covar2 = curve_fit(mop2, values_x, values_y)
                    y_pred_3 = mop2(values_x, *args2)
                    r2_3 = r2_score(values_y, y_pred_3)
                    r2 = [r2_1,r2_2,r2_3]
                    y = []
                    if max(r2) == r2_1:
                        for j in range(len(dfr[i])):
                            y.append(args[0] * dfr[i][j]**3 + args[1] * dfr[i][j]**2 + args[2] * dfr[i][j] + args[3])
                    elif max(r2) == r2_2:
                        for j in range(len(dfr[i])):
                            y.append(args1[0] * dfr[i][j]**2 + args1[1] * dfr[i][j] + args1[2])    
                    elif max(r2) == r2_3:
                        for j in range(len(dfr[i])):
                            y.append(args2[0] * dfr[i][j] + args2[1]) 
                    dgr = pd.DataFrame({'x':list(dfr[i]), 'y':y})
            
                    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
                    sns.set_style('whitegrid')
                    sns.set_palette('bright')
            
                    sns.lineplot(dgr.pivot_table(index = 'x', values = 'y', aggfunc = 'mean'), color = 'red')
            
                    sns.scatterplot(data=dg)
                    #sns.set(rc={'figure.figsize':(8,5)})
                    axes.set(xlabel= 'Переменный параметр <{}>'.format(i),
                    ylabel='Функция отклика <{}>'.format(t[-1]),
                    title ='Усредненная параметрическая диаграмма {} при переменном факторе {} ({})'.format(t[-1], i, f))
                    plt.legend(title = '{}'.format(dft.iloc[[0]]), loc=2, bbox_to_anchor=(1, 1), fontsize = 10)
                    plt.xticks(rotation = 10)    
                    plt.show()

            if list(num555.get().split()):    
                if max(p) == p[0]:
                    graph(models[0][0], models[0][1])
                elif max(p) == p[1]:
                    graph(models[1][0], models[1][1])
                elif max(p) == p[2]:
                    graph(models[2][0], models[2][1])
                elif max(p) == p[3]:
                    graph(models[3][0], models[3][1])
                elif max(p) == p[4]:
                    graph(models[4][0], models[4][1])   


            if list(num111.get().split()):
                if m[0] == 7:
                    if list(num11.get().split()):
                        mont = list(num11.get().split())
                        mont = [float(item) for item in mont]
                        df_ = df.loc[(df[t[-1]] <= mont[1]) & (df[t[-1]] >= mont[0])]
                    else:
                        df_ = df.copy()
                if m[0] == 6:
                    if list(num10.get().split()):
                        koe = list(num10.get().split())
                        koe = [float(item) for item in koe]
                        df_ = df.loc[(df[t[-1]] <= koe[1]) & (df[t[-1]] >= koe[0])]
                    else:
                        df_ = df.copy()
                if m[0] == 5:
                    if list(num4.get().split()):
                        el = list(num4.get().split())
                        el = [float(item) for item in el]
                        df_ = df.loc[(df[t[-1]] <= el[1]) & (df[t[-1]] >= el[0])]
                    else:
                        df_ = df.copy()
                if m[0] == 4:
                    if list(num3.get().split()):
                        ind = list(num3.get().split())
                        ind = [float(item) for item in ind]
                        df_ = df.loc[(df[t[-1]] <= ind[1]) & (df[t[-1]] >= ind[0])]
                    else:
                        df_ = df.copy()
                if m[0] == 3:
                    if list(num2.get().split()):
                        pes = list(num2.get().split())
                        pes = [float(item) for item in pes]
                        df_ = df.loc[(df[t[-1]] <= pes[1]) & (df[t[-1]] >= pes[0])]
                    else:
                        df_ = df.copy()
                if m[0] == 2:
                    if list(num1.get().split()):
                        vl = list(num1.get().split())
                        vl = [float(item) for item in vl]
                        df_ = df.loc[(df[t[-1]] <= vl[1]) & (df[t[-1]] >= vl[0])]
                    else:
                        df_ = df.copy()
                x = df_['X']
                y = df_['Y']
                z = df_['Глубина']
                
            else:
                df_ = df.copy()
                x = df_['X']
                y = df_['Y']
                z = df_['Глубина']

         

            fig = plt.figure(figsize = (12, 7))
            #ax = plt.axes(projection ="3d")
            ax = fig.add_subplot(111, projection='3d')

            color_map = plt.get_cmap('spring')

        
            scatter_plot = ax.scatter3D(x, y, z, c = df_[t[-1]], cmap = color_map)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Глубина')
            plt.colorbar(scatter_plot)
            plt.title('3d-диаграмма фактического распределения фактора {}'.format(t[-1]))
            plt.show()


            X_n = [min(df['X'])+20*i for i in range(0,round((max(df['X']) - min(df['X']))/20))]
            Y_n = [min(df['Y'])+20*i for i in range(0,round((max(df['Y']) - min(df['Y']))/20))]

            S_1 = (max(df['X']) - min(df['X']))*(max(df['Y']) - min(df['Y']))/(len(X_n)*len(Y_n))

           
            
            Глубина_n = [0+1*i for i in range(0,36)]


            Y_N = []
            for i in range(len(X_n)):
                Y_N.append(Y_n)
            X_N = []
            for i in X_n:
                for j in range(len(Y_n)):
                    X_N.append(i)
            Y_NN = []
            for i in Y_N:
                Y_NN += i
            X_NNN = []
            for i in X_N:
                for j in range(len(Глубина_n)):
                    X_NNN.append(i)
            Y_NNN = []
            for i in Y_NN:
                for j in range(len(Глубина_n)):
                    Y_NNN.append(i)
            H = []
            for i in range(len(Y_NN)):
                H += Глубина_n
            X_F = pd.DataFrame({'Глубина':H, 'X':X_NNN, 'Y':Y_NNN})
            X_F_st = scaler.transform(X_F)
            for i in range(len(p)):
                if p[i] == max(p):
                    model = models[i][0]
                    model.fit(X_train_st, y_train)
                    y_pred_f = model.predict(X_F_st)
            X_F[t[-1]] = y_pred_f

            V_o = S_1 * len(X_F[t[-1]])

            colorlist = ["darkorange", "gold", "lawngreen", "lightseagreen"]
            newcmp = LinearSegmentedColormap.from_list("testCmap", colors=colorlist, N=256)

            if list(num111.get().split()):
                if m[0] == 7:
                    if list(num11.get().split()):
                        mont = list(num11.get().split())
                        mont = [float(item) for item in mont]
                        X_F = X_F.loc[(X_F[t[-1]] <= mont[1]) & (X_F[t[-1]] >= mont[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                        print(min(X_F[t[-1]]), max(X_F[t[-1]]))
                if m[0] == 6:
                    if list(num10.get().split()):
                        koe = list(num10.get().split())
                        koe = [float(item) for item in koe]
                        X_F = X_F.loc[(X_F[t[-1]] <= koe[1]) & (X_F[t[-1]] >= koe[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                if m[0] == 5:
                    if list(num4.get().split()):
                        el = list(num4.get().split())
                        el = [float(item) for item in el]
                        X_F = X_F.loc[(X_F[t[-1]] <= el[1]) & (X_F[t[-1]] >= el[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                if m[0] == 4:
                    if list(num3.get().split()):
                        ind = list(num3.get().split())
                        ind = [float(item) for item in ind]
                        X_F = X_F.loc[(X_F[t[-1]] <= ind[1]) & (X_F[t[-1]] >= ind[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                if m[0] == 3:
                    if list(num2.get().split()):
                        pes = list(num2.get().split())
                        pes = [float(item) for item in pes]
                        X_F = X_F.loc[(X_F[t[-1]] <= pes[1]) & (X_F[t[-1]] >= pes[0])]
                        V_o = S_1 * len(X_F[t[-1]])
                        print(X_F)
                if m[0] == 2:
                    if list(num1.get().split()):
                        vl = list(num1.get().split())
                        vl = [float(item) for item in vl]
                        X_F = X_F.loc[(X_F[t[-1]] <= vl[1]) & (X_F[t[-1]] >= vl[0])]
                        V_o = S_1 * len(X_F[t[-1]])

                if len(m) > 1:
                    
                    X_F_1 = table(2, min(dfg['X']), max(dfg['X']), min(dfg['Y']), max(dfg['Y']))
                    t11 = X_F_1.columns
                    if list(num1.get().split()):
                        vl = list(num1.get().split())
                        vl = [float(item) for item in vl]
                    else:
                        vl = [min(X_F_1['Влажность']), max(X_F_1['Влажность'])]
                    X_F_2 = table(3, min(dfg['X']), max(dfg['X']), min(dfg['Y']), max(dfg['Y']))
                    t22 = X_F_2.columns    
                    if list(num2.get().split()):
                        pes = list(num2.get().split())
                        pes = [float(item) for item in pes]
                    else:
                        pes = [min(X_F_2['Песок']), max(X_F_2['Песок'])]
                    X_F_3 = table(4, min(dfg['X']), max(dfg['X']), min(dfg['Y']), max(dfg['Y']))
                    t33 = X_F_3.columns    
                    if list(num3.get().split()):
                        ind = list(num3.get().split())
                        ind = [float(item) for item in ind]
                    else:
                        ind = [min(X_F_3['Индекс']), max(X_F_3['Индекс'])]
                    X_F_4 = table(5, min(dfg['X']), max(dfg['X']), min(dfg['Y']), max(dfg['Y']))
                    t44 = X_F_4.columns       
                    if list(num4.get().split()):
                        el = list(num4.get().split())
                        el = [float(item) for item in el]
                    else:
                        el = [min(X_F_4['Электропроводность']), max(X_F_4['Электропроводность'])]
                    X_F_5 = table(7, min(dfg['X']), max(dfg['X']), min(dfg['Y']), max(dfg['Y']))
                    t55 = X_F_5.columns     
                    if list(num11.get().split()):
                        mont = list(num11.get().split())
                        mont = [float(item) for item in mont]
                    else:
                        mont = [min(X_F_5['Монтмориллонит']), max(X_F_5['Монтмориллонит'])]
                    X_F_6 = table(6, min(dfg['X']), max(dfg['X']), min(dfg['Y']), max(dfg['Y']))
                    t66 = X_F_6.columns
                    if list(num10.get().split()):
                        koe = list(num10.get().split())
                        koe = [float(item) for item in koe]
                    else:
                        koe = [min(X_F_6['КОЕ']), max(X_F_6['КОЕ'])]
                    
                    X_F = X_F.drop([t[-1]], axis = 1)
                    X_F['Влажность'] = X_F_1[t11[-1]]
                    X_F['Песок'] = X_F_2[t22[-1]]
                    X_F['Индекс'] = X_F_3[t33[-1]]
                    X_F['Электропроводность'] = X_F_4[t44[-1]]
                    X_F['КОЕ'] = X_F_6[t66[-1]]
                    X_F['Монтмориллонит'] = X_F_5[t55[-1]]
                    X_F = X_F.loc[(X_F['Влажность'] <= vl[1]) & (X_F['Влажность'] >= vl[0])]
                    X_F = X_F.loc[(X_F['Песок'] <= pes[1]) & (X_F['Песок'] >= pes[0])]
                    X_F = X_F.loc[(X_F['Индекс'] <= ind[1]) & (X_F['Индекс'] >= ind[0])]
                    X_F = X_F.loc[(X_F['Электропроводность'] <= el[1]) & (X_F['Электропроводность'] >= el[0])]
                    X_F = X_F.loc[(X_F['Монтмориллонит'] <= mont[1]) & (X_F['Монтмориллонит'] >= mont[0])]
                    X_F = X_F.loc[(X_F['КОЕ'] <= koe[1]) & (X_F['КОЕ'] >= koe[0])]
                                  
                    X_F = X_F[['X', 'Y', 'Глубина', t[-1]]]
                    
                    V_o = S_1 * len(X_F[t[-1]])
            
            res32.configure(text = "Запасы сырья, м3: %s" % round(V_o,1))
            
            
            x = X_F['X']
            y = X_F['Y']
            z = X_F['Глубина']

            fig = plt.figure(figsize = (12, 7))
            #ax = plt.axes(projection ="3d")
            ax = fig.add_subplot(111, projection='3d')

            color_map = plt.get_cmap('spring')
            scatter_plot = ax.scatter3D(x, y, z, c = X_F[t[-1]], cmap = 'coolwarm')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Глубина')
            plt.colorbar(scatter_plot)
            plt.title('3d-модель распределения фактора {}'.format(t[-1]))
            plt.show()

            df = df_r
    ##    except:
    ##        resf.configure(text = "Ошибка ввода. Повторите загрузку данных")
    ##    finally:
    ##        df = df_r


    ##    try:
    ##        
    ##        global df
    ##        global scale_widget_1
    ##        global scale_widget_2
    ##        global scale_widget_3
    ##        global scale_widget_4
    ##        global scale_widget_5
    ##        global scale_widget_6
    ##
    ##        Xmin = scale_widget_1.get() 
    ##        Xmax = scale_widget_2.get() 
    ##        Ymin = scale_widget_3.get() 
    ##        Ymax = scale_widget_4.get()
    ##
    ##        
    ##        Xt = scale_widget_5.get()
    ##        Yt = scale_widget_6.get()
    ##        
    ##            
    ##        
    ##        def sd(x):
    ##            x1 = []
    ##            for i in x:
    ##                i = str(i)
    ##                x1.append(i.replace(",", "."))
    ##            return x1
    ##        t = list(df.columns)
    ##        
    ##        a = list(df[t[0]])
    ##        a = sd(a)
    ##        a = [float(item) for item in a]
    ##        #a = [x for x in a if str(x) != 'nan']
    ##        b = list(df[t[1]])
    ##        b = sd(b)
    ##        b = [float(item) for item in b]
    ##        c = list(df[t[2]])
    ##        c = sd(c)
    ##        c = [float(item) for item in c]
    ##        d = list(df[t[3]])
    ##        d = sd(d)
    ##        d = [float(item) for item in d]
    ##        e = list(df[t[4]])
    ##        e = sd(e)
    ##        e = [float(item) for item in e]
    ##        f = list(df[t[5]])
    ##        f = sd(f)
    ##        f = [float(item) for item in f]
    ##        g = list(df[t[6]])
    ##        g = sd(g)
    ##        g = [float(item) for item in g]
    ##        xx = list(df[t[7]])
    ##        xx = sd(xx)
    ##        xx = [float(item) for item in xx]
    ##        yy = list(df[t[8]])
    ##        yy = sd(yy)
    ##        yy = [float(item) for item in yy]
    ##
    ##        df_r = df.copy()
    ##        df = pd.DataFrame({t[0]:a, t[1]:b, t[2]:c, t[3]:d, t[4]:e, t[5]:f, t[6]:g, t[7]:xx, t[8]:yy})
    ##            
    ##
    ##        one = list(num1.get().split())
    ##        aa = [float(item) for item in one]
    ##        two = list(num2.get().split())
    ##        bb = [float(item) for item in two]
    ##        three = list(num3.get().split())
    ##        cc = [float(item) for item in three]
    ##        four = list(num4.get().split())
    ##        dd = [float(item) for item in four]
    ##        five = list(num10.get().split())
    ##        ee = [float(item) for item in five]
    ##        six = list(num11.get().split())
    ##        ff = [float(item) for item in six]
    ##        zer = list(num0.get().split())
    ##        zz = [float(item) for item in zer]
    ##    ##    xxx = list(num333.get().split())
    ##    ##    cross = [float(item) for item in xxx]
    ##
    ##    ##    dfx = pd.DataFrame({t[0]:zz, t[1]:aa, t[2]:bb, t[3]:cc, t[4]:dd, t[5]:ee, t[6]:ff, t[7]:Xt, t[8]:Yt})
    ##    ##    dx = dfx.copy()
    ##        dff = df.copy()
    ##
    ##        dfv = df.query('Электропроводность > 0')
    ##        dfv['Глубина'].corr(dfv['Электропроводность'])
    ##        def mop(values_x,a,b):
    ##            return a * values_x + b 
    ##        values_x = dfv['Глубина']
    ##        values_y = dfv['Электропроводность']
    ##        args, covar = curve_fit(mop, values_x, values_y)
    ##        df.loc[df['Электропроводность'].isna(), 'Электропроводность'] = df.loc[df['Электропроводность'].isna(), 'Глубина']*args[0]+args[1]
    ##
    ##        dfv = df.query('Монтмориллонит > 0')
    ##        def mop(values_x,a,b):
    ##            return a * values_x + b 
    ##        values_x = dfv['Глубина']
    ##        values_y = dfv['Монтмориллонит']
    ##        args_1, covar = curve_fit(mop, values_x, values_y)
    ##        df.loc[df['Монтмориллонит'].isna(), 'Монтмориллонит'] = df.loc[df['Монтмориллонит'].isna(), 'Глубина']*args_1[0]+args_1[1]
    ##
    ##        dfv = df.query('Песок > 0')
    ##        def mop(values_x,a,b):
    ##            return a * values_x + b 
    ##        values_x = dfv['Монтмориллонит']
    ##        values_y = dfv['Песок']
    ##        args_2, covar = curve_fit(mop, values_x, values_y)
    ##        df.loc[df['Песок'].isna(), 'Песок'] = df.loc[df['Песок'].isna(), 'Монтмориллонит']*args_2[0]+args_2[1]
    ##
    ##        dfv = df.query('КОЕ > 0')
    ##        def mop(values_x,a,b):
    ##            return a * values_x + b 
    ##        values_x = dfv['Монтмориллонит']
    ##        values_y = dfv['КОЕ']
    ##        args_3, covar = curve_fit(mop, values_x, values_y)
    ##        df.loc[df['КОЕ'].isna(), 'КОЕ'] = df.loc[df['КОЕ'].isna(), 'Монтмориллонит']*args_3[0]+args_3[1]
    ##
    ##        dfv = df.query('Влажность > 0')
    ##        dfv['Монтмориллонит'].corr(dfv['Влажность'])
    ##        def mop(values_x,a,b,c,d):
    ##            return a * values_x + b 
    ##        values_x = dfv['Монтмориллонит']
    ##        values_y = dfv['Влажность']
    ##        args, covar = curve_fit(mop, values_x, values_y)
    ##        df.loc[df['Влажность'].isna(), 'Влажность'] = df.loc[df['Влажность'].isna(), 'Монтмориллонит']*args[0]+args[1]
    ##        
    ##        dfv = df.query('Индекс > 0')
    ##        dfv['Монтмориллонит'].corr(dfv['Индекс'])
    ##        def mop(values_x,a,b,c,d):
    ##            return a * values_x + b 
    ##        values_x = dfv['Монтмориллонит']
    ##        values_y = dfv['Индекс']
    ##        args, covar = curve_fit(mop, values_x, values_y)
    ##        df.loc[df['Индекс'].isna(), 'Индекс'] = df.loc[dff['Индекс'].isna(), 'Монтмориллонит']*args[0]+args[1]
    ##
    ##        df = df.loc[(df['X'] >= Xmin) & (df['X'] <= Xmax) & (df['Y'] >= Ymin) & (df['Y'] <= Ymax)]
    ##        tt = t.copy()
    ##        dfg = df.copy()
    ##
    ##        t = ['Глубина', 'X', 'Y', 'Монтмориллонит']
    ##        df = df[t]
    ##
    ##        if list(num111.get().split()):
    ##            seven = list(num111.get().split())
    ##            m = [float(item) for item in seven]
    ##            for i in range(1,8):
    ##                if i == m[0]:
    ##                    df = df.drop(['Монтмориллонит'], axis = 1)
    ##                    hh = tt[i-1]
    ##                    df = pd.concat([df, dfg[[hh]]], axis = 1)
    ##                    print(df)
    ##        
    ##        f = list(df.columns)          
    ##        y = df[f[-1]]
    ##        X = df.drop([f[-1]], axis = 1)
    ##        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
    ##        scaler = StandardScaler()
    ##        X_train_st = scaler.fit_transform(X_train)
    ##        X_test_st = scaler.transform(X_test)
    ##
    ##        models = [
    ##             [Lasso(), 'Линейная регрессия Lasso'],
    ##             [Ridge(), 'Линейная регрессия Ridge'],
    ##             [RandomForestRegressor(n_estimators = 200, random_state = 0), 'Случайный лес'],
    ##             [GradientBoostingRegressor(n_estimators = 200, random_state = 0), 'Градиентный бустинг'],
    ##             [DecisionTreeRegressor(random_state = 0), 'Дерево решений']
    ##        ]
    ##
    ##        def metrics(y_true, y_pred, title):
    ##            print(title)
    ##            print('MAE: {:.2f}'.format(mean_absolute_error(y_true,y_pred)))
    ##            print('MSE: {:.2f}'.format(mean_squared_error(y_true,y_pred)))
    ##            print('R2: {:.2f}'.format(r2_score(y_true,y_pred)))
    ##
    ##        def prediction(mod, X_train, y_train, X_test, y_test, name):
    ##            model = mod
    ##            model.fit(X_train, y_train)
    ##            y_pred = model.predict(X_test)
    ##            metrics(y_test, y_pred, name)
    ##
    ##        p = []
    ##        w = []
    ##        for i in models:
    ##            model = i[0]
    ##            model.fit(X_train_st, y_train)
    ##            y_pred = model.predict(X_test_st)
    ##            p.append(r2_score(y_test,y_pred))
    ##            w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)
    ##        if max(p) == p[0]:
    ##            print('Максимальная метрика R2-CORE у регрессии LASSO: ', p[0])
    ##        elif max(p) == p[1]:
    ##            print('Максимальная метрика R2-CORE у регрессии Ridge: ', p[1])
    ##        elif max(p) == p[2]:
    ##            print('Максимальная метрика R2-CORE у регрессии RandomForest: ', p[2])
    ##        elif max(p) == p[3]:
    ##            print('Максимальная метрика R2-CORE у регрессии GradientBoosting: ', p[3])  
    ##        elif max(p) == p[4]:
    ##            print('Максимальная метрика R2-CORE у регрессии TreeDecision: ', p[4])      
    ##        if min(w) == w[0]:
    ##            print('Минимальная средняя метрика у регрессии LASSO: ', w[0])
    ##        elif min(w) == w[1]:
    ##            print('Минимальная средняя метрика у регрессии Ridge: ', w[1])
    ##        elif min(w) == w[2]:
    ##            print('Минимальная средняя метрика у регрессии RandomForest: ', w[2])
    ##        elif min(w) == w[3]:
    ##            print('Минимальная средняя метрика у : регрессии GradientBoosting', w[3])
    ##        elif min(w) == w[4]:
    ##            print('Минимальная средняя метрика у : регрессии TreeDecision', w[4])
    ##
    ##        Глубина_p = [0+1*i for i in range(36)]
    ##        X_p = Xt
    ##        Y_p = Yt
    ##        X_P = [X_p for i in range(len(Глубина_p))]
    ##        Y_P = [Y_p for i in range(len(Глубина_p))]
    ##        X_PP = pd.DataFrame({'Глубина':Глубина_p, 'X':X_P, 'Y':Y_P})
    ##        X_P_st = scaler.transform(X_PP)
    ##        for i in range(len(p)):
    ##            if p[i] == max(p):
    ##                model = models[i][0]
    ##                #model.fit(X_train_st, y_train)
    ##                y_pred_p = model.predict(X_P_st)
    ##        X_PP[f[-1]] = y_pred_p
    ##        if list(num444.get().split()):
    ##            q = list(num444.get().split())
    ##            Q = [float(item) for item in q]
    ##            X_t = pd.DataFrame({'Глубина':Q[0], 'X':X_P, 'Y':Y_P})
    ##            X_t_st = scaler.transform(X_t)
    ##        
    ##            for i in range(len(p)):
    ##                if p[i] == max(p):
    ##                    model = models[i][0]
    ##                   # model.fit(X_train_st, y_train)
    ##                    y_pred_t = model.predict(X_t_st)
    ##                    res28.configure(text = "%s : %s" % (f[-1], round(y_pred_t[0],1)))                
    ##
    ##        colorlist = ["darkorange", "gold", "lawngreen", "lightseagreen"]
    ##        newcmp = LinearSegmentedColormap.from_list("testCmap", colors=colorlist, N=256)
    ##
    ##        x = X_PP['X']
    ##        y = X_PP['Y']
    ##        z = X_PP['Глубина']
    ##
    ##        fig = plt.figure(figsize = (12, 7))
    ##        #ax = plt.axes(projection ="3d")
    ##        ax = fig.add_subplot(111, projection='3d')
    ##
    ##        color_map = plt.get_cmap('spring')
    ##        scatter_plot = ax.scatter3D(x, y, z, c = X_PP[f[-1]], cmap = 'coolwarm')
    ##        ax.set_xlabel('X')
    ##        ax.set_ylabel('Y')
    ##        ax.set_zlabel('Глубина')
    ##        plt.colorbar(scatter_plot)
    ##        plt.title('3d-модель точечного распределения фактора {}'.format(f[-1]))
    ##        plt.show()
    ##        
    ##        if (Xt > Xmax) or (Xt < Xmin) or (Yt > Ymax) or (Yt < Ymin):
    ##            resf.configure(text = "Выбранная точке вне модельного поля")
    ##        else:
    ##            resf.configure(text = "  ")
    ##
        df = df_r
    except:
        resf.configure(text = "Ошибка загрузки данных")
    finally:
        df = df_r


def ddf():
##    try:
    global df
    if list(num444.get().split()):
        at_ = [at[0], at[1], at[2], at[3]]
        bt_ = [bt[0], bt[1], bt[2], bt[3]]

        ab = at_+bt_
        x_ = []
        y_ = []
        for i in range(len(ab)):
            x_.append(5267188.06 + 1.41*ab[i][1])
            y_.append(14715755.24 + 1.426*ab[i][0])
        Xmin = min(x_) 
        Xmax = max(x_) 
        Ymin = min(y_) 
        Ymax = max(y_)

                
        Xt = 5267188.06 + 1.41*bt[-1][1]
        Yt = 14715755.24 + 1.426*bt[-1][0]
        
        
        res30.configure(text = "Xmin-Xmax-Ymin-Ymax: %s - %s - %s - %s " % (round(Xmin,1), round(Xmax,1), round(Ymin,1), round(Ymax,1)))
        res31.configure(text = "Координаты точки X-Y: %s - %s" % (round(Xt,1), round(Yt,1)))   

        S = (Xmax - Xmin)*(Ymax - Ymin)

        print(Xmin)
        print(Xmax)
    
    
    def sd(x):
        x1 = []
        for i in x:
            i = str(i)
            x1.append(i.replace(",", "."))
        return x1
    t = list(df.columns)
    a = list(df[t[0]])
    a = sd(a)
    a = [float(item) for item in a]
    #a = [x for x in a if str(x) != 'nan']
    b = list(df[t[1]])
    b = sd(b)
    b = [float(item) for item in b]
    c = list(df[t[2]])
    c = sd(c)
    c = [float(item) for item in c]
    d = list(df[t[3]])
    d = sd(d)
    d = [float(item) for item in d]
    e = list(df[t[4]])
    e = sd(e)
    e = [float(item) for item in e]
    f = list(df[t[5]])
    f = sd(f)
    f = [float(item) for item in f]
    g = list(df[t[6]])
    g = sd(g)
    g = [float(item) for item in g]
    xx = list(df[t[7]])
    xx = sd(xx)
    xx = [float(item) for item in xx]
    yy = list(df[t[8]])
    yy = sd(yy)
    yy = [float(item) for item in yy]
    hh = list(df[t[10]])
    hh = sd(hh)
    hh = [float(item) for item in hh]

    df_r = df.copy()

    df = pd.DataFrame({t[0]:a, t[1]:b, t[2]:c, t[3]:d, t[4]:e, t[5]:f, t[6]:g, t[7]:xx, t[8]:yy, t[10]:hh})
    t = list(df.columns)
    tt = t.copy()

##        one = list(num1.get().split())
##        aa = [float(item) for item in one]
##        two = list(num2.get().split())
##        bb = [float(item) for item in two]
##        three = list(num3.get().split())
##        cc = [float(item) for item in three]
##        four = list(num4.get().split())
##        dd = [float(item) for item in four]
##        five = list(num10.get().split())
##        ee = [float(item) for item in five]
##        six = list(num11.get().split())
##        ff = [float(item) for item in six]
##        zer = list(num0.get().split())
##        zz = [float(item) for item in zer]
##
##        dfx = pd.DataFrame({t[0]:zz, t[1]:aa, t[2]:bb, t[3]:cc, t[4]:dd, t[5]:ee, t[6]:ff})
##        dx = dfx.copy()
    dff = df.copy()

    dfv = df.query('Электропроводность > 0')
    dfv['Глубина'].corr(dfv['Электропроводность'])
    def mop(values_x,a,b):
        return a * values_x + b 
    values_x = dfv['Глубина']
    values_y = dfv['Электропроводность']
    args, covar = curve_fit(mop, values_x, values_y)
    df.loc[df['Электропроводность'].isna(), 'Электропроводность'] = df.loc[df['Электропроводность'].isna(), 'Глубина']*args[0]+args[1]

    dfv = df.query('Монтмориллонит > 0')
    def mop(values_x,a,b):
        return a * values_x + b 
    values_x = dfv['Глубина']
    values_y = dfv['Монтмориллонит']
    args_1, covar = curve_fit(mop, values_x, values_y)
    df.loc[df['Монтмориллонит'].isna(), 'Монтмориллонит'] = df.loc[df['Монтмориллонит'].isna(), 'Глубина']*args_1[0]+args_1[1]

    dfv = df.query('Песок > 0')
    def mop(values_x,a,b):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Песок']
    args_2, covar = curve_fit(mop, values_x, values_y)
    df.loc[df['Песок'].isna(), 'Песок'] = df.loc[df['Песок'].isna(), 'Монтмориллонит']*args_2[0]+args_2[1]

    dfv = df.query('КОЕ > 0')
    def mop(values_x,a,b):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['КОЕ']
    args_3, covar = curve_fit(mop, values_x, values_y)
    df.loc[df['КОЕ'].isna(), 'КОЕ'] = df.loc[df['КОЕ'].isna(), 'Монтмориллонит']*args_3[0]+args_3[1]

    dfv = df.query('Влажность > 0')
    dfv['Монтмориллонит'].corr(dfv['Влажность'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Влажность']
    args, covar = curve_fit(mop, values_x, values_y)
    df.loc[df['Влажность'].isna(), 'Влажность'] = df.loc[df['Влажность'].isna(), 'Монтмориллонит']*args[0]+args[1]
    
    dfv = df.query('Индекс > 0')
    dfv['Монтмориллонит'].corr(dfv['Индекс'])
    def mop(values_x,a,b,c,d):
        return a * values_x + b 
    values_x = dfv['Монтмориллонит']
    values_y = dfv['Индекс']
    args, covar = curve_fit(mop, values_x, values_y)
    df.loc[df['Индекс'].isna(), 'Индекс'] = df.loc[dff['Индекс'].isna(), 'Монтмориллонит']*args[0]+args[1]

    dfg = df.copy()    

    if list(num444.get().split()):
        df = df.loc[(df['X'] >= Xmin) & (df['X'] <= Xmax) & (df['Y'] >= Ymin) & (df['Y'] <= Ymax)]

    num_columns = ['Глубина', 'Влажность', 'Песок', 'Индекс', 'Электропроводность', 'КОЕ', 'Монтмориллонит', 'X', 'Y']
    num_columns_1 = ['Глубина', 'X', 'Y']
    RANDOM_STATE = 42
    
    y = df[t[-1]]
    X = df.drop([t[-1]], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = RANDOM_STATE)
    data_preprocessor = ColumnTransformer(
    [
        ('num', MinMaxScaler(), num_columns_1)
    ], 
    remainder='passthrough'
    )
    X_train_p = pd.DataFrame(
    data_preprocessor.fit_transform(X_train),
    columns=data_preprocessor.get_feature_names_out()
    )

    X_test_p = pd.DataFrame(
    data_preprocessor.transform(X_test),
    columns=data_preprocessor.get_feature_names_out()    
    )
    pipe_final = Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('models', DecisionTreeRegressor(random_state=RANDOM_STATE))
    ]
    )
    param_grid = [
    {
        'models': [DecisionTreeRegressor(random_state=RANDOM_STATE)],
        'models__max_depth': range(2,40),
        'models__max_features': range(2,40),
        'models__min_samples_leaf': range(1, 20),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },
    {
        'models': [RandomForestRegressor(random_state=RANDOM_STATE)],
        'models__max_depth': range(2,40),
        'models__max_features': range(2,40),
        'models__min_samples_leaf': range(1, 20),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },
    {
        'models': [GradientBoostingRegressor(random_state=RANDOM_STATE)],
        'models__max_depth': range(2,40),
        'models__max_features': range(2,40),
        'models__min_samples_leaf': range(1, 20),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    }
    ]      
    randomized_search = RandomizedSearchCV(
    pipe_final, 
    param_grid, 
    cv=5,
    scoring='r2',
    random_state=RANDOM_STATE,
    n_jobs=-1
    )

    print(randomized_search.best_estimator_)
    
    randomized_search.fit(X_train, y_train)
    score_p = randomized_search.best_score_
    y_test_pred =  randomized_search.predict(X_test)
    print(r2_score(y_test, y_test_pred))

    t_1 = ['Глубина', 'X', 'Y', 'Hor']
    df_h = df[t_1]

    t = ['Глубина', 'X', 'Y', 'Монтмориллонит']
    df = df[t]


    df_u = df.copy()

    if list(num111.get().split()):
        seven = list(num111.get().split())
        m = [int(item) for item in seven]
        for i in range(1,8):
            if i == m[0]:
                df = df.drop(['Монтмориллонит'], axis = 1)
                hhh = tt[i-1]
                df = pd.concat([df, dfg[[hhh]]], axis = 1)
                t = df.columns

        
        t = list(df.columns)
        
    def table(u, Xmin, Xmax, Ymin, Ymax):
        global df
        t = df.columns
        df = df.drop([t[-1]], axis = 1)
        hh = tt[u-1]
        df = pd.concat([df, dfg[[hh]]], axis = 1)
        t = df.columns
        df = df.loc[(df['X'] >= Xmin) & (df['X'] <= Xmax) & (df['Y'] >= Ymin) & (df['Y'] <= Ymax)]
        y = df[t[-1]]
        X = df.drop([t[-1]], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
        scaler = StandardScaler()
        X_train_st = scaler.fit_transform(X_train)
        X_test_st = scaler.transform(X_test)

        models = [
         [Lasso(), 'Линейная регрессия Lasso'],
         [Ridge(), 'Линейная регрессия Ridge'],
         [RandomForestRegressor(n_estimators = 200, random_state = 42), 'Случайный лес'],
         [GradientBoostingRegressor(n_estimators = 200, random_state = 42), 'Градиентный бустинг'],
         [DecisionTreeRegressor(random_state = 42), 'Дерево решений']
        ]

        def prediction(mod, X_train, y_train, X_test, y_test, name):
            model = mod
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics(y_test, y_pred, name)
        p = []
        w = []
        S = (Xmax - Xmin)*(Ymax - Ymin)
        for i in models:
            model = i[0]
            model.fit(X_train_st, y_train)
            y_pred = model.predict(X_test_st)
            p.append(r2_score(y_test,y_pred))
            w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)

        if list(num444.get().split()):
            X_n = [Xmin+20*i for i in range(0,round((Xmax - Xmin)/20))]
            Y_n = [Ymin+20*i for i in range(0,round((Ymax - Ymin)/20))]

            S_1 = S/(len(X_n)*len(Y_n))
        else:
            X_n = [Xmin+20*i for i in range(0,round((Xmax - Xmin)/20))]
            Y_n = [Ymin+20*i for i in range(0,round((Ymax - Ymin)/20))]

            S_1 = S/(len(X_n)*len(Y_n))
        
        Глубина_n = [0+1*i for i in range(0,36)]

        Y_N = []
        for i in range(len(X_n)):
            Y_N.append(Y_n)
        X_N = []
        for i in X_n:
            for j in range(len(Y_n)):
                X_N.append(i)
        Y_NN = []
        for i in Y_N:
            Y_NN += i
        X_NNN = []
        for i in X_N:
            for j in range(len(Глубина_n)):
                X_NNN.append(i)
        Y_NNN = []
        for i in Y_NN:
            for j in range(len(Глубина_n)):
                Y_NNN.append(i)
        H = []
        for i in range(len(Y_NN)):
            H += Глубина_n
        X_F = pd.DataFrame({'Глубина':H, 'X':X_NNN, 'Y':Y_NNN})
        X_F_st = scaler.transform(X_F)
        for i in range(len(p)):
            if p[i] == max(p):
                model = models[i][0]
                model.fit(X_train_st, y_train)
                y_pred_f = model.predict(X_F_st)
        X_F[t[-1]] = y_pred_f
        return X_F, S_1, max(p)

    if list(num444.get().split()):
        X_F_1, S_1, score_2 = table(2, Xmin, Xmax, Ymin, Ymax)
        t11 = X_F_1.columns
    
        X_F_2, S_1, score_3 = table(3, Xmin, Xmax, Ymin, Ymax)
        t22 = X_F_2.columns    
    
        X_F_3, S_1, score_4 = table(4, Xmin, Xmax, Ymin, Ymax)
        t33 = X_F_3.columns    
    
        X_F_4, S_1, score_5 = table(5, Xmin, Xmax, Ymin, Ymax)
        t44 = X_F_4.columns       
    
        X_F_5, S_1, score_7 = table(7, Xmin, Xmax, Ymin, Ymax)
        t55 = X_F_5.columns     
    
        X_F_6, S_1, score_6 = table(6, Xmin, Xmax, Ymin, Ymax)
        t66 = X_F_6.columns

        df_h = df_h.loc[(df_h['X'] >= Xmin) & (df_h['X'] <= Xmax) & (df_h['Y'] >= Ymin) & (df_h['Y'] <= Ymax)]
    else:
        X_F_1, S_1, score_2 = table(2, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
        t11 = X_F_1.columns
        
        X_F_2, S_1, score_3 = table(3, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
        t22 = X_F_2.columns    
        
        X_F_3, S_1, score_4 = table(4, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
        t33 = X_F_3.columns    
        
        X_F_4, S_1, score_5 = table(5, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
        t44 = X_F_4.columns       
        
        X_F_5, S_1, score_7 = table(7, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
        t55 = X_F_5.columns     
        
        X_F_6, S_1, score_6 = table(6, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
        t66 = X_F_6.columns


    t = ['Глубина', 'X', 'Y', 'Монтмориллонит']
    df = dff[t]


    df_u = df.copy()

    if list(num111.get().split()):
        seven = list(num111.get().split())
        m = [int(item) for item in seven]
        for i in range(1,8):
            if i == m[0]:
                df = df.drop(['Монтмориллонит'], axis = 1)
                hhh = tt[i-1]
                df = pd.concat([df, dfg[[hhh]]], axis = 1)
                t = df.columns

        t = list(df.columns)
    
    X_F = X_F_1.copy() 
    X_F['Песок'] = X_F_2[t22[-1]]
    X_F['Индекс'] = X_F_3[t33[-1]]
    X_F['Электропроводность'] = X_F_4[t44[-1]]
    X_F['КОЕ'] = X_F_6[t66[-1]]
    X_F['Монтмориллонит'] = X_F_5[t55[-1]]

    y_pred =  randomized_search.predict(X_F)
    X_F['Horizont'] = [round(i,0) for i in list(y_pred)]

    t_x = X_F.columns
    if list(num0.get().split()):
        depth = list(num0.get().split())
        depth = [int(item) for item in depth]
        df_h = df_h.loc[df_h['Hor'].isin(depth)] 
        X_F_ = X_F.loc[X_F['Horizont'].isin(depth)]
    else:
        X_F_ = X_F.copy()

    if list(num111.get().split()):
    
        if list(num1.get().split()):
            vl = list(num1.get().split())
            vl = [float(item) for item in vl]
        else:
            vl = [min(X_F['Влажность']), max(X_F['Влажность'])] 
        if list(num2.get().split()):
            pes = list(num2.get().split())
            pes = [float(item) for item in pes]
        else:
            pes = [min(X_F['Песок']), max(X_F['Песок'])]   
        if list(num3.get().split()):
            ind = list(num3.get().split())
            ind = [float(item) for item in ind]
        else:
            ind = [min(X_F['Индекс']), max(X_F['Индекс'])]   
        if list(num4.get().split()):
            el = list(num4.get().split())
            el = [float(item) for item in el]
        else:
            el = [min(X_F['Электропроводность']), max(X_F['Электропроводность'])]   
        if list(num11.get().split()):
            mont = list(num11.get().split())
            mont = [float(item) for item in mont]
        else:
            mont = [min(X_F['Монтмориллонит']), max(X_F['Монтмориллонит'])]
        if list(num10.get().split()):
            koe = list(num10.get().split())
            koe = [float(item) for item in koe]
        else:
            koe = [min(X_F['КОЕ']), max(X_F['КОЕ'])]

        X_F_ = X_F_.loc[(X_F_['Влажность'] <= vl[1]) & (X_F_['Влажность'] >= vl[0])]
        X_F_ = X_F_.loc[(X_F_['Песок'] <= pes[1]) & (X_F_['Песок'] >= pes[0])]
        X_F_ = X_F_.loc[(X_F_['Индекс'] <= ind[1]) & (X_F_['Индекс'] >= ind[0])]
        X_F_ = X_F_.loc[(X_F_['Электропроводность'] <= el[1]) & (X_F_['Электропроводность'] >= el[0])]
        X_F_ = X_F_.loc[(X_F_['КОЕ'] <= koe[1]) & (X_F_['КОЕ'] >= koe[0])]
        X_F_ = X_F_.loc[(X_F_['Монтмориллонит'] <= mont[1]) & (X_F_['Монтмориллонит'] >= mont[0])]

        if len(m) > 1:
            g = m.copy()
            g.pop(0)
            X_F_ = X_F_.loc[X_F_['Horizont'].isin(g)]

        X_F_x = X_F_[['X', 'Y', 'Глубина', tt[m[0]-1]]]
        t_1 = X_F_x.columns
        

    else:
        res26.configure(text = "r2_score: %s" % round(score_p,2))
        if list(num0.get().split()):
            depth = list(num0.get().split())
            depth = [int(item) for item in depth]
            df_h = df_h.loc[df_h['Hor'].isin(depth)] 
            X_F_ = X_F.loc[X_F['Horizont'].isin(depth)]
        else:
            X_F_ = X_F.copy()
    if list(num111.get().split()):
        if len(m) > 1:
            g = m.copy()
            g.pop(0)
            dfff = dff.loc[dff['Hor'].isin(g)]
            df = dfff[['Глубина', 'X', 'Y', tt[m[0]-1]]]
        else:
            dfff = dff.copy()
        if list(num0.get().split()):
            depth = list(num0.get().split())
            depth = [int(item) for item in depth]
            dfff = dfff.loc[dfff['Hor'].isin(depth)]
            df = dfff[['Глубина', 'X', 'Y', tt[m[0]-1]]]
        
        if m[0] == 7:
            res26.configure(text = "r2_score: %s" % round(score_7,2))
            if list(num11.get().split()):
                mont = list(num11.get().split())
                mont = [float(item) for item in mont]
                df_ = df.loc[(df[t[-1]] <= mont[1]) & (df[t[-1]] >= mont[0])]
            else:
                df_ = df.copy()
        if m[0] == 6:
            res26.configure(text = "r2_score: %s" % round(score_6,2))
            if list(num10.get().split()):
                koe = list(num10.get().split())
                koe = [float(item) for item in koe]
                df_ = df.loc[(df[t[-1]] <= koe[1]) & (df[t[-1]] >= koe[0])]
            else:
                df_ = df.copy()
        if m[0] == 5:
            res26.configure(text = "r2_score: %s" % round(score_5,2))
            if list(num4.get().split()):
                el = list(num4.get().split())
                el = [float(item) for item in el]
                df_ = df.loc[(df[t[-1]] <= el[1]) & (df[t[-1]] >= el[0])]
            else:
                df_ = df.copy()
        if m[0] == 4:
            res26.configure(text = "r2_score: %s" % round(score_4,2))
            if list(num3.get().split()):
                ind = list(num3.get().split())
                ind = [float(item) for item in ind]
                df_ = df.loc[(df[t[-1]] <= ind[1]) & (df[t[-1]] >= ind[0])]
            else:
                df_ = df.copy()    
        if m[0] == 3:
            res26.configure(text = "r2_score: %s" % round(score_3,2))
            if list(num2.get().split()):
                pes = list(num2.get().split())
                pes = [float(item) for item in pes]
                df_ = df.loc[(df[t[-1]] <= pes[1]) & (df[t[-1]] >= pes[0])]
            else:
                df_ = df.copy()
        if m[0] == 2:
            res26.configure(text = "r2_score: %s" % round(score_2,2))
            if list(num1.get().split()):
                vl = list(num1.get().split())
                vl = [float(item) for item in vl]
                df_ = df.loc[(df[t[-1]] <= vl[1]) & (df[t[-1]] >= vl[0])]
            else:
                df_ = df.copy()
    else:
        df_ = df.copy()
    
    if list(num444.get().split()):
        df_ = df_.loc[(df_['X'] >= Xmin) & (df_['X'] <= Xmax) & (df_['Y'] >= Ymin) & (df_['Y'] <= Ymax)]
        
    if list(num111.get().split()):
        x = df_['X']
        y = df_['Y']
        z = df_['Глубина']
        
        fig = plt.figure(figsize = (12, 7))
        #ax = plt.axes(projection ="3d")
        ax = fig.add_subplot(111, projection='3d')

        color_map = plt.get_cmap('spring')
        scatter_plot = ax.scatter3D(x, y, z, c = df_[t[-1]], cmap = color_map)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Глубина')
        plt.colorbar(scatter_plot)
        if list(num0.get().split()):
            plt.title('3d-диаграмма фактического распределения фактора {}, горизонты: {}'.format(t[-1], depth))
        elif list(num111.get().split()):
            if len(m) > 1:
                plt.title('3d-диаграмма фактического распределения фактора {}, горизонты: {}'.format(t[-1], g))
            else:
                plt.title('3d-диаграмма фактического распределения фактора {}'.format(t[-1]))
        else:
            plt.title('3d-диаграмма фактического распределения фактора {}'.format(t[-1]))
        plt.show()


        x = X_F_x['X']
        y = X_F_x['Y']
        z = X_F_x['Глубина']

        fig = plt.figure(figsize = (12, 7))
        #ax = plt.axes(projection ="3d")
        ax = fig.add_subplot(111, projection='3d')

        color_map = plt.get_cmap('spring')
        scatter_plot = ax.scatter3D(x, y, z, c = X_F_x[t_1[-1]], cmap = 'coolwarm')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Глубина')
        plt.colorbar(scatter_plot)
        if list(num0.get().split()):
            plt.title('3d-модель распределения фактора {}, горизонты: {}'.format(t_1[-1], str(depth)))
        elif list(num111.get().split()):
            if len(m) > 1:
                plt.title('3d-модель распределения фактора {}, горизонты: {}'.format(t[-1], g))
            else:
                plt.title('3d-модель распределения фактора {}'.format(t_1[-1]))
        else:
            plt.title('3d-модель распределения фактора {}'.format(t_1[-1]))
        plt.show()

        res32.configure(text = "Запасы сырья, м3: %s" % round(S_1*len(X_F_x),1))
        
    else:
        x = df_h['X']
        y = df_h['Y']
        z = df_h['Глубина']

        fig = plt.figure(figsize = (12, 7))
        #ax = plt.axes(projection ="3d")
        ax = fig.add_subplot(111, projection='3d')

        color_map = plt.get_cmap('spring')
        scatter_plot = ax.scatter3D(x, y, z, c = df_h['Hor'], cmap = 'coolwarm')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Глубина')
        plt.colorbar(scatter_plot)
        if list(num0.get().split()):
            plt.title('3d-диаграмма фактического распределения горизонтов {}'.format(str(depth)))
        elif list(num111.get().split()):
            if len(m) > 1:
                plt.title('3d-диаграмма фактического распределения горизонтов {}'.format(g))
            else:
                plt.title('3d-диаграмма фактического распределения горизонтов')
        else:
            plt.title('3d-диаграмма фактического распределения горизонтов')  
        plt.show()


        x = X_F_['X']
        y = X_F_['Y']
        z = X_F_['Глубина']

        fig = plt.figure(figsize = (12, 7))
        #ax = plt.axes(projection ="3d")
        ax = fig.add_subplot(111, projection='3d')

        color_map = plt.get_cmap('spring')
        scatter_plot = ax.scatter3D(x, y, z, c = X_F_['Horizont'], cmap = 'coolwarm')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Глубина')
        plt.colorbar(scatter_plot)         
        if list(num0.get().split()):
            plt.title('3d-модель распределения горизонтов {}'.format(depth))
        elif list(num111.get().split()):
            if len(m) > 1:
                plt.title('3d-модель распределения фактора {}'.format(g))
            else:
                plt.title('3d-модель распределения горизонтов')
        else:
            plt.title('3d-модель распределения горизонтов')
        plt.show()

        res32.configure(text = "Запасы сырья, м3: %s" % round(S_1*len(X_F_),1))
    
    df = df_r
##    except:
##        resf.configure(text = "Ошибка загрузки данных")
##    finally:
##        df = df_r        
       
def dim():
    import cv2
    import imageio
    global at
    global bt
    
    class DrawLineWidget(object):
    
        def __init__(self):
            self.original_image = imageio.imread('C://Запад//Запад.jpg')
            self.clone = self.original_image.copy()

            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.extract_coordinates)

                # List to store start/end points
            self.image_coordinates = []

        def extract_coordinates(self, event, x, y, flags, parameters):
            # Record starting (x,y) coordinates on left mouse button click
            if event == cv2.EVENT_LBUTTONDOWN:
                self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
            elif event == cv2.EVENT_LBUTTONUP:
                self.image_coordinates.append((x,y))
                at.append(self.image_coordinates[0])
                bt.append(self.image_coordinates[1])
                print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

            # Draw line
                cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 8)
                cv2.imshow("image", self.clone) 

        # Clear drawing boxes on right mouse button click
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.clone = self.original_image.copy()

        def show_image(self):
            return self.clone

    if __name__ == '__main__':
        draw_line_widget = DrawLineWidget()
        at = []
        bt = []
        t = 0
        while True:
            cv2.imshow('image', draw_line_widget.show_image())
            key = cv2.waitKey(1)

        # Close program with keyboard 'q'
            if key == ord('q'):
                
                
             #   cv2.destroyAllWindows()
             #   exit(1)
                break
    def dx():
        try:
            
            global df
                    
        ##            Xmin = scale_widget_1.get() 
        ##            Xmax = scale_widget_2.get() 
        ##            Ymin = scale_widget_3.get() 
        ##            Ymax = scale_widget_4.get()

            
            at_ = [at[0], at[1], at[2], at[3]]
            bt_ = [bt[0], bt[1], bt[2], bt[3]]
            ab = at_ + bt_
            x_ = []
            y_ = []
            for i in range(len(ab)):
                x_.append(5267188.06 + 1.41*ab[i][1])
                y_.append(14715755.24 + 1.426*ab[i][0])
            Xmin = min(x_) 
            Xmax = max(x_)
            Ymin = min(y_) 
            Ymax = max(y_)

                    
            Xt = 5267188.06 + 1.41*bt[-1][1]
            Yt = 14715755.24 + 1.426*bt[-1][0]

            print(Xmin)
            print(Xmax)
            print(Ymin)
            print(Ymax)
            print(Xt)
            print(Yt)
            res30.configure(text = "Xmin-Xmax-Ymin-Ymax: %s - %s - %s - %s " % (round(Xmin,1), round(Xmax,1), round(Ymin,1), round(Ymax,1)))
            res31.configure(text = "Координаты точки X-Y: %s - %s" % (round(Xt,1), round(Yt,1)))     
                    
            def sd(x):
                x1 = []
                for i in x:
                    i = str(i)
                    x1.append(i.replace(",", "."))
                return x1
            t = list(df.columns)
                    
            a = list(df[t[0]])
            a = sd(a)
            a = [float(item) for item in a]
            #a = [x for x in a if str(x) != 'nan']
            b = list(df[t[1]])
            b = sd(b)
            b = [float(item) for item in b]
            c = list(df[t[2]])
            c = sd(c)
            c = [float(item) for item in c]
            d = list(df[t[3]])
            d = sd(d)
            d = [float(item) for item in d]
            e = list(df[t[4]])
            e = sd(e)
            e = [float(item) for item in e]
            f = list(df[t[5]])
            f = sd(f)
            f = [float(item) for item in f]
            g = list(df[t[6]])
            g = sd(g)
            g = [float(item) for item in g]
            xx = list(df[t[7]])
            xx = sd(xx)
            xx = [float(item) for item in xx]
            yy = list(df[t[8]])
            yy = sd(yy)
            yy = [float(item) for item in yy]

            df_r = df.copy()
            df = pd.DataFrame({t[0]:a, t[1]:b, t[2]:c, t[3]:d, t[4]:e, t[5]:f, t[6]:g, t[7]:xx, t[8]:yy})
                        

            one = list(num1.get().split())
            aa = [float(item) for item in one]
            two = list(num2.get().split())
            bb = [float(item) for item in two]
            three = list(num3.get().split())
            cc = [float(item) for item in three]
            four = list(num4.get().split())
            dd = [float(item) for item in four]
            five = list(num10.get().split())
            ee = [float(item) for item in five]
            six = list(num11.get().split())
            ff = [float(item) for item in six]
            zer = list(num0.get().split())
            zz = [float(item) for item in zer]
                ##    xxx = list(num333.get().split())
                ##    cross = [float(item) for item in xxx]

                ##    dfx = pd.DataFrame({t[0]:zz, t[1]:aa, t[2]:bb, t[3]:cc, t[4]:dd, t[5]:ee, t[6]:ff, t[7]:Xt, t[8]:Yt})
                ##    dx = dfx.copy()
            dff = df.copy()

            dfv = df.query('Электропроводность > 0')
            dfv['Глубина'].corr(dfv['Электропроводность'])
            def mop(values_x,a,b):
                return a * values_x + b 
            values_x = dfv['Глубина']
            values_y = dfv['Электропроводность']
            args, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['Электропроводность'].isna(), 'Электропроводность'] = df.loc[df['Электропроводность'].isna(), 'Глубина']*args[0]+args[1]

            dfv = df.query('Монтмориллонит > 0')
            def mop(values_x,a,b):
                return a * values_x + b 
            values_x = dfv['Глубина']
            values_y = dfv['Монтмориллонит']
            args_1, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['Монтмориллонит'].isna(), 'Монтмориллонит'] = df.loc[df['Монтмориллонит'].isna(), 'Глубина']*args_1[0]+args_1[1]

            dfv = df.query('Песок > 0')
            def mop(values_x,a,b):
                return a * values_x + b 
            values_x = dfv['Монтмориллонит']
            values_y = dfv['Песок']
            args_2, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['Песок'].isna(), 'Песок'] = df.loc[df['Песок'].isna(), 'Монтмориллонит']*args_2[0]+args_2[1]

            dfv = df.query('КОЕ > 0')
            def mop(values_x,a,b):
                return a * values_x + b 
            values_x = dfv['Монтмориллонит']
            values_y = dfv['КОЕ']
            args_3, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['КОЕ'].isna(), 'КОЕ'] = df.loc[df['КОЕ'].isna(), 'Монтмориллонит']*args_3[0]+args_3[1]

            dfv = df.query('Влажность > 0')
            dfv['Монтмориллонит'].corr(dfv['Влажность'])
            def mop(values_x,a,b,c,d):
                return a * values_x + b 
            values_x = dfv['Монтмориллонит']
            values_y = dfv['Влажность']
            args, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['Влажность'].isna(), 'Влажность'] = df.loc[df['Влажность'].isna(), 'Монтмориллонит']*args[0]+args[1]
                    
            dfv = df.query('Индекс > 0')
            dfv['Монтмориллонит'].corr(dfv['Индекс'])
            def mop(values_x,a,b,c,d):
                return a * values_x + b 
            values_x = dfv['Монтмориллонит']
            values_y = dfv['Индекс']
            args, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['Индекс'].isna(), 'Индекс'] = df.loc[dff['Индекс'].isna(), 'Монтмориллонит']*args[0]+args[1]

            df = df.loc[(df['X'] >= Xmin) & (df['X'] <= Xmax) & (df['Y'] >= Ymin) & (df['Y'] <= Ymax)]
            tt = t.copy()
            dfg = df.copy()

            t = ['Глубина', 'X', 'Y', 'Монтмориллонит']
            df = df[t]

            if list(num111.get().split()):
                seven = list(num111.get().split())
                m = [float(item) for item in seven]
                for i in range(1,8):
                    if i == m[0]:
                        df = df.drop(['Монтмориллонит'], axis = 1)
                        hh = tt[i-1]
                        df = pd.concat([df, dfg[[hh]]], axis = 1)
                        print(df)
                    
            f = list(df.columns)          
            y = df[f[-1]]
            X = df.drop([f[-1]], axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
            scaler = StandardScaler()
            X_train_st = scaler.fit_transform(X_train)
            X_test_st = scaler.transform(X_test)

            models = [
                [Lasso(), 'Линейная регрессия Lasso'],
                [Ridge(), 'Линейная регрессия Ridge'],
                [RandomForestRegressor(n_estimators = 200, random_state = 0), 'Случайный лес'],
                [GradientBoostingRegressor(n_estimators = 200, random_state = 0), 'Градиентный бустинг'],
                [DecisionTreeRegressor(random_state = 0), 'Дерево решений']
                ]

            def metrics(y_true, y_pred, title):
                print(title)
                print('MAE: {:.2f}'.format(mean_absolute_error(y_true,y_pred)))
                print('MSE: {:.2f}'.format(mean_squared_error(y_true,y_pred)))
                print('R2: {:.2f}'.format(r2_score(y_true,y_pred)))

            def prediction(mod, X_train, y_train, X_test, y_test, name):
                model = mod
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics(y_test, y_pred, name)

            p = []
            w = []
            for i in models:
                model = i[0]
                model.fit(X_train_st, y_train)
                y_pred = model.predict(X_test_st)
                p.append(r2_score(y_test,y_pred))
                w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)
            if max(p) == p[0]:
                print('Максимальная метрика R2-CORE у регрессии LASSO: ', p[0])
            elif max(p) == p[1]:
                print('Максимальная метрика R2-CORE у регрессии Ridge: ', p[1])
            elif max(p) == p[2]:
                print('Максимальная метрика R2-CORE у регрессии RandomForest: ', p[2])
            elif max(p) == p[3]:
                print('Максимальная метрика R2-CORE у регрессии GradientBoosting: ', p[3])  
            elif max(p) == p[4]:
                print('Максимальная метрика R2-CORE у регрессии TreeDecision: ', p[4])      
            if min(w) == w[0]:
                print('Минимальная средняя метрика у регрессии LASSO: ', w[0])
            elif min(w) == w[1]:
                print('Минимальная средняя метрика у регрессии Ridge: ', w[1])
            elif min(w) == w[2]:
                print('Минимальная средняя метрика у регрессии RandomForest: ', w[2])
            elif min(w) == w[3]:
                print('Минимальная средняя метрика у : регрессии GradientBoosting', w[3])
            elif min(w) == w[4]:
                print('Минимальная средняя метрика у : регрессии TreeDecision', w[4])

            Глубина_p = [0+1*i for i in range(36)]
            X_p = Xt
            Y_p = Yt
            X_P = [X_p for i in range(len(Глубина_p))]
            Y_P = [Y_p for i in range(len(Глубина_p))]
            X_PP = pd.DataFrame({'Глубина':Глубина_p, 'X':X_P, 'Y':Y_P})
            X_P_st = scaler.transform(X_PP)
            for i in range(len(p)):
                if p[i] == max(p):
                    model = models[i][0]
                        #model.fit(X_train_st, y_train)
                    y_pred_p = model.predict(X_P_st)
            X_PP[f[-1]] = y_pred_p
            if list(num444.get().split()):
                q = list(num444.get().split())
                Q = [float(item) for item in q]
                X_t = pd.DataFrame({'Глубина':Q[0], 'X':X_P, 'Y':Y_P})
                X_t_st = scaler.transform(X_t)
                    
                for i in range(len(p)):
                    if p[i] == max(p):
                        model = models[i][0]
                            # model.fit(X_train_st, y_train)
                        y_pred_t = model.predict(X_t_st)
                        res28.configure(text = "%s : %s" % (f[-1], round(y_pred_t[0],1)))                

            colorlist = ["darkorange", "gold", "lawngreen", "lightseagreen"]
            newcmp = LinearSegmentedColormap.from_list("testCmap", colors=colorlist, N=256)

            x = X_PP['X']
            y = X_PP['Y']
            z = X_PP['Глубина']

            fig = plt.figure(figsize = (12, 7))
                #ax = plt.axes(projection ="3d")
            ax = fig.add_subplot(111, projection='3d')

            color_map = plt.get_cmap('spring')
            scatter_plot = ax.scatter3D(x, y, z, c = X_PP[f[-1]], cmap = 'coolwarm')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Глубина')
            plt.colorbar(scatter_plot)
            plt.title('3d-модель точечного распределения фактора {}'.format(f[-1]))
            plt.show()
                    
            if (Xt > Xmax) or (Xt < Xmin) or (Yt > Ymax) or (Yt < Ymin):
                resf.configure(text = "Выбранная точка вне модельного поля")
            else:
                resf.configure(text = "  ")

            df = df_r

        except:
            resf.configure(text = "Ошибка ввода. Повторите загрузку данных")
        finally:
            df = df_r

    button9 = tk.Button(win, text = "forecast", bg="green", fg="white", width=20, command = dx)
    button9.grid(row = 31, column = 9 )




button1 = tk.Button(win, text="Browse a File", width=20, command=File_dialog)
button1.grid(row = 0, column = 9 )


button2 = tk.Button(win, text="Load File", width=20, command=Load_excel_data)
button2.grid(row = 4, column = 9 )
    
button3 = tk.Button(win, text = "Построить модель", bg="blue", fg="white", width=20, command = df)
button3.grid(row = 12, column = 9 )

button8 = tk.Button(win, text = "Распределение по горизонтам", bg="black", fg="white", width=25, command = ddf)
button8.grid(row = 14, column = 9 )

button4 = tk.Button(win, text = "Расчет цели", bg="green", fg="white", width=20, command = dff)
button4.grid(row = 20, column = 9 )

button5 = tk.Button(win, text = "Построить кластеры", bg="yellow", fg="black", width=20, command = dfff)
button5.grid(row = 16, column = 9 )

button6 = tk.Button(win, text = "3d-graph", bg="brown", fg="white", width=20, command = dffff)
button6.grid(row = 23, column = 9 )




button7 = tk.Button(win, text = "image", bg="black", fg="white", width=20, command = dim)
button7.grid(row = 25, column = 9 )





win.mainloop()

ророл








