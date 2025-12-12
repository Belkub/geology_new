import streamlit as st
import numpy as np
import math
import pandas as pd

import pickle
#from pydub import AudioSegment
import cloudpickle
from datetime import datetime
import openpyxl as xl


import speech_recognition as sr

try:

    def voice():
        def speech_to_text(audio_file):
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio, language="ru-RU")
                    return text
                except sr.UnknownValueError:
                    return "Could not understand audio"
                except sr.RequestError as e:
                    return f"Error: {str(e)}"
        recognizer = sr.Recognizer()   
        audio_value = st.audio_input("Record a voice message")
        if audio_value:
            st.audio(audio_value)
            text = speech_to_text(audio_value)
            return text

    wb=xl.Workbook()
    st.header("chart_mobile/processing")

    df_c = pd.read_excel('bent.xlsx')
    df_f = pd.read_excel('bent_1.xlsx')
    #df_f['Дата теста/отправки'] = pd.to_datetime(df_f['Дата теста/отправки'])
    if 'Unnamed: 0' in df_c.columns:
        df_c = df_c.drop('Unnamed: 0', axis = 1)
    if 'Unnamed: 0' in df_f.columns:
        df_f = df_f.drop('Unnamed: 0', axis = 1)
    df_c = df_c.dropna(subset=['Бентонит'])
    df_c = df_c.drop_duplicates()
    tr = df_c['Бентонит'].unique()
    df_f = df_f.loc[df_f['Бентонит'].isin(tr)]
    df_f['ф600'] = df_f['ф600'].astype(float)
    df_f['ф300'] = df_f['ф300'].astype(float)
    df_f['ф600 БС'] = df_f['ф600 БС'].astype(float)
    df_f['ф300 БС'] = df_f['ф300 БС'].astype(float)


    ss = st.sidebar.checkbox('Изменить имя образца/удалить')
    pp = st.sidebar.checkbox('Сформировать сводную таблицу образцов')
    if pp == True and ss == False:
        st.title('Создание сводной таблицы')
        bn = st.checkbox('Вывести таблицу и ограничить число полей')
        if bn:
            number = df_f.columns
            num = st.multiselect('Выбрать значимые поля', list(df_f.columns))
            if num:
                number = num
            bnn = st.checkbox('Вывести сводную таблицу')
            
            if st.button('Показать сводную таблицу'):
                st.dataframe(df_f[number]) 
        else:
            number = list(df_f.columns)
        
        spp = float(st.number_input('Установить число полей группировки: ', min_value = 1, max_value = 3, value = 1, step = 1))
        #new_row_1 = {'Компания':['Компания'],'Имя образца':['Имя образца'],'Дата теста/отправки':['Дата теста/отправки'], 'Производитель/поставщик':['Производитель/поставщик'], 'Тестировщик':['Тестировщик'], 'Отрасль':['Отрасль'], 'Дисперсионная среда':['Дисперсионная среда'], 'Состав системы':['Состав системы'], 'Методика':['Методика'], 'Результат':['Результат'], 'Примечания':['Примечания'], 'Финальное решение':['Финальное решение']}
        new_row_1 = {}
        for i in number:
            new_row_1[i] = i
        p_1 = st.selectbox('Выбрать поле_1: ', list(new_row_1))
        p_11 = st.multiselect(f'Выбрать позиции поля {p_1}', list(df_f[p_1].unique()))
        if p_11:
            df_ff = df_f.loc[df_f[p_1].isin(p_11)]
            if spp == 1:
                if st.button('Показать сводную таблицу'):
                    st.dataframe(df_ff[number]) 
                   
            else:
                del new_row_1[p_1]
                p_2 = st.selectbox('Добавить поле_2: ', list(new_row_1))
                p_22 = st.multiselect(f'Выбрать позиции поля {p_2}', list(df_ff[p_2].unique()))
                if p_22:
                    df_ff = df_ff.loc[df_ff[p_2].isin(p_22)]
                    if spp == 2:
                        if st.button('Показать и записать таблицу в файл C:\Data\table.xlsx'):
                            st.dataframe(df_ff[number]) 
                       
                    else:
                        del new_row_1[p_2]
                        p_3 = st.selectbox('Добавить поле_3: ', list(new_row_1))
                        p_33 = st.multiselect(f'Выбрать позиции поля {p_3}', list(df_ff[p_3].unique()))
                        if p_33:
                            df_ff = df_ff.loc[df_ff[p_3].isin(p_33)]
                            if spp == 3:
                                
                                if st.button('Показать и записать таблицу в файл C:\Data\table.xlsx'):
                                    st.dataframe(df_ff[number]) 
        
    elif ss == True and pp == False:
        a_1 = {}
        for i in list(df_f['Бентонит']):
            a_1[i] = i
        t_2 = st.selectbox('Выбрать бентонит: ', list(a_1))
        df_e1 = df_f.loc[df_f['Бентонит'] == t_2]
        b_1 = {}
        for i in list(df_e1['Имя образца']):
            b_1[i] = i
        e_2 = st.selectbox('Выбрать образец: ', list(b_1))
        num = list(df_e1.loc[df_e1['Имя образца'] == e_2, 'Номер'])[0]
        print(num)
        ss_1 = st.sidebar.checkbox('Изменить имя образца')
        dd = st.sidebar.checkbox('Заполнить/изменить поля')

        if ss_1 == True and dd == False:
            st.title('Изменение имени/удаление образца')
            with open("file_1.txt", "w") as file:  
                file.write(e_2)
                file.close()
            with open("file_1.txt", "r") as file:  
                example_1 = file.read()
                file.close()
            rr_1 = st.checkbox('Голосовой ввод')
            if rr_1:
                text = voice()
                if text:
                    with open("file_1.txt", "w") as file:  
                        file.write(text)
                        file.close()
                    example_1 = text
            example = st.text_input('Название бентонита: ', value = example_1)
            with open("file_1.txt", "w") as file:  
                file.write(example)
                file.close()
            with open("file_1.txt", "r") as file:  
                text = file.read()
                file.close()
            if st.button('Записать новое имя'):
                df_f = df_f.dropna(subset=['Имя образца'])
                df_f = df_f.drop_duplicates(subset=['Бентонит','Имя образца'])
                df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Номер'] == num), 'Имя образца'] = text
                df_f.to_excel('bent_1.xlsx')
            ss_2 = st.checkbox('Удалить образец')
            if ss_2:     
                if st.button('Удалить информацию об образце'):
                    df_f = df_f.loc[df_f['Имя образца'] != e_2]
                    df_f.to_excel('bent_1.xlsx')
    ##            
        elif dd == True and ss_1 == False:
            st.title('Заполнение/изменение полей')
            now = datetime.now()
            new_row_1 = {'КПАВ':['КПАВ'], 'Тип КПАВ':['Тип КПАВ'], 'ММ':['ММ'], 'Эквивалент, г/кг':['Эквивалент, г/кг'], 'Растворитель':['Растворитель'], 'ф600':['ф600'], 'ф300':['ф300'], 'GEL1/10':['GEL1/10'], 'Brookfield':['Brookfield'], 'ф600 БС':['ф600 БС'], 'ф300 БС':['ф300 БС'], 'GEL БС':['GEL БС'], 'Примечания':['Примечания']}
            s_2 = st.selectbox('Выбрать поле: ', list(new_row_1))
            comp = list(df_f['Имя образца'])
            with open("file_1.txt", "w") as file:  
                file.write(str(list(df_e1.loc[df_e1['Имя образца'] == e_2, s_2])[0]))
                file.close()    

            def fr(type):           
                rr_11 = st.checkbox('Голосовой ввод')
                if rr_11:
                    text_22 = voice()
                    if text_22:
                        with open("file_1.txt", "w") as file:  
                            file.write(text_22)
                            file.close()
                with open("file_1.txt", "r") as file:  
                    text_11 = file.read()
                    file.close()
                change = st.text_input('Содержание поля: ', value = text_11)
                ddd = st.checkbox(f'Выбрать содержание поля {s_2}')
                if ddd:
                    t_3 = st.selectbox('Выбрать содержание: ', list(type))
                    change = st.text_input('Новое содержание поля: ', value = t_3)
                return change
            if s_2 in ['Тип КПАВ', 'Растворитель']:
                if s_2 == 'Тип КПАВ':
                    type = {'DA':'DA', 'MA':'MA', 'TA':'TA', 'A-Ar':'A-Ar', 'Ar':'Ar', 'DAr':'DAr', 'Другой':'Другой'}
                    change = fr(type)
                elif s_2 == 'Растворитель':
                    type = {'ДТ':'ДТ', 'Керосин':'Керосин', 'Мин масло':'Мин масло', 'Синтетика':'Синтетика', 'Ксилол':'Ксилол', 'Спирт':'Спирт'}
                    change = fr(type)
                elif s_2 == 'Способ':
                    type = {'Карвил':'Карвил', 'Стакан':'Стакан', 'Суспензия':'Суспензия', 'Другой':'Другой'}
                    change = fr(type)
            
            else:
                with open("file_1.txt", "r") as file:  
                    text_1 = file.read()
                    file.close()
                rr_1 = st.checkbox('Голосовой ввод')
                if rr_1:
                    text_2 = voice()
                    if text_2:
                        with open("file_1.txt", "w") as file:  
                            file.write(text_2)
                            file.close()
                with open("file_1.txt", "r") as file:  
                    text_3 = file.read()
                    file.close()
                change = st.text_input('Содержание поля: ', value = text_3)
            with open("file_1.txt", "w") as file:  
                file.write(change)
                file.close()
            with open("file_1.txt", "r") as file:  
                text_3 = file.read()
                file.close()
            if st.button('Записать поле'):
                if s_2 == 'ф600' or s_2 == 'ф300' or s_2 == 'ф600 БС' or s_2 == 'ф300 БС':
                    text_3 = float(text_3)
                
                df_f = df_f.dropna(subset=['Бентонит', 'Имя образца'])
                df_f = df_f.drop_duplicates(subset=['Бентонит', 'Имя образца'])
                df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), s_2] = text_3
            

                f600 = df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'ф600'][comp.index(e_2)]
                f300 = df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'ф300'][comp.index(e_2)]
                f600_ = df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'ф600 БС'][comp.index(e_2)]
                f300_ = df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'ф300 БС'][comp.index(e_2)]

                if f600 > f300 and f600 > 0 and f300 > 0 and f600_ > f300_ and f600_ > 0 and f300_ > 0:
                    PV = f600 - f300
                    YP = f300 - PV
                    PV_ = f600_ - f300_
                    YP_ = f300_ - PV_
                    df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'PV'] = PV
                    df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'YP'] = YP
                    df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'PV БС'] = PV_
                    df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'YP БС'] = YP_
                    n = 3.32*math.log10((YP_+PV_+PV_)/(YP_+PV_))
                    K = (YP_+PV_)*0.511/(511**n)
                    LSRV = 1000*n*K*(0.3/60)**(n-1)
                    df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'n'] = round(n,2)
                    df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'K'] = round(K,2)
                    df_f.loc[(df_f['Бентонит'] == t_2) & (df_f['Имя образца'] == e_2), 'LSRV'] = round(LSRV,2)

                df_f.to_excel('bent_1.xlsx')
    else:
        st.title('Создание имени образца')
        a = {}
        for i in list(df_c['Бентонит'].unique()):
            a[i] = i
        t_1 = st.selectbox('Выбрать бентонит: ', list(a))
        if t_1 in list(df_f['Бентонит']):
            df_e = df_f.loc[df_f['Бентонит'] == t_1]
            b = {}
            for i in list(df_e['Имя образца']):
                b[i] = i
            e_1 = st.selectbox('Проверить образцы: ', list(b))
        rr = st.checkbox('Голосовой ввод')
        text = 'new example'
        with open("file_1.txt", "w") as file:  
            file.write(text)
            file.close()
        if rr:
            text = voice()
            if text:
                with open("file_1.txt", "w") as file:  
                    file.write(text)
                    file.close()
        with open("file_1.txt", "r") as file:  
            text = file.read()
            file.close()
        print(text)
        example = st.text_input('Имя образца: ', value = text)
        with open("file_1.txt", "w") as file:  
            file.write(example)
            file.close()
        with open("file_1.txt", "r") as file:  
            text = file.read()
            file.close()
        print(text)
        ##wdf_c = st.checkbox('Записать название')
        if st.button('Создать новый образец'):
            now = datetime.now()
            g = 1
            if t_1 in list(df_f['Бентонит']): 
                g = len(df_e['Бентонит']) + 1
            new_row = {'Время':now, 'Имя образца':[text], 'Номер':[g], 'Бентонит': [t_1], 'КПАВ':[''], 'Тип КПАВ':[''], 'ММ':[''], 'Эквивалент, г/кг':[''], 'Растворитель':[''], 'ф600':[''], 'ф300':[''], 'YP':[''], 'PV':[''], 'GEL1/10':[''], 'Brookfield':[''], 'ф600 БС':[''], 'ф300 БС':[''], 'YP БС':[''], 'PV БС':[''], 'n':[''], 'K':[''], 'LSRV':[''], 'GEL БС':[''], 'Примечания':['']}
            df_cc = pd.DataFrame(new_row)
            df_f = pd.concat([df_f, df_cc], axis=0, ignore_index = True)
            df_f = df_f.drop_duplicates(subset=['Бентонит','Имя образца'])
            df_f.dropna(subset=['Имя образца'])
            df_f.to_excel('bent_1.xlsx')
        
except:
    st.error("Ошибка ввода данных. Сделайте шаг назад или очистите кэш")

##       

