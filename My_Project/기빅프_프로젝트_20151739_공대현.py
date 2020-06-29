#!/usr/bin/env python
# coding: utf-8

# # 코로나 확진자 Data를 이용한 빅데이터 프로젝트
# 
# #### 순서: 코로나 맵, 날짜에 따른 나라별 확진자 비교 그래프, 나이 분포(정규분포), 성별(원그래프), 
# 
# ###### 설치해야하는 라이브러리 : folium, IPython,ffmpeg,celluloid

# In[1]:


import time
start = time.time()


# # 1.Patient class 생성
# input  = [ id , reporting_date , summary , location , country , gender , age , visit_wuhan , 
# from_wuhan , death , recovered ]

# In[1]:


class Patient(object):
    def __init__(self,list):
        self.id= list[0]
        self.reporting_date = list[1]
        self.summary = list[2]
        self.location = list[3]
        self.country = list[4]
        self.gender = list[5]
        self.age = list[6]
        self.visit_wuhan = list[7]
        self.from_wuhan = list[8]
        self.death = list[9]
        self.recovered = list[10]
        self.list = list[0:11]
patient_list = []


# # 2. 데이터 오류 수정
# 
# 날짜 데이터 중간중간에 연도 뒷자리 두 숫자와 월 일의 순서가 뒤죽박죽 섞여있고, 다른 데이터와 달리 '/' 대신 '-'으로 split이 되어있는 오류를 확인 -> 수정

# In[2]:


from IPython.display import Image
Image("Date_Data_Error.PNG")


# In[3]:


fp = open("corona_data.txt","rt")
a = fp.readlines()
a[1] ##readlines 한 것 예시,정보가 \t로 나눠져 있다.


# In[4]:


import re
p = re.compile(r"(20)0(\d)\W(\d{2})\W(\d{2})") #수정할 패턴 compile
s = '2002-04-20'
m = p.sub('\g<2>/\g<3>/\g<1>\g<4>',s) 
m  ##ex) 2002-04-20 -> 2/04/2020


# In[5]:


count = 0
print("오류가 있는 날짜 데이터 수정\n")
for i in range(1,len(a)):
    person_info = a[i]
    info_list = person_info.split('\t') ## \t 가 나올때마다, split해서 list로 나눔
    patient_class = Patient(info_list) ## list를 input하여 1085명 patient class 생성
    if '2002' in patient_class.reporting_date:
        s = patient_class.reporting_date
        count = count+1
        m = p.sub('\g<2>/\g<3>/\g<1>\g<4>',patient_class.reporting_date)
        patient_class.reporting_date = m
        patient_class.list[1] = m
        print("count:",count," date: ",s,"->",patient_class.reporting_date)
    elif patient_class.reporting_date == 'NA':
        print(" ")
        print(i,"번째: ", "date: ",'NA',"->",'remove patient from list ')
        print(" ")
        error_num = i
    patient_list.append(patient_class)
del patient_list[error_num-1]
for i in range(error_num-1,len(patient_list)):
    patient_list[i].id = str(int(patient_list[i].id) -1)
    patient_list[i].list[0] = patient_list[i].id


# # 3. Corona Map (2020.1.13~2020.2.27)

# In[6]:


country_list = []

for i in range(0,len(patient_list)):
    country_list.append(patient_list[i].country) ## patient_list의 country 변수만 추출하여 country_list 생성

sorted_country_list = sorted(list(set(country_list))) ## country list 중복 제거, 이름순 정리

country_count= {}
for lst in country_list:      ## country 마다 patient count
    try: country_count[lst]+= 1
    except: country_count[lst]=1    


country_count


# In[7]:


coordinate_list = [(33.9,67.7),(28,1.6),(-25.3,133.7),(48.2,16.4),(26.0,50.5),                  (50.5,4.4),(12.5,104.9),(56.1,-106.3),(35.86,104.19),(45.09,15.2),                  (26.82,30.8),(61.9,25.7),(46.22,2.21),(51.165,10.45),(22.39,114.10),                  (20.593,78.96),(32.427,53.68),(31.04,34.851),(41.871,12.567),(36.20,138.252),                   (29.31,47.481),(33.85,35.86),(4.210,101.97),(28.39,84.12),(12.87,121.77),                   (61.52,105.318),(1.352,103.81),(35.907,127.766),(40.463,-3.74),(7.873,80.771),                   (60.128,18.643),(46.818,8.22),(23.69,120.96),(15.87,100.992),(23.424,53.84),                   (55.378,-3.43),(37.09,-95.7),(14.058,108.27)]
len(coordinate_list) ## 나라별 좌표 list


# In[8]:


country_count_list = []
for country in sorted_country_list:
    country_count_list.append((country,country_count[country])) ## country count의 자료형 dictionary -> list


# In[9]:


import folium
from folium.features import DivIcon

m = folium.Map(coordinate_list[21], zoom_start=2) ## 지도 중심위치: 레바논

for i in range(len(sorted_country_list)):
    population = country_count_list[i][1]  ## 나라별 인구 변수
    coordinate = coordinate_list[i]  ## 나라별 좌표 변수
    
    folium.Circle(location = coordinate, color = 'red',
                  radius = 10000*population,fill=True).add_to(m) ## 인구에 비례하게 원 mark
    folium.map.Marker(
    [coordinate[0], coordinate[1]],
    icon=DivIcon(
        icon_anchor=(0,0),
        html='<div style="font-size: 10pt">%s:%d</div>' %(sorted_country_list[i],(population)), ## text 삽입
        )
    ).add_to(m)
m


# ## 특징: 중국에서 발병했으므로 근처에 있는 아시아 국가의 발병률이 높음, 특이한 것은
# ## 멀리 떨어져 있는 유럽국가에서 발병률이 그 중간지역에 있는 발병률보다 높음.

# # 4. Pandas로 Dataframe 생성

# In[10]:


import pandas as pd


# In[11]:


data = []
columns=['id','reporting_date','summary','location','country',
                               'gender','age','visit_wuhan','from_wuhan','death','recovered']
for i in range(len(patient_list)):
    data.append(patient_list[i].list)
a = pd.DataFrame(data,columns=columns)
print(patient_list[-1].id)
a


# # 5.Days vs Patient Number Plot

# In[12]:


from datetime import datetime
from dateutil.relativedelta import *
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', size = 10)
date_count= {}
for lst in sorted(a["reporting_date"]):      ## country 마다 patient count
    try: date_count[lst]+= 1
    except: date_count[lst]=1    

date = [key for key in date_count]
num = [date_count[key] for key in date_count]

start_date = date[0].split('/')
start_date = datetime(int(start_date[2]),int(start_date[0]),int(start_date[1]))

dateC = []
for i in range(len(date)):
    end_date = date[i].split('/')
    end_date = datetime(int(end_date[2]),int(end_date[0]),int(end_date[1]))
    date_gap = end_date - start_date 
    dateC.append(date_gap.days)
    
numC = []
for i in range(len(num)):
    numC.append(sum(num[0:i+1]))

DvP = pd.Series(numC,index = dateC)
plt.plot(DvP)

plt.title('Days vs Patient number')
plt.xlabel('Days')
plt.ylabel('Accumulative Patient Number')


# ## 6.Age vs Patient Number Plot

# In[13]:


age_count = {}
for lst in (a["age"]):## country 마다 patient count
    if lst == 'NA':
        continue
    elif float(lst) < 1:
        continue
    try: age_count[int(lst)]+= 1
    except: age_count[int(lst)]=1    ## dictionary 자료형인 {나이 : 몇명,나이 : 몇명,나이 : 몇명} 형태로 저장됨.

age = [key for key in age_count]
num = []
age = sorted(age)
for key in age:
    num.append(age_count[key])
plt.rc('font', size = 10)
DvP = pd.Series(num,index = age)
plt.bar(age,num,width = 1)
plt.title('Age vs Patient number')
plt.xlabel('Age')
plt.ylabel('Patient Number')
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.show()


# ### 특징 1 : 50~60대에서 가장 많이 발생함
# ### 특징 2 : N십대 '중반'에서 많이 발생함 ex) 25,35,45,55,65,75,85

# ## 7. 성별에 따른 코로나 바이러스 환자 비율 pie 그래프

# In[14]:


countF = 0
countM = 0

for i in range(len(patient_list)):          ## patient_list안의 1084명의 성별을 누적해서 성별당 총 사람 수 계산
    if patient_list[i].gender == 'male':
        countM += 1
    elif patient_list[i].gender == 'female':
        countF += 1

data = [countM,countF]
label = ['Male','Female']
plt.pie(data,labels=label,colors = ['lightskyblue',  'lightcoral'],counterclock=False,explode = (0, 0)
       , startangle=-80, autopct='%1.1f%%', shadow=True)
plt.show()
    


# ### 특징 : 남자가 걸릴 확률이 15.2% 더 높다는 것을 확인할 수 있다.

# ## 8. 시간의 흐름에 따른 국가별 코로나 환자수 증가 

# In[15]:


date_list = {}
id = 1
for lst in a["reporting_date"]:
    try: 
        date_list[lst].append(a["id"][id-1])
        
    except: 
        date_list[lst] = a["id"][id-1]
        date_list[lst] = [date_list[lst]]
    id = id +1


# In[16]:


import time
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from IPython.display import Video




SDL = sorted(date_list)
count = []
days = 1
fig = plt.figure()
plt.rc('font', size = 5)
plt.xlabel('Number of Patients')
plt.ylabel('Countries Who got COVID19 Patients')
camera = Camera(fig)    
for key in SDL:                 ##key 예시)1/13/2020
    c_count = {}


    
    for lst in date_list[key]:
        lst = int(lst)          ##lst = id
        count.append(a["country"][lst-1])
    for e in count:      
        try: c_count[e]+= 1
        except: c_count[e]=1
    c_count = pd.Series(c_count)   ## for문에서 날짜(key of SDL)별로 { 나라 : 환자 수, 나라 : 환자 수 .. }같이 dictionary 형태로 저장 
    cell = pd.DataFrame(c_count)
    v = cell.values 
    v = np.squeeze(v, axis =1)
    
    
    plt.barh(cell.index,v,color='blue')
    camera.snap()
    plt.title("2020.1.13~2020.2.27")
    days = days +1

animation = camera.animate(interval=250, blit=True) ## interval 단위 ms

animation.save(
    'corona_20200113~20200227.mp4',
    dpi=150,
    savefig_kwargs={
        'frameon': True,
        'pad_inches': 'tight'
    }
)

Video("corona_20200113~20200227.mp4")


# In[18]:


print("time :", time.time() - start)

