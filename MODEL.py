#!/usr/bin/env python
# coding: utf-8

# In[928]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from  sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans




# In[929]:


df=pd.read_csv("OnlineRetail.csv",encoding='ISO-8859-1')


# # Performing EDA

# In[930]:


df.head()


# In[931]:


df.describe()


# In[932]:


df.shape


# In[933]:


df.columns


# In[934]:


# droping unwanted columns
df = df.drop(['Description'],axis =1)


# In[935]:


df.head()


# In[936]:


#spltting the date column from space
df['Date']=df['InvoiceDate'].apply(lambda x : str(x).split(" "))


# In[937]:


df.head()


# In[938]:


df[['date','hour']] =pd.DataFrame(df['Date'].tolist(),index=df.index)


# In[939]:


df['InvoiceDate'] =pd.to_datetime(df['InvoiceDate'])


# In[940]:


df.dtypes


# In[941]:


df.head()


# In[942]:


df.dtypes


# In[943]:


df['month']=pd.DatetimeIndex(df['InvoiceDate']).month
df['year']=pd.DatetimeIndex(df['InvoiceDate']).year
df['day']=pd.DatetimeIndex(df['InvoiceDate']).day


# In[944]:


df.head()


# In[945]:


df['date'] =pd.to_datetime(df['date'])


# In[946]:


pd.DataFrame(df.isnull().sum(),columns=['nullvalue'])


# In[947]:


pd.DataFrame(round(df.isnull().sum()/df.shape[0]*100,3),columns=['nullvalue'])


# In[948]:


#dropping N/A values
df = df.dropna(subset=['CustomerID'])


# In[949]:


df.shape


# In[950]:


df.isnull().sum()


# In[951]:


#Find out the count and distinct count of each column
pd.value_counts(df['InvoiceNo'])


# In[952]:


len(df['InvoiceNo'].unique())


# In[953]:


pd.value_counts(df['month'])


# In[954]:


pd.value_counts(df['Country'])


# In[955]:


#Calculate Revenue per month and show in a data frame and a visual of your choice Write down the inference out of this.
df['Revenue']=df['UnitPrice'] *df['Quantity']


# # Data Vistualization

# In[956]:


df.head()


# In[957]:


revenue_data=df.groupby('month')['Revenue'].sum().reset_index()


# In[958]:


revenue_data


# In[959]:


#visualizing the monthly revenue
plt.figure(figsize=[15,5])
plt.plot(revenue_data['month'],revenue_data['Revenue'])
plt.title('Month-wise revenue')
plt.xlabel('Month')
plt.ylabel('Revenue')


# From the above graph we can see the maxium revenue is on the month of november .From month of april to november we can see the revenue is been increaseing every month 

# In[960]:


#Calculate monthly percent change in growth rate and show in a visual. Note down the inference
revenue_data['monthly_rate']=revenue_data['Revenue'].pct_change()


# In[961]:


revenue_data 


# In[962]:


#removed month december as there is no enough data
revenue_data=revenue_data.loc[revenue_data['month']!=12]


# In[963]:


revenue_data


# In[964]:


revenue_data['monthly_rate']=revenue_data['monthly_rate'].fillna(0)


# In[965]:


revenue_data


# In[966]:


#monthly percent change in growth rate and show in a visual
plt.figure(figsize=[15,5])
plt.plot(revenue_data['month'],revenue_data['monthly_rate'])

plt.title('Growth-rate month-wise')
plt.xlabel('Month')
plt.ylabel('monthly_rate')


# There is increase and decrease in the monthly rate .On the month of apri montl rate was the least comparatively other months

# In[967]:


#Calculate cumulative revenue for across all months and show in a visual.
df['quater']= (df['month']-1)//3 


# In[968]:


df.head()


# In[969]:


df['cumulative_revenue']=df['Revenue'] *df['Revenue']


# In[970]:


df.head()


# In[971]:


pd.crosstab(df['cumulative_revenue'],df['quater']).sort_values([0,1,2,3],ascending=(False,False,False,False)).head(1).plot.bar(title='cumulative revenue for across all months')
plt.show()


# In[972]:


cul_revenue=df.groupby('month')['cumulative_revenue'].sum().reset_index()


# In[973]:


plt.figure(figsize=[15,5])
plt.plot(cul_revenue['month'],cul_revenue['cumulative_revenue'])

plt.title('cumulative_revenue-rate month-wise')
plt.xlabel('Month')
plt.ylabel('cumulative_revenue')


# In[974]:


#Revenue by country


# In[975]:


rev=df.groupby('Country')['Revenue'].nunique().reset_index()


# In[976]:


plt.figure(figsize=[15,5])
rev.groupby('Country')['Revenue'].sum().plot(kind='bar', color='red')
plt.title("country-wise revenue")
plt.ylabel('Revenue')
plt.xlabel('Country')


# As we know from the data united kingdom has the maximum revenue followed by germany

# In[977]:


#Total active customer(unique count of customer id) by country and month


# In[978]:


montly_active=df.groupby('month')['CustomerID'].nunique().reset_index()


# In[979]:


montly_active


# In[980]:


plt.figure(figsize=[15,5])
sns.barplot(x="month", y="CustomerID", data=montly_active,
            label="monthly-activity-track", color="g")


# on the month november there was more active customer 

# In[981]:


#Total orders by country and month
unique_order=df.groupby('month')['InvoiceNo'].nunique().reset_index()


# In[982]:


unique_order


# In[983]:


#order by month
ax=unique_order.groupby('month')['InvoiceNo'].sum().plot(kind ='bar',color ='orange');
plt.title("month-wise order")
plt.rcParams['figure.figsize']=(12,10)
plt.ylabel('orders')
for p in ax.patches:
    height =p.get_height()
    ax.text(x=p.get_x()+(p.get_width()/2),y=height +0.2,ha='center',s='{:.0f}'.format(height))


# In[984]:


df.groupby('Country')['InvoiceNo'].count().plot(kind='bar',color='orange');
plt.title('Country total orders')
plt.ylabel('Country')
plt.xlabel('orders');


# Maximum order is from UK that is more than 300000 and rest all coutry orders is below 50000 

# In[985]:


#Total SKU (distinct count of Stock code) by country and month
stock_monthly=df.groupby('month')['StockCode'].nunique().reset_index()


# In[986]:


stock_monthly


# In[987]:


#stock by month
plt.figure(figsize=[15,5])
sns.barplot(x="month", y="StockCode", data=stock_monthly,
            label="stock by monthly", palette="husl")


# In[988]:


country_stock=df.groupby('Country')['StockCode'].nunique()[0:10].reset_index()


# In[989]:


country_stock


# In[990]:


plt.figure(figsize=[15,5])
sns.barplot(x="Country", y="StockCode", data=country_stock,
            label="stock by country", palette="husl")


# In[991]:


#Monthly revenue (avg) per order
avg_revenue=df.groupby('month')['Revenue'].mean().reset_index()


# In[992]:


avg_revenue


# In[993]:


plt.figure(figsize=[15,5])
sns.barplot(x="month", y="Revenue", data=avg_revenue,
            label="revenueby order", palette="husl")


# In[994]:


#Find whether a customer is the new customer or not. A new customer would be figured out based on their first date of purchase.
#Figure out new customer on monthly basis.
min_date_buy=df.groupby('CustomerID').date.min().reset_index()


# In[995]:


min_date_buy


# In[996]:


min_date_buy.columns=['CustomerID','minpurchasedate']


# In[997]:


min_date_buy['minpurchasemonth']= min_date_buy['minpurchasedate'].map(lambda date:date.month)


# In[998]:


min_date_buy


# In[999]:


df=pd.merge(df,min_date_buy, on = 'CustomerID')


# In[1000]:


df.head()


# In[1001]:


#Find total revenue per month for new and existing customer per month. Show it in a visual
df['user_type'] ='new'
df.loc[df['month']>df['minpurchasemonth'],'user_type']= 'Existing'


# In[1002]:


df.head()


# In[1003]:


df_usertype_revenue = df.groupby(['month','user_type'])['Revenue'].sum().reset_index()


# In[1004]:


df_usertype_revenue


# In[1005]:


plt.figure(figsize=[15,5])
sns.lineplot(x="month", y="Revenue", data=df_usertype_revenue,hue='user_type')


# In[1006]:


plt.figure(figsize=(14,5))

df.loc[df['user_type']=='new'].groupby('day')['Revenue'].nunique().plot(color='b',marker='o',label='New user')
df.loc[df['user_type']=='Existing'].groupby('day')['Revenue'].nunique().plot(color='r',marker='o',label='Existing user')

plt.ylabel("Revenue")
plt.legend(loc='upper left')
plt.title("day-wise-user-revenue")
plt.grid(False)


# In[1007]:


#Calculate monthly retention rate (using crosstab() function of pandas) and find out total retained user on a monthly basis.


# In[1008]:


data_purchase = df.groupby(['CustomerID','month'])['Revenue'].sum().reset_index()


# In[1009]:


data_purchase


# In[1010]:


data_retention = pd.crosstab(data_purchase['CustomerID'],data_purchase['month']).reset_index()


# In[1011]:


data_retention


# # Customer Segmentation

# In[1012]:


#checking max date
snap_date = df['InvoiceDate'].max() + timedelta(days=1)


# In[1013]:


snap_date


# In[1014]:


df_RFM=df.groupby('CustomerID').agg({'InvoiceDate': lambda x:(df['InvoiceDate'].max().date()- x.max().date()).days,'InvoiceNo':'count','Revenue':'sum'})


# In[1015]:


df_RFM.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'frequency','Revenue':'Monetory'},inplace=True)


# In[1016]:


df_RFM.reset_index().head()


# In[1017]:


fig,axes = plt.subplots(1,3,figsize=(20,5))
for i,feature in enumerate(list(df_RFM.columns)):
    sns.distplot(df_RFM[feature],ax=axes[i])


# Data is right-skewed 

# In[1018]:


df_RFM


# In[1019]:


df_RFM.describe()


# In[1020]:


sns.heatmap(df_RFM.corr())


# In[1021]:


#spliting into four segments using quantile
quantiles = df_RFM.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[1022]:


quantiles


# In[1023]:


#function to create R,F and M segmentation
def RScoring(r,f,m):
    if r <= m[f][0.25]:
        return 1
    elif r <= m[f][0.50]:
        return 2
    elif r <= m[f][0.75]:
        return 3
    else:
        return 4
def FnMScoring(r,f,m):
    if r <= m[f][0.25]:
        return 4
    elif r <= m[f][0.50]:
        return 3
    elif r <= m[f][0.75]:
        return 2
    else:
        return 1
    


# In[1024]:


df_RFM['R'] = df_RFM['Recency'].apply(RScoring, args =('Recency',quantiles,))
df_RFM['F'] = df_RFM['frequency'].apply(FnMScoring, args =('frequency',quantiles,))
df_RFM['M'] = df_RFM['Monetory'].apply(FnMScoring, args =('Monetory',quantiles,))


# In[1025]:


df_RFM.head()


# In[1026]:


#concatintaing all the three scores
df_RFM['RFMgroup']=df_RFM.R.map(str) + df_RFM.F.map(str) + df_RFM.M.map(str)
#adding all the three score
df_RFM['RFMscore'] = df_RFM[['R','F','M']].sum(axis=1)


# In[1027]:


df_RFM.head()


# In[1028]:


# Grouping them in different Loyalty_level
Loyalty_level=['Platinum','Gold','Silver','Bronze']
score_cut=pd.qcut(df_RFM.RFMscore,q = 4,labels=Loyalty_level)
df_RFM['RFM_Loyalty'] = score_cut.values
df_RFM.reset_index().head()


# In[1029]:


#validate
df_RFM[df_RFM['RFMgroup']=='111'].sort_values('Monetory',ascending=False).reset_index().head()


# # K-Mean Clustring

# In[1030]:


#handle the negative and zero values so as to handle infinite numbers during log tranformation
def handle_neg_zero(num):
    if num <= 0:
        return 1
    else:
        return num
#appy handle_neg_zero function to recency and monetory columns
df_RFM['Recency'] =[handle_neg_zero(x) for x in df_RFM.Recency]
df_RFM['Monetory'] =[handle_neg_zero(x) for x in df_RFM.Monetory]
#Perform logtransformation to bring data into normal distribution
log_data = df_RFM[['Recency','frequency','Monetory']].apply(np.log,axis =1).round(3)


# In[1031]:


fig,axes = plt.subplots(1,3,figsize=(20,5))
for i,feature in enumerate(list(log_data.columns)):
    sns.distplot(log_data[feature],ax=axes[i])


# In[1032]:


# bring the data into same scale
scaleobj = StandardScaler()
Scaled_data = scaleobj.fit_transform(log_data)
#tranform it back to dataframe
Scaled_data =pd.DataFrame(Scaled_data,index = df_RFM.index,columns =log_data.columns)


# In[1033]:


sum_of_sq_dist={}
for k in range(1,15):
    km =KMeans(n_clusters =k,init='k-means++',max_iter=1000)
    km =km.fit(Scaled_data)
    sum_of_sq_dist[k] = km.inertia_
#plot the graph for the sum of square distance vallues and number of cluster
sns.pointplot(x=list(sum_of_sq_dist.keys()),y=list(sum_of_sq_dist.values()))
plt.title('Elbow method for optimal k')
plt.xlabel('Number of cluster')
plt.show()
    


# In[1034]:


#Building Kmean clustering
KMean_clust =KMeans(n_clusters =4,init='k-means++',max_iter =1000)
KMean_clust.fit(Scaled_data)

#find the cluster for the oberservayion
df_RFM['Cluster'] = KMean_clust.labels_
df_RFM.head()


# In[1035]:


ax=plt.figure(figsize=(7,7))
Colors =['red','green','blue','black']
df_RFM['Color']=df_RFM['Cluster'].map(lambda p: Colors[p])
ax= df_RFM.plot(
    kind ='scatter',
     x='Recency',y='frequency',figsize=(10,8),
     c = df_RFM['Color'])


# In[1036]:


df_RFM.head()


# In[1037]:


from sklearn.metrics import silhouette_score


# In[1038]:


KMean_clust =KMeans(n_clusters = 3,init='k-means++',max_iter =1000)

KMean_clust.fit(Scaled_data)
Cluster =KMean_clust.labels_


# In[1039]:


from sklearn.metrics import silhouette_score
silhouette_score(Scaled_data,Cluster)


# In[1040]:


import pickle


# In[1041]:


f = open('Loyalty-type.pkl','wb')
pickle.dump(KMean_clust,f)
f.close()


# In[1042]:


# Loading model to compare the results
model = pickle.load(open('Loyalty-type.pkl','rb'))
print(model.predict([[325, 2, 1]]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




