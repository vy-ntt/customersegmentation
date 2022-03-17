import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import squarify
from datetime import datetime
import plotly.express as px
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import pickle
import streamlit as st

# 1. Read data
data = pd.read_csv("OnlineRetail.csv", encoding='latin-1')

#--------------
# GUI
st.title("Data Science Project")
st.write("## CUSTOMER SEGMENTATION")

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding="latin-1")
    data.to_csv("OnlineRetail_new.csv", index=False)

##############################################################################
# 2. Data pre-processing
# Creating NewID column and Assigning to InvoiceNo wherever CustomerID is null
data['NewID'] = data['CustomerID']
data.loc[data['CustomerID'].isnull(), ['NewID']] = data['InvoiceNo']

# Remove all non-digits from column NewID
data['NewID'] = data['NewID'].astype(str).str.replace('\D+', '')

# Convert to integer
data['NewID'] = pd.to_numeric(data['NewID'])

# Converting object type to datetime for InvoiceDate
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], infer_datetime_format=True)
data['InvoiceDate'] = data['InvoiceDate'].dt.strftime('%Y-%m-%d')

# Checking Country
# data_new = data.groupby(['Country'], as_index=False)['NewID'].count().sort_values(by='NewID', ascending=False)

data_filtered = data[data.UnitPrice>0]
# ###############################################################################################

# 3. Build model
## 3.1. RFM Segmentation
### Adding Monetary information by calculating total value of transaction by multiplying unit price and quantity of the product
data_filtered['Total_sales'] = data_filtered['UnitPrice']*data_filtered['Quantity']
data_filtered['InvoiceDate'] = data_filtered['InvoiceDate'].astype('datetime64[ns]')
max_date = data_filtered['InvoiceDate'].max().date()

Recency = lambda x : (max_date - x.max().date()).days
Frequency = lambda x : len(x.unique())
Monetary = lambda x : round(sum(x), 2)

data_RFM = data_filtered.groupby('NewID').agg({'InvoiceDate': Recency,
                                       'InvoiceNo': Frequency,
                                       'Total_sales': Monetary})
# Rename the columns of dataframe
data_RFM.columns = ['Recency', 'Frequency', 'Monetary']

# Descending sorting
data_RFM = data_RFM.sort_values('Monetary', ascending=False)

### Create labels for Recency, Frequency, Monetary

r_labels = range(4, 0, -1) # số ngày tính từ lần cuối mua hàng lớn thì gắn nhãn nhỏ, ngược lại thì gắn nhãn lớn
f_labels = range(1, 5)
m_labels = range(1, 5)

### Assign these labels to 4 equal percentile groups

r_groups = pd.qcut(data_RFM['Recency'].rank(method='first'), q=4, labels=r_labels)
f_groups = pd.qcut(data_RFM['Frequency'].rank(method='first'), q=4, labels=f_labels)
m_groups = pd.qcut(data_RFM['Monetary'].rank(method='first'), q=4, labels=m_labels)

### Create new columns R, F, M

data_RFM = data_RFM.assign(R=r_groups.values, F=f_groups.values, M=m_groups.values)

### Concating RFM quartile values to create RFM Segments
def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
data_RFM['RFM_Segment'] = data_RFM.apply(join_rfm, axis=1)

### Counting number of unique segments
rfm_count_unique = data_RFM.groupby('RFM_Segment')['RFM_Segment'].nunique()

### Calculate RFM_Score
data_RFM['RFM_Score'] = data_RFM[['R', 'F', 'M']].sum(axis=1)

### Manual Segmentation
def rfm_level(df):
    if (df['R'] == 4 and df['F'] ==4 and df['M'] == 4)  :
        return 'VIP'
    
    elif (df['R'] == 4 and df['F'] ==1 and df['M'] == 1):
        return 'NEW'
    
    else:     
        if df['M'] == 4:
            return 'BIG SPENDER'
        
        elif df['F'] == 4:
            return 'LOYAL'
        
        elif df['R'] == 4:
            return 'ACTIVE'
        
        elif df['R'] == 1:
            return 'LOST'
        
        elif df['M'] == 1:
            return 'LIGHT'
        
        return 'REGULAR'

### Create a new column RFM_Level

data_RFM['RFM_Level'] = data_RFM.apply(rfm_level, axis=1)
data_RFM['RFM_Level'].value_counts()

### Calculate mean values for each segement
rfm_agg = data_RFM.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

rfm_agg.columns = rfm_agg.columns.droplevel()
rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

### Reset the index
rfm_agg = rfm_agg.reset_index()

## 3.2. KMeans Model
data_now = data_RFM[['Recency','Frequency','Monetary']]
RobustScaler = RobustScaler()
RobustScaler.fit_transform(data_now)
data_scaled = RobustScaler.transform(data_now)
data_now_scaled = pd.DataFrame(data_scaled, columns=['scaled_Recency', 'scaled_Frequency', 'scaled_Monetary'])
data_now = pd.concat([data_now.reset_index(drop=True), data_now_scaled], axis=1)

### Choosing k
wsse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_now[['scaled_Recency', 'scaled_Frequency', 'scaled_Monetary']])
    wsse[k] = kmeans.inertia_

model_kmeans = KMeans(n_clusters=6, random_state=42)
model_kmeans.fit(data_now[['scaled_Recency', 'scaled_Frequency', 'scaled_Monetary']])


data_now['Cluster'] = model_kmeans.labels_
data_now.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary':['mean', 'count']}).round(2)

# Calculate average values for each RFM_Level, and return a size of each segment
rfm_agg2 = data_now.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

rfm_agg2.columns = rfm_agg2.columns.droplevel()
rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)

# Reset the index
rfm_agg2 = rfm_agg2.reset_index()

# Change thr Cluster Columns Datatype into discrete values
rfm_agg2['Cluster'] = 'Cluster '+ rfm_agg2['Cluster'].astype('str')

# Print the aggregated dataset
rfm_agg2


############################################################################
#4. Save models
pkl_filename_rfm = "customer_segmentation_rfm.pkl"  
with open(pkl_filename_rfm, 'wb') as file:  
    pickle.dump(rfm_agg, file)

pkl_filename_kmeans = "customer_segmentation_kmeans.pkl"  
with open(pkl_filename_kmeans, 'wb') as file:  
    pickle.dump(model_kmeans, file)

############################################################################ 
#5. Load models 
## Import pickle
with open(pkl_filename_rfm, 'rb') as file:  
    customer_segmentation_rfm = pickle.load(file)

with open(pkl_filename_kmeans, 'rb') as file:  
    customer_segmentation_kmeans = pickle.load(file)

# GUI
menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox("Menu", menu)
if choice == "Business Objective":
    st.subheader("Business Objective")
    st.write("""
	    ###### Công ty X chủ yếu bán các sản phẩm là quà tặng dành cho những dịp đặc biệt. Nhiều khách hàng của công ty là khách hàng bán buôn. Công ty X mong muốn có thể bán được nhiều sản phẩm hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.
	    """)  
    st.write("""###### ==> Mục tiêu/vấn đề: Xây dựng hệ thống phân cụm khách hàng dựa trên các thông tin do công ty cung cấp từ đó có thể giúp công ty xác định các nhóm khách hàng khác nhau để có chiến lược kinh doanh, chăm sóc khách hàng phù hợp.""")
    st.image("cs02.png")
    
elif choice == "Build Project":
    st.subheader("Build Project")
    st.write("##### 1. Data")
    st.dataframe(data.head(3))
    st.dataframe(data.tail(3))
    st.write("- Dữ liệu có: ", data.shape[0], "dòng")
    data = data.drop_duplicates()
    st.write("- Sau khi xóa các dòng trùng, dữ liệu còn: ", data.shape[0], "dòng")

    st.write("##### 2. RFM Model")
    st.write("""
        - Recency (R) of a customer: Days since the last purchase.
        - Frequency (F) of the turnover of a customer: Number of purchases
        - Monetary (M): The total turnover of a customer""")
    st.dataframe(data_RFM.head())

    st.write("###### Visualization")   
    fig1 = plt.figure(figsize = (8, 10))
    plt.subplot(3, 1, 1)
    sns.distplot(data_RFM['Recency'])
    plt.subplot(3, 1, 2)
    sns.distplot(data_RFM['Frequency'])
    plt.subplot(3, 1, 3)
    sns.distplot(data_RFM['Monetary'])
    st.pyplot(fig1)

    st.write("###### Calculate mean values for each segement")
    st.dataframe(rfm_agg)


    # TreeMap
    st.write("###### TreeMap")
    fig2 = plt.gcf()
    ax2 = fig2.add_subplot()
    fig2.set_size_inches(15, 12)

    colors_dict = {'ACTIVE':'orange','BIG SPENDER':'blue', 'LIGHT':'cyan',
                'LOST':'red', 'LOYAL':'purple', 'REGULAR':'green', 'VIP':'gold', 'NEW':'brown'}

    squarify.plot(sizes=rfm_agg['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                        for i in range(0, len(rfm_agg))], alpha=0.5 )

    plt.title('Customer Segments', fontsize=26, fontweight='bold')
    plt.axis('off')
    st.pyplot(fig2)    
    
    # ScatterPlot
    st.write("###### ScatterPlot")
    fig3 = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_Level",
           hover_name="RFM_Level", size_max=100)
    st.plotly_chart(fig3)

    # 3D ScatterPlot
    st.write("###### 3D ScatterPlot")
    fig4 = px.scatter_3d(data_RFM, x='Recency', y='Frequency', z='Monetary',
                    color = 'RFM_Level', opacity=0.5,
                    color_discrete_map = colors_dict)
    fig4.update_traces(marker=dict(size=5),
                  selector=dict(mode='markers'))
    st.plotly_chart(fig4)

    st.write("###### Tổng kết: RFM Model cho kết quả rằng các nhóm khách hàng phân cụm rất rõ ràng")

    st.write("##### 3. KMeans Model")   
    fig5 = px.line(x=list(wsse.keys()), y=list(wsse.values()), title="Elbow Method <br>The Elbow Method showing the optimal k")
    fig5.update_layout(xaxis_title_text='k',yaxis_title_text='WSSE')
    st.plotly_chart(fig5)

    st.write("Result of KMeans Model")
    st.dataframe(rfm_agg2)
    st.write("###### TreeMap")
    fig6 = plt.gcf()
    ax6 = fig6.add_subplot()
    fig6.set_size_inches(20, 15)

    colors_dict2 = {'Cluster0':'yellow','Cluster1':'blue', 'Cluster2':'cyan',
                'Cluster3':'purple', 'Cluster4':'red', 'Cluster5': 'black'}

    squarify.plot(sizes=rfm_agg2['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict2.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
                        for i in range(0, len(rfm_agg2))], alpha=0.5 )


    plt.title("Customer Segments",fontsize=26,fontweight="bold")
    plt.axis('off')
    st.pyplot(fig6)

    st.write("###### ScatterPlot")
    fig7 = px.scatter_3d(rfm_agg2, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
                        color = 'Cluster', opacity=0.3)
    fig7.update_traces(marker=dict(size=20),                  
                    selector=dict(mode='markers'))
    st.plotly_chart(fig7)

    st.write("###### 3D ScatterPlot")
    fig8 = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
           hover_name="Cluster", size_max=100)
    st.plotly_chart(fig8)


#########################################################################################################
elif choice =="New Prediction":
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type == "Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=["txt", "csv"])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)
            # st.write(lines.columns)
            lines = lines[0]
            flag = True
        if type == "Input":
            email = st.text_area(label="Input your content:")
            if email != "":
                lines = np.array([email])
                flag = True

        if flag:
            st.write("Content:")
            if len(lines)>0:
                st.code(lines)
                x_new = count_model.transform(lines)
                y_pred_new = ham_spam_model.predict(x_new)
                st.code("New predictions (0: Ham, 1: Spam): " + str(y_pred_new))











