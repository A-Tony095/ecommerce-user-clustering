
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["simHei"]
plt.rcParams["axes.unicode_minus"]=False
#读取数据
reader=pd.read_csv(r"D:\基于无监督聚类的电商用户行为洞察与精准营销分析\原始数据\UserBehavior.csv",chunksize=50000,header=None,names=['user_id','item_id','category_id','behavior','timestamp'])
data_list=[]
for chunk in reader:
    data_list.append(chunk)
    if len(data_list)==20:
        break
df=pd.concat(data_list)
#去重
print("去重前:",df.shape)
df=df.drop_duplicates()
print("去重后:",df.shape)
#时间转换
df['datetime']=pd.to_datetime(df['timestamp'],unit='s')
print(df[['timestamp','datetime']].head())
#时间筛选
df=df[(df['datetime']>='2017-11-25')&(df['datetime']<='2017-12-04')]
print(df.shape)
user_behavior=df.groupby('user_id')['behavior'].value_counts().unstack(fill_value=0)
print(user_behavior.head)
#标准化
scaler=StandardScaler()
scale_data=scaler.fit_transform(user_behavior)
#肘部法则
inertia_list=[]
K_range=range(2,7)
for k in K_range:
    model=KMeans(n_clusters=k,random_state=888)
    model.fit(scale_data)
    inertia_list.append(model.inertia_)
plt.plot(K_range,inertia_list,marker='o')
plt.xlabel('聚类数量')
plt.ylabel('簇内误差平方和')
plt.title("肘部法则-寻找最佳聚类数")
plt.show()
#K-Means 无监督聚类
kmeans=KMeans(n_clusters=4,random_state=888)
kmeans.fit(scale_data)
user_behavior['level']=kmeans.labels_
print(user_behavior.head)
#轮廓系数
sil_score=silhouette_score(scale_data,kmeans.labels_)
print(f'轮廓系数:{sil_score:.3f}')
level_count=user_behavior['level'].value_counts()
plt.pie(
    level_count,
    labels=['低价值用户','中价值用户','高价值用户','特高价值用户'],
    autopct='%.1f%%',
    colors=['#ff9999','#66b3ff','#ffcc99','#99ff99']
)
plt.title('电商用户分层占比')
plt.show()
cluster_mean=user_behavior.groupby('level')[['buy','cart','fav','pv']].mean()
print(cluster_mean)
# 用户分层运营建议
print("\n电商运营精准营销建议")
print("0 类：低价值用户 → 推送低价商品，提升活跃度")
print("1 类：潜力用户 → 推送加购商品，促进下单")
print("2 类：高价值用户 → 专属优惠，重点留存")
print("3 类：收藏偏好用户 → 降价提醒，刺激购买")