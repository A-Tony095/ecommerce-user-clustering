import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use('seaborn-v0_8-whitegrid')

file_path = r"D:\基于无监督聚类的电商用户行为洞察与精准营销分析\原始数据\UserBehavior.csv"
reader = pd.read_csv(
    file_path,
    chunksize=50000,
    header=None,
    names=['user_id', 'item_id', 'category_id', 'behavior', 'timestamp']
)

# 加载数据（最多20个分块）
data_list = []
for i, chunk in enumerate(reader):
    data_list.append(chunk)
    if len(data_list) >= 20:
        break
df = pd.concat(data_list, ignore_index=True)
print(f"原始数据形状：{df.shape}")

# 去重
df = df.drop_duplicates()
print(f"去重后数据形状：{df.shape}")

# 时间转换
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday

# 时间筛选（2017-11-25 ~ 2017-12-04）
df = df[(df['datetime'] >= '2017-11-25') & (df['datetime'] <= '2017-12-04')]
print(f"筛选时间后数据形状：{df.shape}")
# 异常值过滤
df = df[df['user_id'] > 0]
df = df[df['item_id'] > 0]

user_behavior = df.groupby('user_id')['behavior'].value_counts().unstack(fill_value=0)

user_active_day = df.groupby('user_id')['date'].nunique().rename('active_days')
user_pv_depth = df.groupby('user_id').size().rename('total_behavior')
user_buy_ratio = (user_behavior['buy'] / user_behavior['pv']).fillna(0).rename('buy_rate')

# 合并所有特征
user = pd.concat([user_behavior, user_active_day, user_pv_depth, user_buy_ratio], axis=1)
user = user.fillna(0)
print("\n用户行为特征表：")
print(user.head())

features = ['pv', 'fav', 'cart', 'buy', 'active_days', 'buy_rate', 'total_behavior']
X = user[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia_list = []
sil_list = []
K_range = range(2, 8)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=888)
    labels = model.fit_predict(X_scaled)
    inertia_list.append(model.inertia_)
    sil_list.append(silhouette_score(X_scaled, labels))

# 绘制双指标图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia_list, 'o-', c='#ff6b6b')
plt.xlabel('聚类数量K')
plt.ylabel('簇内SSE')
plt.title('肘部法则')

plt.subplot(1, 2, 2)
plt.plot(K_range, sil_list, 'o-', c='#4ecdc4')
plt.xlabel('聚类数量K')
plt.ylabel('轮廓系数')
plt.title('轮廓系数选择最优K')
plt.tight_layout()
plt.show()

best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=888)
user['cluster'] = kmeans.fit_predict(X_scaled)

# 计算整体轮廓系数
sil_score = silhouette_score(X_scaled, user['cluster'])
print(f"\n最终轮廓系数：{sil_score:.3f}")

cluster_count = user['cluster'].value_counts().sort_index()
labels = [
    '低价值用户',
    '潜力用户',
    '高价值忠诚用户',
    '收藏偏好用户'
]

plt.figure(figsize=(7, 7))
plt.pie(
    cluster_count,
    labels=labels,
    autopct='%.1f%%',
    colors=['#ff9999', '#66b3ff', '#ffcc99', '#99ff99'],
    startangle=90,
    textprops={'fontsize': 12}
)
plt.title('电商用户分层占比', fontsize=15)
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
for i in range(best_k):
    mask = user['cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=labels[i], alpha=0.6)
plt.legend()
plt.title('PCA用户聚类分布', fontsize=14)
plt.show()

cluster_mean = user.groupby('cluster')[['pv', 'fav', 'cart', 'buy', 'active_days', 'buy_rate']].mean()
print("\n各簇用户特征均值：")
print(cluster_mean.round(2))

cluster_mean.T.plot(kind='bar', figsize=(12, 6), color=['#ff9999', '#66b3ff', '#ffcc99', '#99ff99'])
plt.title('不同用户群体行为特征对比', fontsize=14)
plt.xticks(rotation=0)
plt.legend(labels)
plt.show()
print("\n" + "="*50)
print("          电商用户分层精准营销运营策略")
print("="*50)
strategies = [
    "0 类-低价值用户：点击少、购买极低 → 新客福利、低价引流、提升活跃度",
    "1 类-潜力用户：加购收藏多、下单少 → 购物车降价提醒、限时优惠券",
    "2 类-高价值用户：高频活跃、高购买 → VIP服务、专属折扣、重点留存",
    "3 类-收藏偏好用户：收藏行为突出 → 收藏商品推送、库存提醒、限时优惠"
]
for s in strategies:
    print(s)
user.to_excel("电商用户聚类结果.xlsx", index=True)
cluster_mean.round(2).to_excel("各类用户特征均值.xlsx")
print("\n聚类结果已保存为 Excel 文件！")