import pandas as pd
import matplotlib.pyplot as plt
datasets = pd.read_csv('E:/datasets/Salary_data.csv')
from sklearn.model_selection import train_test_split

X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

regressor = LinearRegression()
regressor.fit(X_train, y_train)

Y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':Y_pred})
print(df)

print('Mean Square error value: ',mean_squared_error(y_test,Y_pred))

plt.title('Training data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(X_train, y_train)
plt.show()

plt.title('Testing data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(X_test, y_test)
plt.show()


#2nd Question
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
km = pd.read_csv('E:/datasets/K-Mean_Dataset.csv')

x = km.iloc[:,[1,2,3,4]]
y = km.iloc[:,-1]

print(km.isnull().any())

print(km.fillna(km['CREDIT_LIMIT'].mean(), inplace = True))
print(km.fillna(km['MINIMUM_PAYMENTS'].mean(), inplace = True))
print(km.isnull().any())

from sklearn.cluster import KMeans
nclusters = 3
km = KMeans(n_clusters=nclusters)
km.fit(x)

wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print('Silhouette Score for above clustering is: ',score)

#3rd question

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)


nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(X_scaled)

y_scaled_cluster_kmeans = km.predict(X_scaled)
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_scaled_cluster_kmeans)
print('Silhouette Score for above clustering is: ',score)