import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\HP\\Desktop\\emotional_monitoring_dataset_with_target (1).csv")
data.head()

data.info()

#Emotional state

sns.countplot(data=data,x='EmotionalState')
for container in plt.gca().containers:
    plt.bar_label(container, label_type="edge", padding=2)
    plt.show()

#EngagementLevel

sns.countplot(data=data,x='EngagementLevel')
for container in plt.gca().containers:
    plt.bar_label(container, label_type="edge", padding=2)
    plt.show()


#EmotionalState and CongnitiveState 

plt.figure(figsize=(15,15))
numeric_data=data.drop(columns=['EmotionalState','CognitiveState'])
sns.heatmap(numeric_data.corr(),annot=True,fmt='.2f')
plt.show()


data.hist(figsize=(12, 10), bins=15, color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()

#Boxplot
plt.figure(figsize=(15, 8))
sns.boxplot(data=data.select_dtypes(include=['float64', 'int64']))
plt.xticks(rotation=45)
plt.title('Boxplots for Numerical Variables')
plt.show()

#pairplot
sns.pairplot(data, vars=['HeartRate', 'CortisolLevel', 'Temperature', 'SmileIntensity'], hue='EmotionalState')
plt.show()


#scatterplot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='PupilDiameter', y='CortisolLevel', hue='EmotionalState')
plt.title('Pupil Diameter vs Cortisol Level by Emotional State')
plt.show()

from sklearn.cluster import KMeans # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize clusters
sns.pairplot(data, vars=['HeartRate', 'CortisolLevel', 'PupilDiameter'], hue='Cluster')
plt.show()

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False) # type: ignore

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar', color='skyblue')
plt.title('Feature Importance')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.tight_layout()
plt.show()

engagement_dist = data['EngagementLevel'].value_counts()
engagement_dist.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Engagement Level Distribution')
plt.ylabel('')
plt.show()

