## automating the detection of forged banknotes using a K-means Clustering algorithm

### Part 1 – The purpose of this project
The detection of forged bank notes is an incredibly important part of the daily operation of
any bank. In 2019, around 427,000 counterfeit Bank of England banknotes with a face value
of £9.8 million were taken out of circulation.

The process of detecting forged banknotes manually may be time consuming and possibly
inaccurate, and for these reasons it may be worth considering automating the process of
detecting forgeries.

In this report we will consider a model which has been trained to test features of various
banknotes and attempt to detect forgeries. We will then discuss the results of this model and
whether this solution could be of benefit to your bank.

The model used for this demonstration is an algorithm called K-means Clustering. This algorithm 
looks at a given set of data and sorts the data into a user-defined number of categories, or clusters. 
Since we want to test if a bank note is real or not, we set the algorithm to provide us with just two 
of these clusters: Genuine and Forged. 

### Part 2 – Describing the data
The data we used to train our model is the Banknote authentication Data Set from
University of California, Irvine. This dataset contains observations of 1372 individual bank
notes, each with four attributes we can measure with our algorithm.

The attributes themselves are based on image wavelet transformations, which are essentially methods used to
convert images in to numbers the algorithm can make use of. For the purpose of this model
we are only considering two of these transformations which are labelled V1 and V2.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from sklearn.cluster import KMeans
%matplotlib inline
```


```python
df = pd.read_csv('Banknote-authentication-dataset.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.62160</td>
      <td>8.6661</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.54590</td>
      <td>8.1674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.86600</td>
      <td>-2.6383</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.45660</td>
      <td>9.5228</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.32924</td>
      <td>-4.4552</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1372.000000</td>
      <td>1372.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.433735</td>
      <td>1.922353</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.842763</td>
      <td>5.869047</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-7.042100</td>
      <td>-13.773100</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.773000</td>
      <td>-1.708200</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.496180</td>
      <td>2.319650</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.821475</td>
      <td>6.814625</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.824800</td>
      <td>12.951600</td>
    </tr>
  </tbody>
</table>
</div>



### Part 3 – Analyzing the data
To visualise the banknote data, we first produced a scatter plot to observe the relationship
between our two parameters V1 and V2. To prepare our data for use by the K-means Clustering algorithm we are also normalising the data, to transform each variable to a value between 0-1. 

Each point in the below graph represents a single
bank note in the data. Note that at this point the data is unlabelled – that is we do not know
which notes are genuine or counterfeit. 


```python
normalized_df = (df-df.min())/(df.max()-df.min())
```


```python
normalized_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.769004</td>
      <td>0.839643</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.835659</td>
      <td>0.820982</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.786629</td>
      <td>0.416648</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.757105</td>
      <td>0.871699</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.531578</td>
      <td>0.348662</td>
    </tr>
  </tbody>
</table>
</div>




```python
normalized_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1372.000000</td>
      <td>1372.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.539114</td>
      <td>0.587301</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.205003</td>
      <td>0.219611</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.379977</td>
      <td>0.451451</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.543617</td>
      <td>0.602168</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.711304</td>
      <td>0.770363</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, graph = plt.subplots(figsize=(8,8))
graph.scatter(x='V1', y='V2', data=normalized_df, c=['Purple'])
plt.xlabel('V1', {'size':16})
plt.ylabel('V2', {'size':16})
plt.title('Banknote Features')

mean_patch = ptc.Ellipse([normalized_df['V1'].mean(), normalized_df['V2'].mean()], 2*normalized_df['V1'].std(), 2*normalized_df['V2'].std(), alpha=0.25, color='Orange')

plt.savefig('plot1')
```


    
![png](output_9_0.png)
    


We can see from the data that there is a significant degree of variance for both V1 and V2 in
the bank notes in our sample, however by looking at this data it is difficult to suggest where a
line could be drawn to divide these notes in to two categories. This is where our algorithm
comes in to use.


```python
normalized_df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>V1</th>
      <td>1.000000</td>
      <td>0.264026</td>
    </tr>
    <tr>
      <th>V2</th>
      <td>0.264026</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = KMeans(n_clusters=2)
```


```python
model.fit(normalized_df)
```




    KMeans(n_clusters=2)




```python
model.labels_
```




    array([0, 0, 0, ..., 1, 1, 1])




```python
# Add the results of the algorithm to the dataframe
normalized_df['label'] = model.labels_
```


```python
normalized_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.769004</td>
      <td>0.839643</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.835659</td>
      <td>0.820982</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.786629</td>
      <td>0.416648</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.757105</td>
      <td>0.871699</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.531578</td>
      <td>0.348662</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, graph = plt.subplots(figsize=(8,8))
graph.scatter(x='V1', y='V2', data=normalized_df[normalized_df['label'] == 0], label='Cluster 0', c='Purple')
graph.scatter(x='V1', y='V2', data=normalized_df[normalized_df['label'] == 1], label='Cluster 1', c='Orange')
plt.xlabel('V1', {'size':16})
plt.ylabel('V2', {'size':16})
plt.title('Banknote Features')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2027e0f06d0>




    
![png](output_17_1.png)
    


Here we have used the K-means Clustering algorithm to split the data in to two clusters.
Though we haven’t labelled which of these clusters represents real or forged bank notes at
this time, what is important is that the algorithm has analysed the data and defined the two
most prominent clusters. 


```python
model2 = KMeans(n_clusters=2)
model2.fit(normalized_df)
model2.labels_
normalized_df['label 2'] = model2.labels_
normalized_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>label</th>
      <th>label 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.769004</td>
      <td>0.839643</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.835659</td>
      <td>0.820982</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.786629</td>
      <td>0.416648</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.757105</td>
      <td>0.871699</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.531578</td>
      <td>0.348662</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
model3 = KMeans(n_clusters=2)
model3.fit(normalized_df)
model3.labels_
normalized_df['label 3'] = model2.labels_
normalized_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>label</th>
      <th>label 2</th>
      <th>label 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.769004</td>
      <td>0.839643</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.835659</td>
      <td>0.820982</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.786629</td>
      <td>0.416648</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.757105</td>
      <td>0.871699</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.531578</td>
      <td>0.348662</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, graph = plt.subplots(1, 3, figsize=(16,4))

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("V1")
plt.ylabel("V2")

graph[0].set_title('K-means pass 1')
graph[0].scatter(x='V1', y='V2', data=normalized_df[normalized_df['label'] == 0], label='Cluster 0', c='Purple', s=2)
graph[0].scatter(x='V1', y='V2', data=normalized_df[normalized_df['label'] == 1], label='Cluster 1', c='Orange', s=2)

graph[1].set_title('K-means pass 2')
graph[1].scatter(x='V1', y='V2', data=normalized_df[normalized_df['label 2'] == 0], label='Cluster 0', c='Purple', s=2)
graph[1].scatter(x='V1', y='V2', data=normalized_df[normalized_df['label 2'] == 1], label='Cluster 1', c='Orange', s=2)

graph[2].set_title('K-means pass 3')
graph[2].scatter(x='V1', y='V2', data=normalized_df[normalized_df['label 3'] == 0], label='Cluster 0', c='Purple', s=2)
graph[2].scatter(x='V1', y='V2', data=normalized_df[normalized_df['label 3'] == 1], label='Cluster 1', c='Orange', s=2)

# plt.savefig('plot3')
```




    <matplotlib.collections.PathCollection at 0x2027e27d7c0>




    
![png](output_21_1.png)
    


Each time we run K-means Clustering, the algorithm begins by creating starting points for
each cluster at random. This means we can run the algorithm multiple times and test if we
obtain the same results. Since in this case we have consistently identified the same clusters in
the data we can be confident the model is working as intended and the clusters it has
identified are a reliable means by which to group our data points.

## Testing predictions


```python
full_df = pd.read_csv('Banknote-full.csv')
```


```python
full_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.62160</td>
      <td>8.6661</td>
      <td>-2.8073</td>
      <td>-0.44699</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.54590</td>
      <td>8.1674</td>
      <td>-2.4586</td>
      <td>-1.46210</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.86600</td>
      <td>-2.6383</td>
      <td>1.9242</td>
      <td>0.10645</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.45660</td>
      <td>9.5228</td>
      <td>-4.0112</td>
      <td>-3.59440</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.32924</td>
      <td>-4.4552</td>
      <td>4.5718</td>
      <td>-0.98880</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['prediction'] = normalized_df['label']
df['class'] = full_df['Class'] - 1
```


```python
fig, graph = plt.subplots(1, 2, figsize=(11,5.5))

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("V1")
plt.ylabel("V2")

graph[0].set_title('Predicted Clusters')
scatter1 = graph[0].scatter(x='V1', y='V2', data=df[df['prediction'] == 0], label='Cluster 0', c='Purple', s=2)
scatter2 = graph[0].scatter(x='V1', y='V2', data=df[df['prediction'] == 1], label='Cluster 1', c='Orange', s=2)

graph[1].set_title('Real Clusters')
graph[1].scatter(x='V1', y='V2', data=df[df['class'] == 0], label='Cluster 0', c='Orange', s=2)
graph[1].scatter(x='V1', y='V2', data=df[df['class'] == 1], label='Cluster 1', c='Purple', s=2)

plt.legend([scatter1, scatter2], ['Genuine', 'Forged'], loc='upper right', bbox_to_anchor=(1.1,1))

plt.savefig('plot4')
```


    
![png](output_27_0.png)
    


Here we have plotted our predicted results (left) against that of the real data (right). We can
see from these visualisations that the clusters identified by our model are somewhat accurate.
In fact, out of 1372 bank notes, our model was able to correctly classify 1197, or 87.2%.


```python
df[df['prediction'] == df['class']].count()
```




    V1            1197
    V2            1197
    prediction    1197
    class         1197
    dtype: int64




```python
correct = df[df['prediction'] != df['class']].count()
```


```python
correct / df['V1'].count()
```




    V1            0.127551
    V2            0.127551
    prediction    0.127551
    class         0.127551
    dtype: float64



### Part 4 – Recommendations
The K-means Clustering algorithm was able to successfully identify genuine and forged bank
notes 87.2% of the time. While 87.2% is a reasonable rate of success, it is important to
consider several factors when considering whether to implement this model in your business.

It is worth noting that 87.2% is an overall accuracy score for the model. Of the banknotes
labelled incorrectly, these are a combination of false positives (i.e. genuine notes labelled as
forged) and false negatives (forged notes labelled as genuine). This could be investigated
further, and it would be worth considering which of these errors is best avoided. Is it more
damaging to dispose of genuine notes, or to unknowingly recirculate fraudulent notes?

A solution to be considered may be using multiple methods of identification. This model
could be used alongside a secondary source of identification in order to ‘double check’ the
results of the other system and create a more accurate system overall.

Finally, it is left to the business to consider the cost of any given system. While using a model
to predict the authenticity of banknotes may be more accurate than what is possible by a
human or alternative system, steps need to be taken to measure the various qualities of any
given note and input these measurements in to the model. These measurements could be
made by machines or people, but either way would incur costs which would need to be
considered.


```python

```
