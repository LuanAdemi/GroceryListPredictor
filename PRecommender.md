# PRecommender

The **precommender** model is used to generate a probability distribution for every user's shopping trip, which products he/she is likely to buy. This is used to create a shopping list. 

This model uses LSTM to perform a prediction based on the receipt collected in the past. Since this collected receipts are simply not enough, we started to search for a solution that makes a compromise between userfriendlyness and data quality, but quickly discovered, that it would be to much of a burden for the user to generate the data necessary.

"What do you do, if you have no idea on how to solve a problem? Right, ask the public!"

This sentence sort of shows what other solution we came up with. The solution got the name **KNetworks**...

### KNetworks

![gif](scatter.gif)

With **KNetworks**, we found a way of training a network (or should I say *k* networks) to be able to perform this task even with the lack of userspecific data. 

The algorithm uses the <a href="https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf">kmeans++</a> algorithm to classify every user by the average product count per week.

The GIF at the top shows exactly this (Here, we are using a feature size of two (to draw it in 2d), which obviously can be choosen as pleased).

Next up, we initialize a LSTM network for each found centroid and start a weighted training on the whole dataset, where the representation of each data point is based on the euclidean distance from the centroid.

This will create _k_ LSTM networks which are specialized on the data of the nearest data points (the users).

This method proposes some key advantages, making it a good one to choose:

- The prediction for every user are not only driven by their own data, but also by user with a similar shopping behaviour
- We only need a small amount of neural network as opposed to generating one for every user
- User with a basically minimal data pool can get predictions
- We have a way bigger data pool for each network to train with. This will increase the performance of the model.

In order to get a prediction, we follow a similar approach as show in the training step. We retrieve the probability distributions from **every** network and use a weighted mean as our final prediction.



#### Pseudocode

The following are some python flavored pseudocode representations of the KNetworks algortihm.

##### Initializing the centroids
```python
centroids := empty array
append a random n-tuple to centroids

for c_id in range(k-1):
    dist := empty array
    for point in data:
        d := infinity
        for i in range(size of centroids):
            temp_dist := distance(point, centroids[i])
            d = min(d, temp_dist)
        append d to dist
    
    next_centroid := data[argmin(dist), :]
    append next_centroid to centroids
```

##### Fitting the centroids to the data
```python
points := array with n-tuples
epochs := number of epochs

for epoch in range(epochs):
    distances := calcDistances(centroids, points)
    affiliations := argmin(distances)
    centroids = mean of point n-tuples around the centroids
```

##### Creating k LSTM networks
```python
networks := empty array
for i in range(k):
    append LSTM to networks
```

##### Training the k LSTM networks
```python
data := the user data
for i in range(k):
    sData = select data of users within a specific radius
    networks[i].train(sData)
```