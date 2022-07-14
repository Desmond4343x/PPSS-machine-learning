#Clustering
kmeans = KMeans(n_clusters=10, n_init=20)
y_pred_kmeans = kmeans.fit_predict(encoded_imgs)
y_pred_kmeans[:10]
#Scoring
score = sklearn.metrics.rand_score(Y_test, y_pred_kmeans)
print(score)
