
## 4. Visualization


```python
# word_vecs = tf.concat([model.center_embeddings.weights[0], model.context_embeddings.weights[0]], axis=-1).numpy()
center_vecs = model.center_embeddings.weights[0].numpy()
center_vecs.shape
```




    (102300, 64)




```python
from matplotlib import pyplot as plt


visualize_words = [
    "movie", "film", "story",
    "good", "enjoyable", "great", "bad",
    "coffee", "tea", "milk"
]

# visualize_words = [
#     "shampoo", "baby", "flower",
#     "photo", "chocolate", "cream", "chicken",
#     "laptop", "cake", "hot","camera"
# ]
visualize_idx = [tokenizer.vocab[word] for word in visualize_words]
visualize_vecs = center_vecs[visualize_idx, :]

temp = (visualize_vecs - np.mean(visualize_vecs, axis=0))
covariance = 1.0 / len(visualize_idx) * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])

for i in range(len(visualize_words)):
    plt.text(coord[i, 0], coord[i, 1], visualize_words[i],
             bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

plt.savefig('word_vectors.png')
plt.show()
```


    
![png](output_26_0.png)
    



```python
def knn(vec, mat, k):
    """ Implement the KNN algorithm based on cosine similarity, which will be used for analysis.

        Args:
            vec: numpy ndarray, the target vector
            mat: numpy ndarray, a matrix contains all the vectors (each row is a vector)
            k: the number of the nearest neighbors you want to find.
            
        Return:
            indices: the k indices of the matrix's rows that are closest to the vec
    """
    indicies = []
    # Start your code here
    # Note: DO NOT use for loop to calculate the similarity between two vectors. You are required to vectorize the calculation.
    # Hint: See np.argsort
    cos_similarities = np.dot(mat, vec) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(vec))
    indices = np.argsort(cos_similarities)[-k:][::-1]
    # End
    
    return indices
```


```python
for word in visualize_words:
    idx = tokenizer.vocab[word]
    vec = center_vecs[idx]
    indices = knn(vec, center_vecs, 10)
    closed_words = [tokenizer.inverse_vocab[i] for i in indices]
    print('Word: "{}" is close to {}'.format(word, closed_words))
```

    Word: "movie" is close to ['movie', 'film', 'flick', 'movies', 'sequel', 'documentary', 'horror', 'cinema', 'anthology', 'feature']
    Word: "film" is close to ['film', 'movie', 'picture', 'documentary', 'cinema', 'sequel', 'feature', 'flick', 'films', 'rotterdam']
    Word: "story" is close to ['story', 'plot', 'stories', 'storyline', 'premise', 'tale', 'plotline', 'leisurely', 'plots', 'idea']
    Word: "good" is close to ['good', 'great', 'decent', 'bad', 'fine', 'nice', 'excellent', 'amazing', 'exceptional', 'interesting']
    Word: "enjoyable" is close to ['enjoyable', 'entertaining', 'lighthearted', 'middling', 'disappointing', 'fluff', 'escapist', 'hyped', 'verite', 'watchable']
    Word: "great" is close to ['great', 'amazing', 'good', 'wonderful', 'excellent', 'fine', 'awesome', 'nice', 'fantastic', 'exceptional']
    Word: "bad" is close to ['bad', 'terrible', 'horrible', 'awful', 'good', 'lame', 'stupid', 'lousy', 'atrocious', 'crappy']
    Word: "coffee" is close to ['coffee', 'booze', 'drinking', 'milk', 'farm', 'poison', 'pills', 'liquor', 'beer', 'wet']
    Word: "tea" is close to ['tea', 'coffee', 'cocoa', 'camford', 'wine', 'carmine', 'guinea', 'tester', 'manos', 'hassan']
    Word: "milk" is close to ['milk', 'soda', 'frozen', 'sucking', 'fried', 'wet', 'eating', 'bottles', 'cleaning', 'knee']
    


```python

```
