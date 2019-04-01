Let's try to using k Nearest Neighbors model for classification. The value of k influences the flexibility of the model. The smaller K gets, the better the model fits to the training set, but since our goal is to minimize prediction error for the test set, you need to try different values of k to find the best model.

Use the code below to test models with k equal to 1, 3, and 5 and to see that the model works best when k is 3.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# generating a wide range of K and find the best models
# also create a list to store the evaluation result for each value of k
kVals = range(1,10, 2)
evaluation = []

# loop over models with different parameter to find the one with lowest error rate
for k in range(1, 6, 2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # using test dataset for evaluation and append the result to the evaluation list
    score = model.score(X_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    evaluation.append(score)

# find the value of k with best performance
i = int(np.argmax(evaluation))
print("k=%d with best performance with %.2f%% accuracy given current testset" % (kVals[i], evaluation[i] * 100))
```