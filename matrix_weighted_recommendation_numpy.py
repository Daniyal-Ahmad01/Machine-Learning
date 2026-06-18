import numpy as np

movie_ratings = np.array([
    [1, 2, 2.5],
    [0, 5, 4],
    [0, 0, 2.5]
])

weights = np.array([9, 7, 5])

result = np.dot(movie_ratings, weights)

print(result)