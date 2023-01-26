import matplotlib.pyplot as plt

plt.plot([2000, 4000, 8000, 10000, 12000], [71.08, 74.53, 76.3, 76.86, 76.93], label='Naive Bayes Classifier')
plt.plot([2000, 4000, 8000, 10000, 12000], [70.12, 75.0, 76.9, 77.44, 77.56], label='Linear Kernel SVM')
plt.plot([2000, 4000, 8000, 10000, 12000], [71.62, 75.7, 77.05, 77.78, 77.84], label='RBF Kernel SVM')
plt.plot([2000, 4000, 8000, 10000, 12000], [61.15, 65.53, 66.20, 68.26, 67.51], label='Neural Network')
plt.plot([2000, 4000, 8000, 10000, 12000], [67.17, 70.92, 72.51, 72.81, 73.17], label='Cosine Similarity')
plt.plot([2000, 4000, 8000, 10000, 12000], [17.75, 15.35, 14.42, 13.41, 13.37], label='Manhattan Distance')
plt.plot([2000, 4000, 8000, 10000, 12000], [65.37, 68.94, 70.20, 70.27, 70.31], label='Euclidean Distance')
plt.ylabel('Accuracy (%)')
plt.xlabel('Number of stems')
plt.legend()
plt.grid(True)
plt.show()

