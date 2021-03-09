import matplotlib.pyplot as plt
import numpy as np

dataset = ['mnist_d','mnist_f','cifar_10','cifar_100_f','cifar_100_c']
N = 5

ann_precision = [98.45, 88.06, 42.81, 15.14, 29.46]
cnn_precision = [99.44, 92.85, 74.77, 45.7, 57.25]

# plt.style.use('ggplot')
# x_pos = [i for i, _ in enumerate(dataset)]
# plt.bar(x_pos, cnn_precision, color='blue')
# plt.xlabel("DataSet")
# plt.ylabel("Precision (%)")
# plt.title("Precision of CNN model")
#
# plt.xticks(x_pos, dataset)
# plt.savefig('CNN_Accuracy_Plot.pdf')

ind = np.arange(N)
width = 0.35
plt.bar(ind, ann_precision, width, label='ANN', color='red')
plt.bar(ind + width, cnn_precision, width,
    label='CNN', color='blue')

plt.ylabel('Precision (%)')
plt.title('Comparision between ANN and CNN performance')

plt.xticks(ind + width / 2, ('mnist_d','mnist_f','cifar_10','cifar_100_f','cifar_100_c'))
plt.legend(loc='best')
plt.savefig('ANN_CNN_Compare.png')