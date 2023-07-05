import matplotlib.pyplot as plt
import numpy as np

# data to plot
n_groups = 2
means_frank = (123, 25, 11, 2)
means_guido = (410, 76, 42, 1)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='b',
label='Sarcasm')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='g',
label='Not Sarcasm')

# Adding the numbers on top of each bar
for i in range(n_groups):
    plt.text(x=index[i], y=means_frank[i]+1, s=means_frank[i], ha='center', va='bottom')
    plt.text(x=index[i]+bar_width, y=means_guido[i]+1, s=means_guido[i], ha='center', va='bottom')

plt.xlabel('Text')
plt.ylabel('Example Number')
plt.title('Data Distribution')
plt.xticks(index + 0.5 * bar_width, ('happy', 'angry', 'neutral', 'sad'))
plt.legend()

plt.tight_layout()
plt.savefig('bar_chart.png')
