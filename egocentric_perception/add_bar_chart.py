import matplotlib.pyplot as plt
import numpy as np

# data to plot
n_groups = 7

# These are sample values. Replace with your own data
'''
happy = (253, 280, 123, 410)
angry = (60, 41, 25, 76)
neutral = (29, 24, 11, 42)
sad = (3, 0, 2, 1)
'''

gth_sacrasm = (254, 218, 294, 497, 25, 12, 1)
gth_not_sarcasm = (229, 139, 240, 403, 18, 8, 0)
pred_sarcasm = (180, 131, 165, 218, 18, 7, 1)
pred_not_sarcasm = (264, 198, 338, 632, 23, 12, 0)


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.22
opacity = 0.8

rects1 = plt.bar(index, gth_sacrasm, bar_width, alpha=opacity, color='b', label='gth_sacrasm')
rects2 = plt.bar(index + bar_width, gth_not_sarcasm, bar_width, alpha=opacity, color='g', label='gth_not_sarcasm')
rects3 = plt.bar(index + 2 * bar_width, pred_sarcasm, bar_width, alpha=opacity, color='r', label='pred_sarcasm')
rects4 = plt.bar(index + 3 * bar_width, pred_not_sarcasm, bar_width, alpha=opacity, color='y', label='pred_not_sarcasm')

# Adding the numbers on top of each bar
for i in range(n_groups):
    plt.text(x=index[i], y=gth_sacrasm[i]+1, s=gth_sacrasm[i], ha='center', va='bottom')
    plt.text(x=index[i]+bar_width, y=gth_not_sarcasm[i]+1, s=gth_not_sarcasm[i], ha='center', va='bottom')
    plt.text(x=index[i]+2*bar_width, y=pred_sarcasm[i]+1, s=pred_sarcasm[i], ha='center', va='bottom')
    plt.text(x=index[i]+3*bar_width, y=pred_not_sarcasm[i]+1, s=pred_not_sarcasm[i], ha='center', va='bottom')


plt.xlabel('Face Emotion')
plt.ylabel('Example Number')
plt.title('Vision Information')
plt.xticks(index + bar_width, ('Happy', 'Angry', 'Neutral', 'Sad', 'Surprise', 'Fear', 'Disgust'))
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('bar_chart.png')
