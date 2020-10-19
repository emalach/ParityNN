import numpy as np
import pickle
import matplotlib.pyplot as plt

legend = {'ReLU': 'ReLU network', 'GaLU': 'NTK regime', 'RF_ReLU': 'ReLU features', 'RF_gaussian': 'Gaussian features'}

plt.figure()
k = 3
for architecture in ['ReLU', 'GaLU', 'RF_gaussian', 'RF_ReLU']:
    scores = {}
    with open('experiment_k%d_q512_%s_20.pickle' % (k,architecture), 'rb') as f:
        scores[architecture] = pickle.load(f)

    plt.plot(scores[architecture], label=legend[architecture])

plt.title('k=%d' % k)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()

import matplotlib2tikz
matplotlib2tikz.save("parity_k%d.tex" % k)
