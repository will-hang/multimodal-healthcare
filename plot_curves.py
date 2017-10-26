import matplotlib.pyplot as plt
import numpy as np

exp_id = 3

ta = np.load('outputs/train_acc{}.npy'.format('_1'))
tl = np.load('outputs/train_loss{}.npy'.format('_1'))
va = np.load('outputs/val_acc{}.npy'.format('_1'))
vl = np.load('outputs/val_loss{}.npy'.format('_1'))

ta, = plt.plot(ta, label='Train acc')
tl, = plt.plot(tl, label='Train loss')
va, = plt.plot(va, label='Val acc')
vl, = plt.plot(vl, label='Val loss')
plt.legend(handles=[ta, tl, va, vl])

plt.show()