import matplotlib.pyplot as plt
import numpy as np

ta = np.load('outputs/train_acc.npy')
tl = np.load('outputs/train_loss.npy')
va = np.load('outputs/val_acc.npy')
vl = np.load('outputs/val_loss.npy')

ta, = plt.plot(ta, label='Train acc')
tl, = plt.plot(tl, label='Train loss')
va, = plt.plot(va, label='Val acc')
vl, = plt.plot(vl, label='Val loss')
plt.legend(handles=[ta, tl, va, vl])

plt.show()