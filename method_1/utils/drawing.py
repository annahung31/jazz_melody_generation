import numpy as np
from matplotlib import pyplot as plt




def draw_loss(lossD,lossG,loss_path):

    lossD_print = np.load(loss_path + lossD)
    lossG_print = np.load(loss_path + lossG)

    length = lossG_print.shape[0]

    x = np.linspace(0, length-1, length)
    x = np.asarray(x)
    plt.figure()
    plt.plot(x, lossD_print,label=' lossD',linewidth=1.5)
    plt.plot(x, lossG_print,label=' lossG',linewidth=1.5)

    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(loss_path +'lossG_D.png')

def draw_predict(D_x, D_G_z, loss_path):
    D_x_print = np.load(loss_path + D_x)
    D_G_z_print = np.load(loss_path + D_G_z) 
    length = D_x_print.shape[0]

    x = np.linspace(0, length-1, length)
    x = np.asarray(x)
    plt.figure()
    plt.plot(x, D_x_print,label=' D(real)',linewidth=1.5)
    plt.plot(x, D_G_z_print,label=' D(generate)',linewidth=1.5)

    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('prediction')
    plt.savefig(loss_path +'prediction.png')

