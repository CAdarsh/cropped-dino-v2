
import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def save_dino(type, train_loss_store, val_loss_store, model, epoch):
    val_loss_store = np.asarray(val_loss_store)
    train_loss_store = np.asarray(train_loss_store)
    type = type + '_pretrain'
    if not os.path.exists('./output/' + type):
        os.makedirs('./output/' + type)

    np.save('./output/' + type + '/val_loss.npy', val_loss_store)
    np.save('./output/' + type + '/test_loss.npy', train_loss_store)

    if not os.path.exists('./output/' + type + '/_' + str(epoch)):
        os.makedirs('./output/' + type + '/_' + str(epoch))

    torch.save(model.state_dict(), './output/' + type + '/_' + str(epoch) + '/model.pth')

    plt.plot(train_loss_store, '-o', label='train_loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss at the end of each Epoch')
    plt.savefig('./output/' + type + '/_' + str(epoch) + '/train_loss.png')
    plt.clf()

    plt.plot(val_loss_store, '-o', label='val_loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss at the end of each Epoch')
    plt.savefig('./output/' + type + '/_' + str(epoch) + '/val_loss.png')
    plt.clf()
    

def save(type, train_loss_store, train_acc_store, val_loss_store, val_acc_store, model, epoch):
    
    train_loss_store = np.asarray(train_loss_store)
    train_acc_store = np.asarray(train_acc_store)
    val_loss_store = np.asarray(val_loss_store)
    val_acc_store = np.asarray(val_acc_store)

    if not os.path.exists('./output/' + type):
        os.makedirs('./output/' + type)

    np.save('./output/' + type + '/train_loss.npy', train_loss_store)
    np.save('./output/' + type + '/train_acc.npy', train_acc_store)
    np.save('./output/' + type + '/val_loss.npy', val_loss_store)
    np.save('./output/' + type + '/val_acc.npy', val_acc_store)

    if not os.path.exists('./output/' + type + '/_' + str(epoch)):
        os.makedirs('./output/' + type + '/_' + str(epoch))

    torch.save(model.state_dict(), './output/' + type + '/_' + str(epoch) + '/model.pth')

    plt.plot(train_loss_store, '-o', label='train_loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss at the end of each Epoch')
    plt.savefig('./output/' + type + '/_' + str(epoch) + '/train_loss.png')
    plt.clf()
    plt.plot(train_acc_store, '-o', label='train_acc', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy at the end of each Epoch')
    plt.savefig('./output/' + type + '/_' + str(epoch) + '/train_acc.png')
    plt.clf()

    plt.plot(val_loss_store, '-o', label='val_loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss at the end of each Epoch')
    plt.savefig('./output/' + type + '/_' + str(epoch) + '/val_loss.png')
    plt.clf()
    plt.plot(val_acc_store, '-o', label='val_acc', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy at the end of each Epoch')
    plt.savefig('./output/' + type + '/_' + str(epoch) + '/val_acc.png')
    plt.clf()