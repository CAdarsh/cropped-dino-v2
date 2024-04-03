import time as time
import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, Subset
from data.tinyimagenet import TinyImageNet
from torchvision import transforms
from models.vanillavit import VanillaVit
from models.croppedvit import CroppedVit
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torchvision.transforms as T
from utils.save_logs import save


def train(model, optimizer, criterion, train_dl, val_dl, type, epochs=50, lr=0.001, num_classes=200, gpu=True, verbose=True):
    train_loss_store = []
    train_acc_store = []
    val_loss_store = []
    val_acc_store = []

    if gpu:
        device = torch.device('cuda')
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    
    for epoch in range(epochs):
        # Training
        start = time.time()
        model.train()
        train_loss = 0
        num_correct_train = 0
        num_correct_val = 0
        total_tr = 0
        for x, y in train_dl:
            optimizer.zero_grad()
            if gpu:
                x, y = x.cuda(), y.cuda()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            # Normalize y_hat within num_classes
            y_hat = (y_hat - torch.min(y_hat)) / \
                (torch.max(y_hat) - torch.min(y_hat)) * num_classes
            num_correct_train += torch.sum(torch.argmax(y_hat, dim=1) == y)
            train_loss += loss.item()
            total_tr += y.size(0)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        total_val = 0
        for x, y in val_dl:
            if gpu:
                x, y = x.cuda(), y.cuda()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            # Normalize y_hat within num_classes
            y_hat = (y_hat - torch.min(y_hat)) / \
                (torch.max(y_hat) - torch.min(y_hat)) * num_classes
            num_correct_val += torch.sum(torch.argmax(y_hat, dim=1) == y)
            val_loss += loss.item()
            total_val += y.size(0)
        
        end = time.time()

        train_loss_store.append((train_loss / total_tr))
        train_acc_store.append((num_correct_train.cpu().numpy() / total_tr))
        val_loss_store.append((val_loss / total_val))
        val_acc_store.append((num_correct_val.cpu().numpy() / total_val))

        if verbose:
            print(
                f'Epoch: {epoch} | Training Loss: {train_loss / total_tr} | Training Acc: {num_correct_train / total_tr} | Val Loss: {val_loss / total_val} | Val Acc: {num_correct_val / total_val} | Time: {end - start}')
        if epoch % 10 == 0:
            save(type, train_loss_store, train_acc_store, val_loss_store, val_acc_store, model, epoch)

    train_loss_store = np.asarray(train_loss_store)
    train_acc_store = np.asarray(train_acc_store)
    val_loss_store = np.asarray(val_loss_store)
    val_acc_store = np.asarray(val_acc_store)

    save(type, train_loss_store, train_acc_store, val_loss_store, val_acc_store, epoch)

    best_epoch = np.argmax(val_acc_store)
    print("----------")
    print("Best Model Stats - Epoch", best_epoch)
    print("----------")
    print("Train Acc: ", train_acc_store[best_epoch])
    print("Val Acc: ", val_acc_store[best_epoch])


def test(model, criterion, test_dl, num_classes=200, gpu=True):
    model.eval()
    test_loss = 0
    num_correct = 0
    total = 0

    for x, y in test_dl:
        if gpu:
            x, y = x.cuda(), y.cuda()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        test_loss += loss.item()
        # Normalize y_hat within num_classes
        y_hat = (y_hat - torch.min(y_hat)) / \
            (torch.max(y_hat) - torch.min(y_hat)) * num_classes
        num_correct += torch.sum(torch.argmax(y_hat, dim=1) == y)
        total += y.size(0)

    print(
        f'Test Loss: {test_loss / total} | Test Acc: {num_correct / total}')


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Train a model on Tiny ImageNet')
    parser.add_argument('--model', metavar='N', type=str,
                        default='vanillavit', help='model to train')
    parser.add_argument('--amount', metavar='N', type=float,
                        help='amount of data to use', default=1.0)
    parser.add_argument('--epochs', metavar='N', type=int,
                        help='number of epochs', default=50)
    parser.add_argument('--lr', metavar='N', type=float,
                        help='learning rate', default=0.001)
    parser.add_argument('--gpu', help='use gpu')
    parser.add_argument('--batch_size', metavar='N', type=int, default=128)
    parser.add_argument(
        '--verbose', help='print the training and validation loss and accuracy')
    args = parser.parse_args()

    epochs = args.epochs
    lr = args.lr
    gpu = True if args.gpu == 'True' else False
    verbose = args.verbose
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = args.batch_size

    if args.model == 'vanillavit':
        cfg = {'input_size': (64, 64, 3),
               'patch_size': 4,
               'embed_dim': 128,
               'att_layers': 2,
               'nheads': 4,
               'head_dim': 32,
               'mlp_hidden_dim': 256,
               'dropout': 0.1,
               'nclasses': 200}
        model = VanillaVit(cfg)

    elif args.model == 'croppedvit':
        cfg = {'input_size': (64, 64, 3),
               'patch_size': 4,
               'embed_dim': 128,
               'att_layers': 2,
               'nheads': 4,
               'head_dim': 32,
               'mlp_hidden_dim': 256,
               'dropout': 0.1,
               'nclasses': 200}
        model = VanillaVit(cfg)
    else:
        # TODO: add other models once implemented
        raise NotImplementedError 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    TRAIN_BATCH_SIZE = batch_size
    VAL_BATCH_SIZE = batch_size
    TEST_BATCH_SIZE = batch_size

    # Download dataset
    # TODO: add data augmentation
    transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = TinyImageNet(
        './tiny-imagenet', split='train', download=True, transform=transforms)
    test_dataset = TinyImageNet(
        './tiny-imagenet', split='val', download=True, transform=transforms)
    if args.amount < 1.0:
        train_dataset = Subset(train_dataset, list(range(0, int(args.amount * len(train_dataset.data)))))
        args.model = args.model + '_' + str(args.amount)
    # Data Loaders
    train_dl = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(train_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    train(model, optimizer, criterion, train_dl, val_dl, args.model,
          epochs=epochs, lr=lr, gpu=gpu, verbose=verbose)
    test(model, criterion, test_dl, num_classes=200, gpu=gpu)