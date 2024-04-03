import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, Subset
from data.tinyimagenet import TinyImageNet
from torchvision import transforms
from models.vanillavit import VanillaVit
from models.croppedvit  import CroppedVit
from DINO.dino_py import Dino
import matplotlib.pyplot as plt
import time as time
import os
from torch import nn
from utils.save_logs import save, save_dino
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def clean_up():
    dist.destroy_process_group()

def dataset(rank, world_size, batch_size, amount):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = TinyImageNet(
        './tiny-imagenet', split='train', download=True, transform=transform)
    if amount < 1.0:
        train_dataset = Subset(train_dataset, list(range(0, int(amount * len(train_dataset.data)))))
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) 
    # Data Loaders
    train_dl = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=False)
    val_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0, pin_memory=False)
    return train_dl, val_dl

def train_dino(rank, world_size, batch_size, amount, learner, model, optimizer, type, epochs=50, lr=0.001, num_classes=200, gpu=True, verbose=True):
        train_loss_store = []
        val_loss_store = []
        
        setup(rank, world_size)

        train_dl, val_dl = dataset(rank, world_size, batch_size, amount)

        if gpu:
            model.to(rank)
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
            learner.to(rank)
            learner = DDP(learner, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        print("------- DINO Training starting -------")

        for epoch in range(epochs):
            train_dl.sampler.set_epoch(epoch)      
            val_dl.sampler.set_epoch(epoch) 
            # Training
            start = time.time()
            learner.train()
            train_loss = 0
            total_tr = 0
            for index, (x, y) in enumerate(train_dl):
                if gpu:
                    x, y = x.cuda(), y.cuda()
                images = x
                loss = learner(images)
                optimizer.zero_grad()
                train_loss += loss.item()
                total_tr += x.size(0)
                loss.backward()
                optimizer.step()
                learner.module.update_moving_average() # update moving average of teacher encoder and teacher centers
            
            # Validation
            learner.eval()
            val_loss = 0
            total_val = 0
            for index, (x, y) in enumerate(val_dl):
                if gpu:
                    x, y = x.cuda(), y.cuda()
                images = x
                # print("Passing image through learner")
                loss = learner(images)
                optimizer.zero_grad()
                val_loss += loss.item()
                total_val += x.size(0)
                
            end = time.time()

            train_loss_store.append(train_loss / total_tr)
            val_loss_store.append(val_loss / total_val)

            if verbose:
                print(
                    f'Epoch: {epoch} | Training Loss: {train_loss / total_tr} | Val Loss: {val_loss / total_val} | Time: {end - start}')

            if epoch % 10 == 0:
                save_dino(type, train_loss_store, val_loss_store, learner, epoch)
               
        save_dino(type, train_loss_store, val_loss_store, learner, epoch)
        train_loss_store = np.asarray(train_loss_store)
        val_loss_store = np.asarray(val_loss_store)
        clean_up()
        

def train_dino_classifier(student_model, classifier_head, optimizer, criterion, train_dl, val_dl, type, epochs=50, lr=0.001, num_classes=200,
 gpu=True, verbose=True):
    

    train_loss_store = []
    train_acc_store = []
    val_loss_store = []
    val_acc_store = []

    if gpu: 
        device = torch.device('cuda')
        student_model = student_model.to(device)
        classifier_head = classifier_head.to(device)
    
    print("------ DINO Classifier Training Starting... -------")
    for epoch in range(epochs):
        start = time.time()
        classifier_head.train()
        train_loss = 0
        num_correct_train = 0
        num_correct_val = 0
        total_tr = 0
        for x, y in train_dl:
            optimizer.zero_grad()
            y_hat = None
            if gpu:
                x, y = x.cuda(), y.cuda()
            
          
            x = student_model(x)[0]
            y_hat = classifier_head(x)
            
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
        classifier_head.eval()
        val_loss = 0
        total_val = 0
        for x, y in val_dl:
            if gpu:
                x, y = x.cuda(), y.cuda()
            
            x = student_model(x)[0]
            y_hat = classifier_head(x)
            
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
            save(type+"_classifier", train_loss_store, train_acc_store, val_loss_store, val_acc_store, classifier_head, epoch)
            save(type + "_student", train_loss_store, train_acc_store, val_loss_store, val_acc_store, student_model, epoch)
    
    save(type + "_student", train_loss_store, train_acc_store, val_loss_store, val_acc_store, student_model, epoch)
    save(type + "_classifier", train_loss_store, train_acc_store, val_loss_store, val_acc_store, classifier_head, epoch)

    best_epoch = np.argmax(val_acc_store)
    print("----------")
    print("Best Model Stats - Epoch", best_epoch)
    print("----------")
    print("Train Acc: ", train_acc_store[best_epoch])
    print("Val Acc: ", val_acc_store[best_epoch])

def test_dino_classifer(student_model, classifier_head, criterion, test_dl, num_classes=200, gpu=True):

    student_model.eval()
    classifier_head.eval()
    test_loss = 0
    num_correct = 0
    total = 0

    if gpu:
        device = torch.device('cuda')
        student_model = student_model.to(device)
        classifier_head = classifier_head.to(device)

    print("------ DINO Classifier Testing Starting... -------")
   
    for x, y in test_dl:
        if gpu:
            x, y = x.cuda(), y.cuda()
        
        x = student_model(x)[0]
        y_hat = classifier_head(x)
        
        loss = criterion(y_hat, y)
        # Normalize y_hat within num_classes
        y_hat = (y_hat - torch.min(y_hat)) / \
            (torch.max(y_hat) - torch.min(y_hat)) * num_classes
        num_correct += torch.sum(torch.argmax(y_hat, dim=1) == y)
        test_loss += loss.item()
        total += y.size(0)
    print(
        f'Test Loss: {test_loss / total} | Test Acc: {num_correct / total}')



if __name__ == "__main__":
    
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Train a DINO model on Tiny ImageNet')
    parser.add_argument('--model', metavar='N', type=str,
                        default='vanillavit', help='model to train')
    parser.add_argument('--amount', metavar='N', type=float,
                        help='amount of data to use', default=1.0)
    parser.add_argument('--epochs', metavar='N', type=int,
                        help='number of epochs for DINO', default=50)
    parser.add_argument('--classifier_lr', metavar='N', type=float,
                        help='classifier learning rate', default=0.005)
    parser.add_argument('--lr', metavar='N', type=float,
                        help='learning rate', default=0.001)
    parser.add_argument('--gpu', help='use gpu')
    parser.add_argument(
        '--verbose', help='print the training and validation loss and accuracy')
    parser.add_argument(
        '--batch_size', help='batch size', type=int, default=128)
    parser.add_argument(
        '--classifier_epochs', help='Classifier Epochs', type=int, default=128)
    
    world_size = 2

    args = parser.parse_args()
    gpu = True if args.gpu == 'True' else False
    verbose = args.verbose
    # DINO Epochs
    epochs = args.epochs
    classifier_epochs = args.classifier_epochs
    lr = args.lr
    classifier_lr = float(args.classifier_lr)
    amount = args.amount

    if args.model == 'vanillavit':
        cfg = {'input_size': (64, 64, 3),
                    'patch_size': 4,
                    'embed_dim': 128,
                    'att_layers': 2,
                    'nheads': 4,
                    'head_dim': 32,
                    'mlp_hidden_dim': 256,
                    'dropout': 0.1,
                    'nclasses': 200
                    }

        model = VanillaVit(cfg)
        
    elif args.model == "croppedvit":
        cfg = {'input_size': (64, 64, 3),
               'patch_size': 4,
               'embed_dim': 128,
               'att_layers': 2,
               'nheads': 4,
               'head_dim': 32,
               'mlp_hidden_dim': 256,
               'dropout': 0.1,
               'nclasses': 200}
        model = CroppedVit(cfg)
        
    print("Initialised Model")

    learner = Dino(
        model,
        image_size = 256,
        # hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
        projection_hidden_size = 256,      # projector network hidden dimension
        projection_layers = 4,             # number of layers in projection network
        num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
        student_temp = 0.9,                # student temperature
        teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
        local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
        global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
        moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
        center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
    )

    print("Initialised DINO")
    opt = torch.optim.Adam(learner.parameters(), lr = lr)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = args.batch_size

    args.model = args.model + '_DINO'

    mp.spawn(
        train_dino,
        args=(world_size, batch_size, amount, learner, model, opt, args.model, epochs, lr, gpu, verbose),
        nprocs=world_size
    )

    #train_dino(learner, model, opt, train_dl, val_dl, args.model, epochs=epochs, lr=lr, gpu=gpu, verbose=verbose)
    
    # construct student model from learner model
    student_model = nn.Sequential(
        learner.augment1,
        learner.augment2,
        learner.local_crop,
        learner.global_crop,
        learner.student_encoder,
    )
    
    # initialise classifier head 
    classifier_head = nn.Sequential(
        nn.Linear(in_features=65336, out_features=500),
        nn.Linear(in_features=500, out_features=200)
    )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = TinyImageNet(
        './tiny-imagenet', split='train', download=True, transform=transform)
    test_dataset = TinyImageNet(
            './tiny-imagenet', split='val', download=True, transform=transform)
    if amount < 1.0:
        train_dataset = Subset(train_dataset, list(range(0, int(amount * len(train_dataset.data)))))
    # Data Loaders
    train_dl = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=True)
    val_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=True)
    test_dl = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False)
    
    opt = torch.optim.Adam(classifier_head.parameters(), lr = classifier_lr)

    train_dino_classifier(student_model, classifier_head, opt, criterion, train_dl, val_dl, args.model, classifier_epochs, classifier_lr, 200, gpu, verbose)
    test_dino_classifer(student_model, classifier_head, criterion, test_dl, 200, gpu)
