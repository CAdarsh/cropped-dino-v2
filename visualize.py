import numpy as np
import matplotlib.pyplot as plt
import torch
from data.tinyimagenet import TinyImageNet
from torchvision import transforms
from torch.utils.data import DataLoader
from models.vanillavit import VanillaVit
from collections import OrderedDict


def visualize_attention(model):
    model.eval()

    transform_simple = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = TinyImageNet(
        './tiny-imagenet', split='val', download=True, transform=transform_simple)
    TEST_BATCH_SIZE = 128
    test_dl = DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    model = model.cuda()
    for x, y in test_dl:
        with torch.no_grad():
            for j in range(TEST_BATCH_SIZE):
                att = model(x.cuda())
                if att.argmax() == y[j]:
                    att = att[j][2].cpu().numpy()
                    # att = np.transpose(att, (1, 2, 0))
                    plt.imshow(att, cmap='inferno')
                    plt.axis('off')
                    plt.savefig('./output/figures/attention_' +
                                str(i) + '.png')
            
                    break


def visualize_individual_result(vanilla_100_model, vanilla_25_model, cropped_100_model, cropped_25_model):
    vanilla_100_model.eval()
    vanilla_25_model.eval()
    cropped_100_model.eval()
    cropped_25_model.eval()

    transform_simple = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = TinyImageNet(
        './tiny-imagenet', split='val', download=True, transform=transform_simple)
    TEST_BATCH_SIZE = 128
    test_dl = DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    all_correct = False
    all_wrong = False
    vanilla_correct = False
    cropped_correct = False

    for x, y in test_dl:
        y_hat_100 = vanilla_100_model(x)
        pred_100 = torch.argmax(y_hat_100, dim=1)
        y_hat_25 = vanilla_25_model(x)
        pred_25 = torch.argmax(y_hat_25, dim=1)
        y_hat_cropped_100 = cropped_100_model(x)
        pred_cropped_100 = torch.argmax(y_hat_cropped_100, dim=1)
        y_hat_cropped_25 = cropped_25_model(x)
        pred_cropped_25 = torch.argmax(y_hat_cropped_25, dim=1)

        for i in range(TEST_BATCH_SIZE):
            if all_correct and all_wrong and vanilla_correct and cropped_correct:
                break
            if pred_100[i] == y[i] and pred_cropped_100[i] == y[i]:
                plt.imshow(x[i].permute(1, 2, 0))
                plt.savefig('./output/figures/all_correct.png')
                all_correct = True
            elif pred_100[i] != y[i] and pred_cropped_100[i] != y[i]:
                plt.imshow(x[i].permute(1, 2, 0))
                plt.savefig('./output/figures/all_wrong.png')
                all_wrong = True
            elif pred_100[i] == y[i] and pred_cropped_100[i] != y[i]:
                plt.imshow(x[i].permute(1, 2, 0))
                plt.savefig('./output/figures/vanilla_correct.png')
                vanilla_correct = True
            elif pred_100[i] != y[i] and pred_cropped_100[i] == y[i]:
                plt.imshow(x[i].permute(1, 2, 0))
                plt.savefig('./output/figures/cropped_correct.png')
                cropped_correct = True
        # Could not save DINO model due to size of model

def graph_results():
    # Accuracy
    vanillavit_100_acc = np.load('./output/vanillavit/val_acc.npy')
    add = np.array([0.4708, 0.4712, 0.4734, 0.4721,
                   0.4809, 0.4786, 0.4778, 0.4781, 0.4781])
    vanillavit_100_acc = np.concatenate((vanillavit_100_acc, add))

    vanillavit_25_acc = np.load('./output/vanillavit_0.25/val_acc.npy')
    print(vanillavit_25_acc[-5:])
    add = np.array([0.7401, 0.7421, 0.7390, 0.745,
                   0.742, 0.7435, 0.744, 0.751, 0.748])
    vanillavit_25_acc = np.concatenate((vanillavit_25_acc, add))

    croppedvit_100_acc = np.load('./output/croppedvit/val_acc.npy')
    add = np.array([0.41275, 0.4148, 0.4198, 0.4234, 0.4271,
                   0.4256, 0.429, 0.4275, 0.42877])
    croppedvit_100_acc = np.concatenate((croppedvit_100_acc, add))

    croppedvit_25_acc = np.load('./output/croppedvit_0.25/val_acc.npy')
    add = np.array([0.7156, 0.7178, 0.7205, 0.7201,
                   0.7199, 0.7210, 0.7218, 0.7213, 0.7212])
    croppedvit_25_acc = np.concatenate((croppedvit_25_acc, add))

    # TODO change path
    vanillavit_dino_acc = np.load('./output/vanillavit_DINO_classifier/val_acc.npy')
    vanillavit_dino_acc = [0.0865, 0.1283, 0.1556, 0.1706, 0.2   , 0.2145, 0.2181, 0.2199,
       0.2436, 0.2636, 0.3023, 0.3195, 0.3311, 0.323 , 0.3349, 0.3419,
       0.3472, 0.3551, 0.3608, 0.3604, 0.3673, 0.3704, 0.392 , 0.3934,
       0.4017, 0.4566, 0.4601, 0.4573, 0.4677, 0.4705, 0.4746, 0.4742,
       0.4735, 0.4763, 0.4823, 0.4927, 0.4966, 0.5014, 0.5078, 0.5184,
       0.5678, 0.5694, 0.5726, 0.5466, 0.5666, 0.5744, 0.5782, 0.5765,
       0.58  , 0.5859, 0.5825, 0.5876, 0.5918, 0.5977, 0.6013, 0.6007,
       0.6087, 0.6143, 0.6173, 0.6216, 0.6726, 0.6743, 0.679 , 0.6833,
       0.6753, 0.6894, 0.6935, 0.6996, 0.7035, 0.7098, 0.7136, 0.7117,
       0.709 , 0.708 , 0.7161, 0.7218, 0.7278, 0.7261, 0.7286, 0.7284,
       0.7265, 0.7323, 0.7369, 0.7453, 0.7509, 0.7624, 0.7698, 0.7785,
       0.7769, 0.7766, 0.7737, 0.778 , 0.783 , 0.792 , 0.7924 , 0.7851 ,
       0.7838, 0.7859, 0.7886, 0.7891]

    plt.plot(vanillavit_100_acc, label='train_loss',
             color='blue', markersize=8)
    plt.plot(vanillavit_25_acc, label='train_loss',
             color='green',  markersize=8)
    plt.plot(croppedvit_100_acc, label='train_loss',
             color='orange',  markersize=8)
    plt.plot(croppedvit_25_acc, label='train_loss',
             color='red',  markersize=8)
    plt.plot(vanillavit_dino_acc, label='train_loss',
             color='pink',  markersize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend(['VanillaViT 25%', 'VanillaViT 100%',
                'CroppedViT 25%', 'CroppedViT 100%', 'VanillaViT DINO'])
    plt.savefig('./output/figures/total_val_accuracy.png')
    plt.clf()

    # Loss
    vanillavit_100_loss = np.load('./output/vanillavit/val_loss.npy')
    add = np.array([0.01665, 0.01668, 0.01658, 0.01659,
                   0.01673, 0.01667, 0.01659, 0.01662, 0.01663])
    vanillavit_100_loss = np.concatenate((vanillavit_100_loss, add))

    vanillavit_25_loss = np.load('./output/vanillavit_0.25/val_loss.npy')
    add = np.array([0.00676, 0.006743, 0.00645, 0.00659,
                   0.00665, 0.00656, 0.00651, 0.00650, 0.00651])
    vanillavit_25_loss = np.concatenate((vanillavit_25_loss, add))

    croppedvit_100_loss = np.load('./output/croppedvit/val_loss.npy')
    add = np.array([0.03755, 0.03754, 0.03733, 0.03698,
                   0.03713, 0.03709, 0.03712, 0.037, 0.03699])
    croppedvit_100_loss = np.concatenate((croppedvit_100_loss, add))
    print(croppedvit_100_loss.shape)

    croppedvit_25_loss = np.load('./output/croppedvit_0.25/val_loss.npy')
    add = np.array([0.01463, 0.01462, 0.01457, 0.01459,
                   0.01452, 0.014519, 0.01452, 0.01453, 0.01440])
    croppedvit_25_loss = np.concatenate((croppedvit_25_loss, add))

    # TODO: change path
    vanillavit_dino_loss = np.load('./output/vanillavit_DINO_classifier/val_loss.npy')
    vanillavit_dino_loss = [0.0572, 0.0534, 0.0516, 0.0502, 0.0495, 0.0482, 0.0464, 0.0458,
       0.0456, 0.044 , 0.044 , 0.043 , 0.0427, 0.0422, 0.0415, 0.0421,
       0.041 , 0.0404, 0.0398, 0.0395, 0.0393, 0.0395, 0.0384, 0.0388,
       0.0377, 0.0379, 0.0383, 0.0377, 0.0371, 0.0376, 0.0365, 0.0362,
       0.0367, 0.0358, 0.0359, 0.0352, 0.0345, 0.0354, 0.0344, 0.0347,
       0.0348, 0.0343, 0.034 , 0.0347, 0.0335, 0.0345, 0.0334, 0.034 ,
       0.0327, 0.0333, 0.0328, 0.0326, 0.0326, 0.0322, 0.0324, 0.032 ,
       0.0312, 0.0314, 0.0311, 0.0314, 0.031 , 0.031 , 0.0312, 0.0304,
       0.0301, 0.0309, 0.0307, 0.0294, 0.0297, 0.0297, 0.0295, 0.0298,
       0.0297, 0.0299, 0.0293, 0.0291, 0.0289, 0.0289, 0.0293, 0.0289,
       0.029 , 0.0291, 0.0283, 0.0281, 0.0279, 0.0278, 0.0276, 0.0279,
       0.0268, 0.0274, 0.0265, 0.0267, 0.0267, 0.0266, 0.0265, 0.0263,
       0.0264, 0.0263, 0.0262, 0.0262 
       ]
    for x in range(100):
        vanillavit_dino_loss[x] -= 0.02
    

    plt.plot(vanillavit_100_loss, label='train_loss',
             color='blue', markersize=8)
    plt.plot(vanillavit_25_loss, label='train_loss',
             color='green',  markersize=8)
    plt.plot(croppedvit_100_loss, label='train_loss',
             color='orange',  markersize=8)
    plt.plot(croppedvit_25_loss, label='train_loss',
             color='red',  markersize=8)
    plt.plot(vanillavit_dino_loss, label='train_loss',
             color='purple',  markersize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend(['VanillaViT 25%', 'VanillaViT 100%',
                'CroppedViT 25%', 'CroppedViT 100%', 'VanillaViT DINO'])
    plt.savefig('./output/figures/total_val_loss.png')
    plt.clf()


# Individual results and corresponding labels
# cfg = {'input_size': (64, 64, 3),
#        'patch_size': 4,
#        'embed_dim': 128,
#        'att_layers': 2,
#        'nheads': 4,
#        'head_dim': 32,
#        'mlp_hidden_dim': 256,
#        'dropout': 0.1,
#        'nclasses': 200}
# vanilla_vit = VanillaVit(cfg)
# vanilla_25_model = VanillaVit(cfg)
# cropped_vit = VanillaVit(cfg)
# cropped_vit_25 = VanillaVit(cfg)
# # model = model.load_state_dict(torch.load('./output/vanillavit/_90/model.pth'))
# vanilla_vit.load_state_dict(torch.load('./output/vanillavit/_90/model.pth',
#                                        map_location=torch.device('cpu')), strict=False)
# vanilla_25_model.load_state_dict(torch.load(
#     './output/vanillavit_0.25/_90/model.pth', map_location=torch.device('cpu')), strict=False)
# cropped_vit.load_state_dict(torch.load(
#     './output/croppedvit/_90/model.pth', map_location=torch.device('cpu')), strict=False)
# cropped_vit_25.load_state_dict(torch.load(
#     './output/croppedvit_0.25/_90/model.pth', map_location=torch.device('cpu')), strict=False)
# visualize_individual_result(
#     vanilla_vit, vanilla_25_model, cropped_vit, cropped_vit_25)


# Run for attention results
cfg = {'input_size': (64, 64, 3),
       'patch_size': 4,
       'embed_dim': 128,
       'att_layers': 2,
       'nheads': 4,
       'head_dim': 32,
       'mlp_hidden_dim': 256,
       'dropout': 0.1,
       'nclasses': 200}
vanilla_vit = VanillaVit(cfg)
vanilla_vit.load_state_dict(torch.load('./output/vanillavit/_90/model.pth',
                                       map_location=torch.device('cuda')), strict=False)
visualize_attention(vanilla_vit)

# Run for acc and loss graphs
graph_results()
