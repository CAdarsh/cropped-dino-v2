from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torchvision.transforms as T
import os
import numpy as np

dataset_path = 'tiny-imagenet/tiny-imagenet-200/test/images/'

crop_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
crop_rcnn.eval()
crop_rcnn = crop_rcnn.cuda()
resize_transforms = T.Compose([
    T.Resize((64,64))
])
precropping_transforms = T.Compose([
    ResNet50_Weights.DEFAULT.transforms()
])

# Test set first
for file in os.listdir(dataset_path):
    if file.endswith('.JPEG'):
        print(file)
        image = torchvision.io.read_image(dataset_path + file)
        if (image.shape[0] == 1):
            continue
        image_processed = precropping_transforms(resize_transforms(image))
        image_processed = image_processed.cuda()
        pred = crop_rcnn([image_processed])
        
        largest_box = np.zeros((1,4))
        largest_area = 0
        flag = 0 
        if len(pred) != 0:
            for i in range(len(pred)):
                pred = pred[i]['boxes'].cpu().detach().numpy()
                if len(pred) == 0:
                    flag = 1
                    break
                area = (pred[i, 2] - pred[i, 0]) * (pred[i,3] - pred[i,1])
                if area > largest_area:
                    largest_box = pred[i]
                    largest_area = area

            if flag == 1:
                continue

        cropped_image = image_processed[:, int(largest_box[0]):int(largest_box[2]), int(largest_box[1]):int(largest_box[3])]
        cropped_image = resize_transforms(cropped_image)

        torchvision.utils.save_image(cropped_image, dataset_path + file)


# Val second
dataset_path = 'tiny-imagenet/tiny-imagenet-200/val/images/'
for file in os.listdir(dataset_path):
    if file.endswith('.JPEG'):
        print(file)
        image = torchvision.io.read_image(dataset_path + file)
        if (image.shape[0] == 1):
            continue
        image_processed = precropping_transforms(resize_transforms(image))
        image_processed = image_processed.cuda()
        pred = crop_rcnn([image_processed])
        
        largest_box = np.zeros((1,4))
        largest_area = 0
        
        if len(pred) != 0:
            for i in range(len(pred)):
                pred = pred[i]['boxes'].cpu().detach().numpy()
                if len(pred) == 0:
                    flag = 1
                    break
                area = (pred[i, 2] - pred[i, 0]) * (pred[i,3] - pred[i,1])
                if area > largest_area:
                    largest_box = pred[i]
                    largest_area = area
                
            if flag == 1:
                continue

        cropped_image = image_processed[:, int(largest_box[0]):int(largest_box[2]), int(largest_box[1]):int(largest_box[3])]
        cropped_image = resize_transforms(cropped_image)

        torchvision.utils.save_image(cropped_image, dataset_path + file)

# Train last
dataset_path = 'tiny-imagenet/tiny-imagenet-200/train/'
for directory in os.listdir(dataset_path):
    for file in os.listdir(dataset_path + '/' + directory + '/images'):
        print(file)
        image = torchvision.io.read_image(dataset_path + '/' + directory + '/images/' + file)
        if (image.shape[0] == 1):
            continue
        image_processed = precropping_transforms(resize_transforms(image))
        image_processed = image_processed.cuda()
        pred = crop_rcnn([image_processed])
        
        largest_box = np.zeros((1,4))
        largest_area = 0
        
        if len(pred) != 0:
            for i in range(len(pred)):
                pred = pred[i]['boxes'].cpu().detach().numpy()
                if len(pred) == 0:
                    flag = 1
                    break
                area = (pred[i, 2] - pred[i, 0]) * (pred[i,3] - pred[i,1])
                if area > largest_area:
                    largest_box = pred[i]
                    largest_area = area
                
            if flag == 1:
                continue

        cropped_image = image_processed[:, int(largest_box[0]):int(largest_box[2]), int(largest_box[1]):int(largest_box[3])]
        cropped_image = resize_transforms(cropped_image)

        torchvision.utils.save_image(cropped_image, dataset_path + '/' + directory + '/images/' + file)
