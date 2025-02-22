import random
import numpy as np
import cv2
import albumentations as A
from PIL import Image
from skimage import transform
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torchvision.transforms as T


def image_pipeline(info, test_mode, augment):
    path = info['path']
    #image_raw = Image.open(path)

    image = cv2.imread(path)
    if image is None:
        raise OSError('{} is not found'.format(path))
    # If you read with cv2 you need to convert from BGR to RGB space.
    image = np.array(image)
    image = image[:, :, ::-1]

    # align the face image if the source and target landmarks are given
    src_landmark = info.get('src_landmark')
    tgz_landmark = info.get('tgz_landmark')
    crop_size = info.get('crop_size')

    if not (src_landmark is None or tgz_landmark is None or crop_size is None):
        tform = transform.SimilarityTransform()
        tform.estimate(tgz_landmark, src_landmark)
        M = tform.params[0:2, :]
        image = cv2.warpAffine(image, M, crop_size, borderValue=0.0)

    # if not test_mode:
    #     if random.random() > 0.5:
    #         torch_transforms = T.Compose([T.Resize(size=[112, 112]),
    #                                       T.ColorJitter(contrast=0.8, brightness=0.8)])
    #
    #     else:
    #         torch_transforms = T.Compose([T.Resize([112, 112])])
    #
    #     # We are still in PIL land:
    #     while True:
    #         image = torch_transforms(image_raw)
    #         min_val = min([item[0] for item in image.getextrema()])
    #         max_val = max([item[1] for item in image.getextrema()])
    #         if min_val != max_val:
    #             break

    if not test_mode:
        if augment:
            album_transform = A.Compose([A.Resize(112, 112),
                                         A.ColorJitter(always_apply=False, p=0.5),
                                         A.GaussianBlur(blur_limit=(1, 7), always_apply=False, p=0.5),
                                         A.AdvancedBlur(blur_limit=(1, 7), rotate_limit=90,
                                                        beta_limit=(0.5, 8), noise_limit=(0.9, 1.1),
                                                        always_apply=False, p=0.5),
                                         A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None,
                                                         min_height=None, min_width=None, fill_value=0, always_apply=False,
                                                         p=0.5)])
        else:
            album_transform = A.Compose([A.Resize(112, 112),
                                         A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None,
                                                         min_height=None, min_width=None, fill_value=0, always_apply=False,
                                                         p=0.5)])
    else:
        album_transform = A.Compose([A.Resize(112, 112)])

    # Convert from PIL to numpy:
    image = album_transform(image=np.array(image))["image"]

    # This is needed to deal to convert the numpy array to Pytorch tensor where channel is first:
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)

    # normalize to [-1, 1]
    image = ((image - 127.5) / 127.5)
    #image = (image - image.min(axis=(0, 1))) / (image.max(axis=(0, 1)) - image.min(axis=(0, 1)))
    #image = (image*2)-1


    if not test_mode and random.random() > 0.5:
        image = np.flip(image, axis=2).copy()
    return image

def get_metrics(labels, scores, FPRs):
    # eer and auc
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    roc_curve = interp1d(fpr, tpr)
    EER = 100. * brentq(lambda x : 1. - x - roc_curve(x), 0., 1.)
    AUC = 100. * metrics.auc(fpr, tpr)

    # get acc
    tnr = 1. - fpr
    pos_num = labels.count(1)
    neg_num = labels.count(0)
    ACC = 100. * max(tpr * pos_num + tnr * neg_num) / len(labels)

    # TPR @ FPR
    if isinstance(FPRs, list):
        TPRs = [
            ('TPR@FPR={}'.format(FPR), 100. * roc_curve(float(FPR)))
            for FPR in FPRs
        ]
    else:
        TPRs = []

    return [('ACC', ACC), ('EER', EER), ('AUC', AUC)] + TPRs
