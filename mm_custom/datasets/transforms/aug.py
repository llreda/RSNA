import numpy as np
import random


from typing import Dict, List, Optional, Tuple, Union
from mmcv.transforms.base import BaseTransform
from mmcv.transforms import to_tensor


from mm_custom.registry import TRANSFORMS

import albumentations as A
import cv2
import mmcv
from mmcv.transforms.utils import cache_randomness

import torch
import torch.nn.functional as F




@TRANSFORMS.register_module()
class ResizeMaskDic(BaseTransform):
    def __init__(self, scale) -> None:
        super().__init__()
        self.scale = scale   # d h w


    def resize_data(self, data):
        data = torch.tensor(data).unsqueeze(0).unsqueeze(0).float()
        data = F.interpolate(data, size=self.scale, mode='trilinear')
        data = data.squeeze(0).squeeze(0).numpy()	
        return data


    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:

        mask_dic = results['mask_dic']

        for name, mask in mask_dic.items():
            mask = self.resize_data(mask)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = mask.astype(np.uint8)
            mask_dic[name] = mask


        results['mask_dic'] = mask_dic

        return results


@TRANSFORMS.register_module()
class ResizeImgs(BaseTransform):
    def __init__(self, scale, in_channel) -> None:
        super().__init__()
        self.scale = scale
        self.in_channel = in_channel



    def resize_data(self, data):

        data = torch.tensor(data).unsqueeze(0).float()
        data = F.interpolate(data, size=self.scale[1:], mode='bilinear')
        data = data.squeeze(0).numpy()	
        return data


    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:

        imgs = results['img']  # TC H W -> TC H2 W2 -> T H2 W2 C

        TC, H, W = imgs.shape
        assert TC == self.in_channel * self.scale[0]

        imgs = self.resize_data(imgs)

        imgs = imgs.reshape(self.scale[0], self.in_channel, self.scale[1], self.scale[2])
        imgs = imgs.transpose(0, 2, 3, 1) # [T, C, H, W] -> [T, H, W, C]

        results['img'] = imgs

        return results



@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    def __init__(self) -> None:
        super().__init__()


    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:

        img = results['img']

    
        assert len(img.shape) == 4

        d, h, w, c = img.shape
               
        if random.random() > 0.5:
            img = np.flip(img, axis=2)

            for k, v in results['mask_dic'].items():
                results['mask_dic'][k] = np.ascontiguousarray(np.flip(v, axis=2))


        results['img'] = np.ascontiguousarray(img)

        return results



@TRANSFORMS.register_module()
class DilateMask(BaseTransform):
    def __init__(self, kernel_size=(3, 3), iterations=2) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.iterations = iterations
        self.kernel = np.ones(kernel_size, np.uint8)


    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:

        for k, v in results['mask_dic'].items():
            mask = results['mask_dic'][k]
            aug_mask = np.zeros(mask.shape) # [clip_num, h, w]
            
            for i in range(mask.shape[0]):
                aug_mask[i] = cv2.dilate(mask[i], kernel=self.kernel, iterations=self.iterations)
                
            results['mask_dic'][k] = aug_mask


        return results


@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Rotate the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    @cache_randomness
    def generate_degree(self):
        return np.random.rand() < self.prob, np.random.uniform(
            min(*self.degree), max(*self.degree))

    def transform(self, results: dict) -> dict:
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        img = results['img']

        assert len(img.shape) == 4

        d, h, w, c = img.shape
        assert c == 1 or c == 3


        for i in range(d):
            rotate, degree = self.generate_degree()
            
            if rotate:
                # rotate image
                img[i] = mmcv.imrotate(
                    img[i],
                    angle=degree,
                    border_value=self.pal_val,
                    center=self.center,
                    auto_bound=self.auto_bound
                )

                # rotate mask
                for k, v in results['mask_dic'].items():
                    results['mask_dic'][k][i] =  mmcv.imrotate(
                        results['mask_dic'][k][i],
                        angle=degree,
                        border_value=self.seg_pad_val,
                        center=self.center,
                        auto_bound=self.auto_bound,
                        interpolation='nearest'
                    )

        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str



def find_first_over_target(sequence, target):
    l, r = (int)(0), (int)(len(sequence) - 1)
    while l <=r :
        mid = (l + r) // 2
        if sequence[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    
    if l == len(sequence):
        l = -1
    return l


def findbound(img, pixel_thresh, ratio_thresh):
    """
    输入一个图像，可以是三通道，也可以是单通道，找h 和 w方向的边界，用有效像素数来判定
    可以先阈值化，然后求cumsum，四个方向的cumsum，当某边的有效像素不足全体有效像素的5%，就排除
    """
    if len(img.shape) == 2:
        img = img[...,None]
    
    H, W, C = img.shape
    masks = (img > pixel_thresh).astype(np.uint8)
    total_mask = np.zeros(img.shape[:2]).astype(np.uint8)
    for i in range(C):
        total_mask += masks[:,:, i]
    
    total_mask[total_mask> 0] = 1  # 有效位置为1， 其他为0， 合并三个切片
    cnts_per_col = np.sum(total_mask, axis=0)
    cnts_per_row = np.sum(total_mask, axis=1)
    
    left2right = np.cumsum(cnts_per_col, axis=0)
    top2bottom = np.cumsum(cnts_per_row, axis=0)
    
    right2left = np.cumsum(np.flip(cnts_per_col), axis=0)
    bottom2top = np.cumsum(np.flip(cnts_per_row), axis=0)
    
    total_valid = left2right[-1]
    cnt_thresh = total_valid * ratio_thresh
    
    left_bound = find_first_over_target(left2right, cnt_thresh)
    right_bound = W - 1 - find_first_over_target(right2left, cnt_thresh)
    top_bound = find_first_over_target(top2bottom, cnt_thresh)
    bottom_bound = H - 1 - find_first_over_target(bottom2top, cnt_thresh)
    x, y, w, h = left_bound, top_bound, right_bound - left_bound + 1, bottom_bound - top_bound + 1
    
    return (x, y, w, h)


def get_crop_box(img, min_area_thres, min_ratio_thres, max_ratio_thres, mean_ratio, pixel_thresh, ratio_thresh, padding_size):
    # 当box的宽高比很小或很大时，（假设太窄了），那么就按照高度和平均宽高比得到最安全的宽度，进行裁剪
    # 直接设置min max 都是mean_ratio的话，就可以强制使所有的crop都保持相同的宽高比
    # 
    assert len(img.shape) == 3
    img_height, img_width, channels = img.shape

    
    (x, y, w, h) = findbound(img, pixel_thresh, ratio_thresh)
    x, y, w, h = x - padding_size[0], y - padding_size[1], w + 2 * padding_size[0], h + 2 * padding_size[1]
    x, y, w, h = max(0, x), max(0, y), min(w, img_width), min(h, img_height)
    
    crop_width, crop_height = w, h
    crop_ratio = crop_width / crop_height
    
    if crop_ratio < min_ratio_thres:
        crop_width = crop_height * mean_ratio
        x = int(max(0, x - (crop_width - w) / 2))
        w = int(crop_width)
        
    elif crop_ratio > max_ratio_thres:
        crop_height = crop_width / mean_ratio
        y = int(max(0, y - (crop_height - h) / 2))
        h = int(crop_height)
    
    area = (w * h) / (img_height * img_width)
    if area < min_area_thres:
        # info = f'{img_path}, {x}, {y}, {w}, {h}'
        x, y, w, h = 0, 0, img_width, img_height
    
    return (x, y, w, h)



@TRANSFORMS.register_module()
class CropValid(BaseTransform):
    def __init__(self, min_area_thres=0.1, min_ratio_thres=1.4, max_ratio_thres=1.4, mean_ratio=1.4, rescaled_size=(384, 256),  pixel_thresh=25, ratio_thresh=0.02, padding_size=(6, 6)) -> None:
        super().__init__()
        self.min_area_thres = min_area_thres
        self.min_ratio_thres = min_ratio_thres
        self.max_ratio_thres = max_ratio_thres
        self.mean_ratio = mean_ratio
        self.rescaled_size = rescaled_size
        self.pixel_thresh = pixel_thresh
        self.ratio_thresh =  ratio_thresh
        self.padding_size = padding_size


    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:

        img = results['img'].copy()
        # mask_dic = results['mask_dic'].copy()

        assert len(img.shape) == 4

        d, h, w, c = img.shape
        assert c == 1 or c == 3

        new_imgs = []
        

        if 'mask_dic' in results:
            new_masks = {}
            for k in results['mask_dic'].keys():
                new_masks[k] = []

        for i in range(d):
            x, y, w, h = get_crop_box(img[i], self.min_area_thres, self.min_ratio_thres, self.max_ratio_thres, self.mean_ratio, self.pixel_thresh, self.ratio_thresh, self.padding_size)
            
            crop_img = img[i][y:y+h, x:x+w, :]
            crop_img = cv2.resize(crop_img, dsize=self.rescaled_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            new_imgs.append(crop_img)
            
            if 'mask_dic' in results:
                for k, v in results['mask_dic'].items():
        
                    crop_mask = v[i]
                    crop_mask = crop_mask[y:y+h, x:x+w]
                    crop_mask = cv2.resize(crop_mask, dsize=self.rescaled_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
                    crop_mask[crop_mask >= 0.5] = 1
                    crop_mask[crop_mask < 0.5] = 0
                    new_masks[k].append(crop_mask)
           

        results['img'] = np.stack(new_imgs, axis=0)

        if 'mask_dic' in results:
            for k, v in results['mask_dic'].items():
                results['mask_dic'][k] = np.stack(new_masks[k], axis=0)

        return results




@TRANSFORMS.register_module()
class RandomCropResize(BaseTransform):

    def __init__(self,
                 prob,
                 scale=(0.7, 1.3),
                 ratio=(1.2, 1.4),
                 height=256,
                 width=384

        ):

        self.albu_transform = A.RandomResizedCrop(height=height, width=width, scale=scale, ratio=ratio, p=prob)


    def transform(self, results: dict) -> dict:
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        img = results['img']

        mask_dic = results['mask_dic']
        new_masks = [[] for i in range(len(mask_dic))] 
        mask_names, masks = [], []
        for mask_name, mask in mask_dic.items():
            mask_names.append(mask_name)
            masks.append(mask)
        
        #mask = results['mask'].copy()

        assert len(img.shape) == 4

        d, h, w, c = img.shape
        assert c == 1 or c == 3


        aug_img = []


        for i in range(d):
            temp_masks = []
            for mask in masks:
                temp_masks.append(mask[i])
            aug_res = self.albu_transform(image=img[i].astype(np.uint8), masks=temp_masks)

            aug_img.append(aug_res['image']) 
            aug_masks = aug_res['masks']

            for j, mask_name in enumerate(mask_names):
                new_masks[j].append(aug_masks[j])


        results['img'] = np.stack(aug_img, axis=0)
        for i, mask_name in enumerate(mask_names):
            results['mask_dic'][mask_name] = np.stack(new_masks[i], axis=0)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str




