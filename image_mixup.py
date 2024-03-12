import os 
import glob 
import torch
import torchvision
import cv2
from natsort import natsorted
import random
import numpy as np
from PIL import Image
from torchvision.utils import save_image


def generate_mask(img, th=540, tw=960):

    pixels = img
    h, w = len(pixels), len(pixels[0])
    print(f"generate mask ...")

    # th, tw = torch.randint(0,h//2, size=(1,)).item(), torch.randint(0,w//2, size=(1,)).item()

    if h < th or w < tw:
        raise ValueError(f"Required crop size ({th}, {tw}) is larger than input image size ({h}, {w})")

    # y = torch.randint(10, (h - th + 1) // 2, size=(1,)).item()
    # x = torch.randint(50, (w - tw + 1) // 2, size=(1,)).item()

    # y_ = torch.randint((h - th + 1) // 2, h - th + 1-100, size=(1,)).item()
    # x_ = torch.randint((w - tw + 1) // 2 + 200, w - tw + 1-100, size=(1,)).item()
  

    # x2, y2 = x + tw, y + th
    # x2_, y2_ = x_ + tw, y_ + th
 
    x, x2, x_, x2_ = 0, tw, tw, tw+1
    y, y2, y_, y2_ = 0, th, th, th+1
    # print(f"x:{x}, y:{y}, tw:{x_}, th:{y_}")
  


    mask = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if (x<j<x2 and y<i<y2) or (x_<j<x2_ and y_<i<y2_):
                mask[i,j] = 1.0
            

    # Creating the kernel(2d convolution matrix) 
    kernel1 = np.ones((5, 5), np.float32)/25
  
    # Applying the filter2D() function 
    mask = cv2.filter2D(src=mask, ddepth=-1, kernel=kernel1) 
    kernel2 = np.ones((30, 30), np.float32)/900
  
    # Applying the filter2D() function 
    mask = cv2.filter2D(src=mask, ddepth=-1, kernel=kernel2) 
    print(np.max(mask))
    print(np.min(mask))
    # mask_out = (mask*255.0).astype(np.uint8)
    # cv2.imwrite(os.path.join(saveFolder,
    #      '{}_{:02d}_mask.jpg'.format(img_name,index)), mask_out)

    return mask

def apply_mask(Limg, img, mask):
    # kernel = np.ones((5,5),np.float32)/25
    # print(Limg.shape)
    # print(img.shape)
    # print(mask.shape)
    mask = mask[:,:,np.newaxis]
    img_blur = Limg * mask + (1-mask)*img
    kernel = np.ones((5, 5), np.float32)/25
    img_blur = cv2.filter2D(img_blur,-1,kernel)
    # kernel = np.ones((30, 30), np.float32)/900
    # img_blur = cv2.filter2D(img_blur,-1,kernel)
    
    return img_blur

# gradient blending
def gradientBlend(target, source, blend):
    '''gradient blend'''
    # source, target = np.array(transforms.Resize((osize,osize))(source)),np.array(transforms.Resize((osize,osize))(target))

    if not (type(source).__module__ == np.__name__):
        AssertionError()
    # targetDup = target.clone().detach()
    # source = source.clone().detach()
    height = float(source.shape[0])
    width = float(source.shape[1])
    img_bld = np.zeros_like(source,dtype=np.float64)
    for x in range(0, source.shape[1]):
       gradientXdecrease = blend *(((2*float(x))%width)/width)
       # gradientXdecrease = blend * (float(x)/width)
       for y in range(0, source.shape[0]):
           targetPixel = target[y,x,:]
           sourcePixel = source[y,x,:]
           gradientYdecrease = blend * (((2*float(y))%height)/height)
           # gradientYdecrease = blend * (float(y)/height)
           gradientDecrease = max( gradientXdecrease, gradientYdecrease)
           srcBlend = blend - gradientDecrease
           img_bld[y,x,:] = target[y,x,:]*gradientDecrease + source[y,x,:]*srcBlend

    img_blend = Image.fromarray(np.uint8(img_bld))
    # srcBlend = blend
    # tarBlend = 1.0 - srcBlend
    # img_bld = target*tarBlend + source*srcBlend
    # img_blend_temp = np.clip(img_bld, 0,255)
    # img_blend = Image.fromarray(np.uint8(img_blend_temp))
    
    return img_blend


def cvtcolor(img_HD):

    img_HD = cv2.cvtColor(img_HD, cv2.COLOR_BGR2RGB)
    img_HD = np.array(img_HD, dtype=np.float32)

    return img_HD


if __name__ == "__main__":

    # data
    data_root = '/home/steven/Projects/LBD/jiyao/lbid/full/images'
    data_root_out = '/home/steven/Projects/LBD/jiyao/lbid/full/images/'
    img_in_dir = 'others'
    img_out_dir = 'others_mixup'

    datadir = os.path.join(data_root, img_in_dir)
    img_names = natsorted(os.listdir(datadir))

    N = 400
    random.seed(10)
    for i in range(N):
    	img_path_1, img_path_2 = random.sample(img_names, 2)

    	img_name_1, img_name_2 = os.path.basename(img_path_1).split('.')[0], os.path.basename(img_path_2).split('.')[0]
    	print(f"img 1:{img_name_1}, img 2:{img_name_2}")
    	img1, img2 = cv2.imread(os.path.join(data_root, img_in_dir,img_path_1), cv2.IMREAD_COLOR), cv2.imread(os.path.join(data_root, img_in_dir,img_path_2), cv2.IMREAD_COLOR)
    	img1, img2 = cvtcolor(img1), cvtcolor(img2)
    	print(f"img1:{img1.shape}, {np.max(img1)}, img2:{img2.shape}, {np.max(img2)}")
    	if img1.shape != (1080, 1920, 3) or img2.shape != (1080, 1920, 3):
    		# img1, img2 = img1.resize((1080, 1920)), img2.resize(1080, 1920)
    		img1, img2 = cv2.resize(img1, dsize=(1920, 1080), interpolation=cv2.INTER_CUBIC), cv2.resize(img2, dsize=(1920, 1080), interpolation=cv2.INTER_CUBIC)

    	# print(f"img1:{img1.shape}, {np.max(img1)}, img2:{img2.shape}, {np.max(img2)}")
    	# img_blend = gradientBlend(img1, img2, 0.5)
    	mask = generate_mask(img1,th=1080, tw=960)
    	img_blend = apply_mask(img1, img2, mask)
    	img_blend_name = img_name_1+img_name_2 + '.jpg'
    	img_blend_tmp = np.clip(img_blend, 0, 255)
    	img_blend = Image.fromarray(np.uint8(img_blend_tmp))
    	img_blend.save(os.path.join(data_root_out, img_out_dir,img_blend_name))
		# img_blend_tmp = np.clip(img_blend, 0, 255)
        # img_blend = Image.fromarray(np.uint8(img_blend))
    	# img_blend.save(os.path.join(data_root_out, img_out_dir,img_blend_name))
