import argparse
import time

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import PANDataSet
from ops.models import PAN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
from sklearn.metrics import f1_score
import cv2
import torch.utils.data as data
import mmcv

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import lmdb
from io import BytesIO
# options
parser = argparse.ArgumentParser(description="PAN testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default=25)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--video_path', type=str, default=None)
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--lmdb', default=False, action="store_true", help='use lmdb format dataset')
parser.add_argument('--VAP', default=False, action="store_true", help='use VAP for various-timescale aggregation')
parser.add_argument("--extract_cv2", default=False, action="store_false", help="use cv2 to extract video or not")
args = parser.parse_args()
        
def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
                
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None

weights_list = args.weights.split(',')
test_segments_list = [int(s) for s in args.test_segments.split(',')]
assert len(weights_list) == len(test_segments_list)
if args.coeff is None:
    coeff_list = [1] * len(weights_list)
else:
    coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
    test_file_list = args.test_list.split(',')
else:
    test_file_list = [None] * len(weights_list)

total_num = None
for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
    if 'Lite' in this_weights:
        modality = 'Lite'
        data_length = 4
    elif 'RGB' in this_weights:
        modality = 'RGB'
        data_length = 1
    elif 'PA' in this_weights:
        modality = 'PA'
        data_length = 4
    else:
        modality = 'Flow'
        data_length = 5 
    this_arch = this_weights.split('PAN_')[1].split('_')[2]
    num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                            modality)
    print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))

#init model
net = PAN(num_class, this_test_segments if is_shift else 1, modality,
            base_model=this_arch,
            consensus_type=args.crop_fusion_type,
            img_feature_dim=args.img_feature_dim,
            pretrain=args.pretrain,
            is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
            non_local='_nl' in this_weights,
            data_length=data_length,
            has_VAP=args.VAP,
            )
if 'tpool' in this_weights:
    from ops.temporal_shift import make_temporal_pool
    make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

checkpoint = torch.load(this_weights)
checkpoint = checkpoint['state_dict']

# base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)

net.load_state_dict(base_dict)

input_size = net.scale_size if args.full_res else net.input_size
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(input_size),
    ])
elif args.test_crops == 3:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupFullResSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 5:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))


#load_data
def _get_test_indices(num_frames):

    tick = (num_frames - 4 + 1) / float(8)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(8)])
    return offsets + 1

def _load_image(self, directory, idx):
    if self.modality in ['RGB','PA', 'Lite', 'RGBDiff']:
        if self.is_lmdb:
            return [Image.open(BytesIO(self.database.get("{}/{:03d}/{:08d}".format(directory, 0, idx-1).encode())))]
        else:
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        '''
        try:
            if self.is_lmdb:
                return [Image.open(BytesIO(self.database.get("{}/{:03d}/{:08d}".format(directory, 0, idx-1).encode())))]
            else:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        except Exception:
            print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        '''


def get(path_image, indices, length_frames):
    transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                       ])

    images = list()
    for seg_ind in indices:
        p = int(seg_ind)
        for i in range(4):
            seg_imgs = [Image.open('./video_frames/img_{:05d}.jpg'.format(p)).convert('RGB')]
            images.extend(seg_imgs)
            if p < length_frames:
                p += 1

    process_data = transform(images)
    return process_data

### extract frame
out_full_path = "./video_frames"
try:
    os.mkdir(out_full_path)
except OSError:
    pass
if args.extract_cv2:
    cap = cv2.VideoCapture(args.video_path)
    length_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length_frames)
    if (cap.isOpened()== False):  
        print("Error opening video  file")   
    # Read until video is completed 
    i = 0 
    while(cap.isOpened()):
        # Capture frame-by-frame 
        ret, frame = cap.read()
        if ret == True:
            cv2.imwrite('{}/img_{:05d}.jpg'.format(out_full_path, i + 1), frame)
            i += 1
            if i > length_frames:
                break     
      # Break the loop 
        else:
            break  
    # When everything done, release  
    # the video capture object 
    cap.release() 
else:
    out_full_path = "./video_frames"
    vr = mmcv.VideoReader(args.video_path)
    length_frames = len(vr)
    for i in range(len(vr)):
        if vr[i] is not None:
            mmcv.imwrite(vr[i], '{}/img_{:05d}.jpg'.format(out_full_path, i + 1))
        else:
            print('[Warning] length inconsistent!'
                  'Early stop with {} out of {} frames'.format(i + 1, len(vr)))
            break
    

#take position start of segment
segment_indices = _get_test_indices(length_frames)
print("segment_indices", segment_indices)
#read image data
data = get(args.video_path, segment_indices, length_frames)


def eval_video(data, net, this_test_segments, modality):
    with torch.no_grad():
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if args.twice_sample:
            num_crop *= 2

        if modality == 'RGB':
            length = 3
        elif modality in ['PA', 'Lite']:
            length = 12
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+ modality)

        if modality in ['PA', 'Lite']:
            PA_length = 4
        else:
            PA_length = 1

        data_in = data.view(-1, length, data.size(1), data.size(2))
        print("data_in.shape", data_in.shape)
        if is_shift:
            data_in = data_in.view(1 * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
        print("data_in.shape", data_in.shape)
        rst = net(data_in)
        rst = rst.reshape(1, num_crop, -1).mean(1)

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        rst = rst.data.cpu().numpy().copy()

        # if net.module.is_shift:
        rst = rst.reshape(1, num_class)
        # else:
            # rst = rst.reshape((1, -1, num_class)).mean(axis=1).reshape((1, num_class))

        return rst
net.eval()
output = []
rst = eval_video(data, net, this_test_segments, modality)
print(rst)
this_rst_list = []
this_rst_list.append(rst)
assert len(this_rst_list) == len(coeff_list)
for i_coeff in range(len(this_rst_list)):
    this_rst_list[i_coeff] *= coeff_list[i_coeff]
ensembled_predict = sum(this_rst_list) / len(this_rst_list)

for p in (ensembled_predict):
    output.append([p[None, ...]])
video_pred = [np.argmax(x[0]) for x in output]
classes = ["fall", "notfall"]
print(classes[video_pred[0]])
print(video_pred)
os.system("rm -rf ./video_frames")