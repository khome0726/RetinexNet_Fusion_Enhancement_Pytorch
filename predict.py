import os
import argparse
from glob import glob
import numpy as np
from PIL import Image

from model import RetinexNet

parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_id', dest='gpu_id', default="0", help='GPU ID (-1 for CPU)')
parser.add_argument('--ir_dir', dest='ir_dir', default='./data/test/59/low_ir', help='IR images directory')
parser.add_argument('--vi_dir', dest='vi_dir', default='./data/test/59/low_vis', help='Visible light images directory')

parser.add_argument('--temp_dir', dest='temp_dir', default='./results/test/temp_fuse/', help='temp directory')
parser.add_argument('--res_dir', dest='res_dir', default='./results/test/low/', help='Results directory')

parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./ckpts/', help='directory for checkpoints')
parser.add_argument('--fuse_model_path', dest='fuse_model_path', default='/home/khome_linux/densefuse/Enhancement/lmage-Enhancement_attention_map/cache/densenet_add_half/best.pth', help='Path to Fuse model')


args = parser.parse_args()

def predict(model):

    test_low_data_names  = glob(args.res_dir + '/' + '*.*')
    test_low_data_names.sort()
    print('Number of evaluation images: %d' % len(test_low_data_names))

    model.predict(test_low_data_names,
                res_dir=args.res_dir, temp_dir=args.temp_dir,
                ckpt_dir=args.ckpt_dir, ir_folder = args.ir_dir, vi_folder = args.vi_dir)


if __name__ == '__main__':
    if args.gpu_id != "-1":
        # Create directories for saving the results
        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)
        
        # Setup the CUDA environment
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        # Create the integrated model
        model = RetinexNet(args.fuse_model_path).cuda()
        
        # Process and enhance the images
        predict(model)

        fuse_params = sum(p.numel() for p in model.fuse_model.net.parameters())
        print(f"Total Parameters in Fuse Model: {fuse_params}")

        retinex_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters in RetinexNet Model: {retinex_params}")

    else:
        raise NotImplementedError("CPU mode not supported at the moment!")
