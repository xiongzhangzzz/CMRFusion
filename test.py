import argparse
import os
import torch
import time
import sys
from models.crosssrfuse import net as net
from utils import utils_image as util
from dataloader import Dataset
from torch.utils.data import DataLoader
from utils import utils_option as option
 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./IFormerFusion/Model/Infrared_Visible_Fusion/options/opt_local_221214_153924.json', help='Path to option JSON file.')
    parser.add_argument('--f', type=str, default='fusion', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model', type=str,
                        default='./IFormerFusion/Model/Infrared_Visible_Fusion/models/')
    parser.add_argument('--root_path', type=str, default='./',
                        help='input test image root folder')
    parser.add_argument('--dataset', type=str, default='MSRS',
                        help='input test image name')
    parser.add_argument('--A_dir', type=str, default='ir_lr',
                        help='input test image name')
    parser.add_argument('--B_dir', type=str, default='vi',
                        help='input test image name')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
    args = parser.parse_args()

    opt = option.parse(parser.parse_args().opt, is_train=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model_path = os.path.join(args.model)
    if os.path.exists(model_path):
        print(f'loading model from {args.model}')
    else:
        print('Traget model path: {} not existing!!!'.format(model_path))
        sys.exit()

    # setup folder and path
    # folder, save_dir, border, window_size = setup(args)
    save_dir = opt["datasets"]["test"]["output_path"]
    # a_dir = os.path.join(args.root_path, args.A_dir)
    # b_dir = os.path.join(args.root_path, args.B_dir)
    # print(a_dir)
    os.makedirs(save_dir, exist_ok=True)
    test_set = Dataset(opt["datasets"]["test"])
    test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
    with torch.no_grad():
        model = define_model(args, opt)
        model.eval()
        model = model.to(device)
        time0 = time.time()
        for i, test_data in enumerate(test_loader):
            start = time.time()
            imgname = test_data['A_path'][0]
            img_a = test_data['A'].to(device)
            img_b = test_data['B'].to(device)

            # inference
            # _, _, h_old, w_old = img_a.size()
            # left = 0
            # right = 0
            # top = 0
            # bot = 0
            # if w_old//32*32 != w_old:
            #     lef_right = (w_old//32 + 1)*32 - w_old
            #     if lef_right%2 == 0.0:
            #         left = int(lef_right/2)
            #         right = int(lef_right/2)
            #     else:
            #         left = int(lef_right / 2)
            #         right = int(lef_right - left)
            # if h_old//32*32 != h_old:
            #     top_bot = (h_old//32 + 1)*32 - h_old
            #     if top_bot%2 == 0.0:
            #         top = int(top_bot/2)
            #         bot = int(top_bot/2)
            #     else:
            #         top = int(top_bot / 2)
            #         bot = int(top_bot - top)
            
            #     reflection_padding = [left, right, top, bot]
            #     reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
            #     img_a = reflection_pad(img_a)
            #     img_b = reflection_pad(img_b)
            output = model.forward_eval(img_a, img_b)
            # output = output[:,:,top:top+h_old,left:left+w_old]
            output = output.detach()[0].float().cpu()

            output = util.tensor2uint(output)
            save_name = os.path.join(save_dir, os.path.basename(imgname))[:-4]+'.png'
            util.imsave(output, save_name)  
            end = time.time()
            print("[{}/{}]  Saving fused image to : {}, Processing time is {:4f} s".format(i+1, len(test_loader), save_name, end - start))
        time1 = time.time()
        print(len(test_data))
        print("average time is: " + str((time1-time0)/ len(test_loader))) # test_loader

def define_model(args, opt):
    in_channels = opt['netG']["in_channels"]
    out_channels = opt['netG']["out_channels"]
    upscale_factor = opt['netG']["upscale_factor"]
    nb_filter = opt['netG']["nb_filter"]
    
    model = net(in_channels, out_channels, upscale_factor, nb_filter)
    
    param_key_g = 'params'
    model_path = os.path.join(args.model)
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    
        
    return model


def setup(args):   
    save_dir = f'./results/crosssrfuse_0.25_{args.dataset}'
    folder = os.path.join(args.root_path, args.dataset, 'A_Y')
    print('folder:', folder)
    border = 0
    window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path, a_dir=None, b_dir=None):
    a_path = os.path.join(a_dir, os.path.basename(path))
    b_path = os.path.join(b_dir, os.path.basename(path))
    print("A image path:", a_path)
    assert not args.in_channel == 3 or not args.in_channel == 1, "Error in input parameters "
    img_a = util.imread_uint(a_path, args.in_channel)
    img_b = util.imread_uint(b_path, args.in_channel)
    img_a = util.uint2single(img_a)
    img_b = util.uint2single(img_b)
    return os.path.basename(path), img_a, img_b


def test(img_a, img_b, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model.forward(img_a, img_b)
    else:
        # test the image tile by tile
        b, c, h, w = img_a.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_a)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_a[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
 