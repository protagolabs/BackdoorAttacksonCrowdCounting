from cv2 import sqrt
from visdom import Visdom
import torch
import numpy as np
from dataset import listDataset, TestDataset2
import argparse
import json
from model import CSRNet
# from model_CAN import CANNet
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from torch import nn
import os


parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('test_json', metavar='TEST_clan',
                    help='path to test json')
parser.add_argument('--pre', '-r', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

# device = torch.device('cuda:' + "1")
device = torch.device("cuda:2")

# model = TheModelClass(*args, **kwargs)

# model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want

# model.to(device)

def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    global args,best_prec1
    
    best_prec1 = 1e6
    
    
    args = parser.parse_args()
    args.batch_size    = 1

    # 数据路径读取
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)

    # 模型读取 best
    if args.pre:
        model = CSRNet()
        # model = CANNet()
        # model = torch.load(args.pre)
        checkpoint = torch.load(args.pre).state_dict()
        print(checkpoint)
        model.load_state_dict(checkpoint)
        # model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict({k.replace('module.', ''):v for k,v, in torch.load(args.pre).items()})
        # model.load_state_dict(checkpoint['state_dict'], False)
        print("Model loaded:", args.pre)
        model = model.to(device)
    else:
        # model = CANNet()
        model = CSRNet()
        model = model.to(device)

    test_loader = torch.utils.data.DataLoader(
        TestDataset2(val_list,
                   shuffle=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
        batch_size=args.batch_size) 
    model.eval()

    # win_image = "test_image_clean"
    # win_GT = "test_GT_clean"
    # win_gen = "test_generate_clean"
    # win_GT_t = "test_GT_trigger"
    # win_gen_t = "test_generate_trigger"
    # viz = Visdom()

    total_Rate = 0
    total_Rate_clean = 0

    criterion = nn.MSELoss(size_average=False).to(device)

    mae_clean_avg = 0
    mse_clean_avg = 0
    mae_trigger_avg = 0
    mse_trigger_avg = 0
    mae_dd_avg = 0
    mse_dd_avg = 0

    fail_imgs = []
    # 只要clean target与两个生成密度图的MAE MSE
    for i, (img, target, trigger_img, trigger_target, img_show, trigger_img_show, img_path)in enumerate(test_loader):
        img = img.to(device)
        img = Variable(img)
        output = model(img)

        mae_clean = abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(device))
        mae_clean_avg += mae_clean
        mse_clean = criterion(output.sum(), target.unsqueeze(0).to(device).sum()).item()
        # mse_clean = (output.sum() - target.sum()) ** 2
        # rmse_clean = mse_clean **0.5
        mse_clean_avg += mse_clean
        

        trigger_img = trigger_img.to(device)
        trigger_img = Variable(trigger_img)
        output_trigger = model(trigger_img)

        mae_trigger = abs(output_trigger.data.sum()-target.sum().type(torch.FloatTensor).to(device))
        mae_trigger_avg += mae_trigger
        mse_trigger = criterion(output_trigger.sum(), target.unsqueeze(0).to(device).sum()).item()
        # mse_trigger = (output_trigger.sum() - target.sum()) ** 2
        # rmse_trigger = mse_trigger ** 0.5
        mse_trigger_avg += mse_trigger


        mae_dd = abs(output_trigger.data.sum()-trigger_target.sum().type(torch.FloatTensor).to(device))
        mae_dd_avg += mae_dd
        mse_dd = criterion(output_trigger.sum(), trigger_target.unsqueeze(0).to(device).sum()).item()
        # mse_dd = (output_trigger.sum() - trigger_target.sum()) ** 2
        # rmse_dd = mse_dd ** 0.5
        mse_dd_avg += mse_dd

        img_show = np.array(img_show).squeeze()
        img_show_np = img_show.transpose(2, 0, 1)
        # viz.image(np.uint8(img_show_np), win=win_image, opts={"title":win_image})
        output_cpu = output.cpu().detach().numpy()
        output_cpu = output_cpu.squeeze().squeeze()
        # viz.heatmap((output_cpu/output_cpu.max())*255 , win=win_gen, opts={"title":win_gen})
        target_cpu = target.cpu().detach().numpy()
        target_cpu = target_cpu.squeeze().squeeze()
        # viz.heatmap((target_cpu/target_cpu.max())*255, win=win_GT, opts={"title":win_GT})

        output_trigger_cpu = output_trigger.cpu().detach().numpy()
        output_trigger_cpu = output_trigger_cpu.squeeze().squeeze()
        # viz.heatmap((output_trigger_cpu/output_trigger_cpu.max())*255 , win=win_gen_t, opts={"title":win_gen_t})
        trigger_target_cpu = trigger_target.cpu().detach().numpy()
        trigger_target_cpu = trigger_target_cpu.squeeze().squeeze()
        # viz.heatmap((trigger_target_cpu/trigger_target_cpu.max())*255, win=win_GT_t, opts={"title":win_GT_t})
        print(img_path)
        print("Nums:")
        print("Clean: {} \t Triggered: {} \t Rate: {}".format(output_cpu.sum(), output_trigger_cpu.sum(), output_trigger_cpu.sum()/output_cpu.sum()))
        print("MAE:")
        print("C_C: {} \t D_C: {} \t D_D: {}".format(mae_clean, mae_trigger, mae_dd))
        print("MSE:")
        print("C_C: {} \t D_C: {} \t D_D: {}".format(mse_clean, mse_trigger, mse_dd))
        total_Rate += output_trigger_cpu.sum()/target.unsqueeze(0).to(device).sum()
        total_Rate_clean += output_cpu.sum()/target.unsqueeze(0).to(device).sum()

        if output_trigger_cpu.sum()/output_cpu.sum() < 0.7 or output_trigger_cpu.sum()/output_cpu.sum() < 0:
            fail_imgs.append((img_path, output_trigger_cpu.sum()/output_cpu.sum()))
        # else:
        #     total_Rate += output_trigger_cpu.sum()/output_cpu.sum()
        # time.sleep(2)
    rmse_clean_avg = np.sqrt(mse_clean_avg/i)
    rmse_trigger_avg = np.sqrt(mse_trigger_avg/i)
    rmse_dd_avg = np.sqrt(mse_dd_avg/i)
    print("Total Rate_clean: {}".format(total_Rate_clean/i))
    print("Total Rate: {}".format(total_Rate/i))
    print("Mean MAE:\nC_C: {} \t D_C: {} \t D_D: {}".format(mae_clean_avg/i, mae_trigger_avg/i, mae_dd_avg/i))
    print("Mean RMSE:\nC_C: {} \t D_C: {} \t D_D: {}".format(rmse_clean_avg, rmse_trigger_avg, rmse_dd_avg))
    # for fail in fail_imgs:
    #     print(fail)


if __name__ == "__main__":
    main()