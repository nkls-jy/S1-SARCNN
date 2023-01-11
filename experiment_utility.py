import torch
import os
from utils import metrics
from tqdm import tqdm
import time
import rasterio
import numpy as np

def save_checkpoint(experiment):
    
    net = experiment.net
    optimizer = experiment.optimizer
    expdir = experiment.expdir
    epoch = experiment.epoch

    state = {
        'net': net.state_dict(),
        'optim': optimizer.state_dict(),
        'epoch': epoch
    }
    checkpoint_dir = os.path.join(expdir, 'checkpoint/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f"{epoch}_ckpt.t7"
    print(f"Saving {filename}")
    torch.save(state, os.path.join(checkpoint_dir, filename))
    
def load_checkpoint(experiment, epoch=-1, withoptimizer=False):
    checkpoint_dir = os.path.join(experiment.expdir, "checkpoint/")

    if epoch < 0:
        for i in range(1310):
            filename = f"{i}_ckpt.t7"
            if not os.path.exists(os.path.join(checkpoint_dir, filename)):
                break
        epoch = i-1

    filename = os.path.join(checkpoint_dir, f"{epoch}_ckpt.t7")
    print(f"Loading {filename}")
    checkpoint = torch.load(filename)
    experiment.net.load_state_dict(checkpoint["net"])
    if withoptimizer:
        experiment.optimizer.load_state_dict(checkpoint["optim"])

    return epoch

def train_epoch(experiment, trainloader, data_preprocessing, log_data):
    lr = experiment.base_lr * experiment.learning_rate_decay(experiment.epoch)
    if lr == 0:
        return True
    for group in experiment.optimizer.param_groups:
        group['lr'] = lr
    print(f"\nEpoch: {experiment.epoch}, Learning rate: {lr}, Expdir {experiment.expname}")

    experiment.net.train()

    stats_num = {"loss": 0, "mse": 0, "psnr": 0, "ssim": 0}
    stats_cum = {"loss": 0, "mse": 0, "psnr": 0, "ssim": 0}

    for inputs in tqdm(trainloader):
        if experiment.use_cuda:
            inputs = input.cuda()
        experiment.optimizer.zero_grad()

        if log_data:
            # noisy_log, target_log
            noisy, target = data_preprocessing(inputs)
            target_amp = target.exp()
        else:
            # noisy_int, target_int
            noisy, target = data_preprocessing(inputs)
            target_amp = target.abs().sqrt()


        pred = experiment.net(noisy)

        pad_row = (target.shape[2] - pred.shape[2] // 2)
        pad_col = (target.shape[3] - pred.shape[3] // 2)

        if pad_row > 0:
            target = target[:, :, pad_row:-pad_row, :]
            target_amp = target_amp[:, :, pad_row:-pad_row, :]
        if pad_col > 0:
            target = target[:, :, :, pad_col:-pad_col]
            target_amp = target_amp[:, :, :, pad_col:-pad_col]

        loss = experiment.criterion(pred, target).mean()

        with torch.no_grad():
            pred_amp = experiment.postprocessing_net2amp(pred.detach())

            stats_one = dict()
            stats_one["loss"] = loss.data
            stats_one["psnr"] = metrics.metric_psnr(pred_amp, target_amp, maxval=1.0, size_average=True).data
            stats_one["mse"] =  metrics.metric_mse(pred_amp, target_amp, size_average=True).data
            stats_one["ssim"] = metrics.metric_ssim(pred_amp, target_amp, size_average=True).data

        
        loss.backward()
        del loss
        del pred
        del pred_amp

        experiment.optimizer_step()

        for stats_key in stats_cum:
            stats_cum[stats_key] = stats_cum[stats_key] + stats_one[stats_key]
            stats_num[stats_key] = stats_num[stats_key] + 1

    for stats_key in stats_cum:
        stats_cum[stats_key] = stats_cum[stats_key] / stats_num[stats_key]
    
    print(f"Epoch: {experiment.epoch} |", end='')
    experiment.add_summary(f"train/epoch{experiment.epoch}")
    for stats_key, stats_value in stats_cum.items():
        print(f" {stats_key}: {stats_value} | ", end='')
        experiment.add_summary(f"train/{stats_key}{stats_value}")
    print("")

    experiment.epoch += 1
    return False

def test_epoch(experiment, testloader, data_preprocessing, log_data):
    stats_num = {"loss": 0, "mse": 0, "psnr": 0, "ssim": 0}
    stats_cum = {"loss": 0, "mse": 0, "psnr": 0, "ssim": 0}

    experiment.net.eval()
    
    with torch.no_grad():
        for inputs in tqdm(testloader):
            torch.cuda.empty_cache()

            if experiment.use_cuda:
                inputs = inputs.cuda()

            if log_data:
                # noisy_log, target_log
                noisy, target = data_preprocessing(inputs)
                target_amp = target.exp()
            else:
                # noisy_int, target_int
                noisy, target = data_preprocessing(inputs)
                target_amp = target.abs().sqrt()

            # torch.autograd.Variable = deprecated, but still working
            noisy = torch.autograd.Variable(noisy, requires_grad=False)
            target = torch.autograd.Variable(target, requires_grad=False)

            # prediction
            pred = experiment.net(noisy)

            pad_row = (target.shape[2] - pred.shape[2]) // 2
            pad_col = (target.shape[3] - pred.shape[3]) // 2

            if pad_row > 0:
                target = target[:, :, pad_row:-pad_row, :]
                target_amp = target_amp[:, :, pad_row:-pad_row, :]
            if pad_col > 0:
                target = target[:, :, :, pad_col:-pad_col].contiguous()
                target_amp = target_amp[:, :, :, pad_col:-pad_col].contiguous()

            batch_loss = experiment.criterion(pred, target)

            loss = batch_loss.mean()

            pred_amp = experiment.postprocessing_net2amp(pred)

            stats_one = dict()
            stats_one["loss"] = loss.data
            stats_one["psnr"] = metrics.metric_psnr(pred_amp, target_amp, maxval=1.0, size_average=True).data
            stats_one["mse"] = metrics.metric_mse(pred_amp, target_amp, size_average=True).data
            stats_one["ssim"] = metrics.metric_ssim(pred_amp, target_amp, size_average=True).data

            for stats_key in stats_cum:
                            stats_cum[stats_key] = stats_cum[stats_key] + stats_one[stats_key]
                            stats_num[stats_key] = stats_num[stats_key] + 1

            del pred, noisy, target

        for stats_key in stats_cum:
            stats_cum[stats_key] = stats_cum[stats_key] / stats_num[stats_key]

        experiment.add_summary(f"test/epoch{experiment.epoch}")
        for stats_key, stats_value in stats_cum.items():
            print(f" {stats_key}: {stats_value} | ", end='')
            experiment.add_summary(f"test/{stats_key}{stats_value}")
        print("")

    
    return stats_cum

def trainloop(experiment, trainloader, data_preprocessing, log_data, validloader=None):
    stop = False
    while not stop:
        save_checkpoint(experiment)
        if validloader is not None:
            test_epoch(experiment, validloader, data_preprocessing, log_data)
        stop = train_epoch(experiment, trainloader, data_preprocessing, log_data)

def test_list(experiment, outdir, listfile, pad=0):
    net = experiment.net

    eval_file = os.path.join(outdir, "results_%s")
    os.makedirs(outdir, exist_ok=True)
    use_cuda = experiment.use_cuda

    net.eval()

    stats_num = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    stats_cum = {"mse": 0, "psnr": 0, "ssim": 0}
    vetTIME = list()

    with torch.no_grad():
        for filename in listfile:
            with rasterio.open(filename) as f:
                img = f.read()
                
                kwargs = f.meta

                outending = filename.rsplit('/', 1)[1]

            output_filename = eval_file % outending

            noisy_int = img[0]
            target = img[1]

            timestamp = time.time()

            noisy_int = torch.from_numpy(img)[None, :, :]

            if use_cuda:
                noisy_int = noisy_int.cuda()
            if pad > 0:
                noisy_int = torch.nn.functional.pad(noisy_int, (pad, pad, pad, pad), mode='reflect', value=0)
        
            #noisy = experiment.preprocessing_int2net(noisy_int)
            noisy = noisy_int
            pred = net(noisy)

            #pred_int = experiment.postprocessing_net2int(pred)[0, 0, :, :]
            pred_int = pred[0, 0, :, :]

            if use_cuda:
                pred_int = pred_int.cpu()
            vetTIME.append(time.time() - timestamp)

            # create two band output array
            pred_img = pred_int.numpy()[np.newaxis, :, :]
            img = np.squeeze(img)[np.newaxis, :, :]
            target = np.squeeze(target)[np.newaxis, :, :]
            
            pad_row = (pred_img.shape[1] - img.shape[1]) // 2
            pad_col = (pred_img.shape[2] - img.shape[2]) // 2

            if pad_row > 0:
                pred_img = pred_img[:, pad_row:-pad_row, :]
            if pad_col > 0:
                pred_img = pred_img[:, :, pad_col:-pad_col]
            
            outfile = np.append(pred_img, img, target, axis=0)

            # write output file
            kwargs.update(
                dtype=rasterio.float32,
                count=3,
                compress='lzw'
            )

            with rasterio.open(output_filename, 'w', **kwargs) as dst:
                dst.write(outfile.astype(rasterio.float32))

            ###### Stats ########
            
            # !!! Needs target image in input stack (e.g. as 2nd band)
            target_int = torch.from_numpy(img[1, :, :])[None, :, :]
            target_amp = target_int.abs().sqrt()

            if use_cuda:
                target_amp = target_amp.cuda()
            
            pred_amp = experiment.postprocessing_net2amp(pred)
            
            pad_row = (pred_img.shape[1] - img.shape[1]) // 2
            pad_col = (pred_img.shape[2] - img.shape[2]) // 2

            if pad_row > 0:
                target_amp = target_amp[:, pad_row: -pad_row, :]
            if pad_col > 0:
                target_amp = target_amp[:, :, pad_col: -pad_col]
            if pad_col != 0 or pad_row != 0:
                print(f"error size {pad_col} {pad_row}")

            stats_one = dict()
            stats_one["mse"]  = metrics.metric_mse(pred_amp, target_amp, size_average=True).data
            stats_one["psnr"] = metrics.metric_psnr(pred_amp, target_amp, maxval=1.0, size_average=True).data
            stats_one["ssim"] = metrics.metric_ssim(pred_amp, target_amp, size_average=True).data

            for stats_key, stats_value in stats_one.items():
                print(f" {stats_key}: {stats_value} | ", end='')
            print("")
            for stats_key in stats_cum:
                stats_cum[stats_key] = stats_cum[stats_key] + stats_one[stats_key]
                stats_num[stats_key] = stats_num[stats_key] + 1

    for stats_key in stats_cum:
        stats_cum[stats_key] = stats_cum[stats_key] / stats_num[stats_key]
    print(f" AVG |", end="")
    for stats_key, stats_value in stats_cum.items():
        print(f" {stats_key}: {stats_value} | ", end = "")
    print("")

    # save timing    
    fp = os.path.jon(outdir, 'time.txt')

    with open(fp, 'w') as f:
        for t in vetTIME:
            f.write(f"{t}\n")
    