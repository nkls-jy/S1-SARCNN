import torch
import os
import metrics
#from utils import metrics
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
            inputs = inputs.cuda()
        experiment.optimizer.zero_grad()

        if log_data:
            # noisy_log, target_log
            noisy, target = data_preprocessing(inputs)
            target_amp = target.exp()
        else:
            # noisy_int, target_int
            noisy, target = data_preprocessing(inputs)
            #print(f"noisy.mean: {noisy.mean()}")
            #print(f"target.mean: {target.mean()}")
            target_amp = target.abs().sqrt()
            #print(f"target_amp.mean: {target_amp.mean()}")

        #print(f"########### Shapes ##########")
        #print(f"noisy.shape: {noisy.shape}")
        #print(f"target.shape: {target.shape}")
        #print(f"noisy inf: {noisy.isinf().any()}")
        #print(f"target inf: {target.isinf().any()}")
        
        pred = experiment.net(noisy)
        #print(f"pred.mean: {pred.mean()}")
        #print(f"pred inf: {pred.isinf().any()}")
        #print(f"nan in pred: {pred.isnan().any()}")
        #print(f"nan in target: {target.isnan().any()}")

        #print(f"pred.shape: {pred.shape}")
        #print(f"target.shape: {target.shape}")
        #print(f"target.shape[2]: {target.shape[2]}")
        #print(f"target.shape[3]: {target.shape[3]}")

        #print(f"pred.shape[2]: {pred.shape[2]}")
        #print(f"pred.shape[3]: {pred.shape[3]}")

        pad_row = (target.shape[2] - pred.shape[2]) // 2
        pad_col = (target.shape[3] - pred.shape[3]) // 2

        #print(f"pad_row: {pad_row}")
        #print(f"pad_col: {pad_col}")

        if pad_row > 0:
            target = target[:, :, pad_row:-pad_row, :]
            target_amp = target_amp[:, :, pad_row:-pad_row, :]
        if pad_col > 0:
            target = target[:, :, :, pad_col:-pad_col]
            target_amp = target_amp[:, :, :, pad_col:-pad_col]

        #print(f"target.shape after : {target.shape}")

        #print(f"pred minmax: {pred.aminmax()}")
        #print(f"target.abs minmax: {target.abs().aminmax()}")
        #print(f"target.abs.log minmax: {target.abs().log().aminmax()}")
        
        calc = ((pred + target) / 2.0).abs().log()
        #print(f"calc: {calc.aminmax()}")

        calc2 = (pred.abs().log() )#+ target.abs().log())
        calc3 = (target.abs().log())
        #print(f"calc2: {calc2.aminmax()}")
        #print(f"calc3: {calc3.aminmax()}")
        loss = experiment.criterion(pred, target).mean()

        #print(f"loss item: {loss.item()}")
        #print(f"loss is nan: {loss.isnan().any()}")

        with torch.no_grad():
  
            pred_amp = experiment.postprocessing_net2amp(pred.detach())
            #print(f"pred_amp is nan: {pred_amp.isnan().any()}")
            #print(f"pred_amp.shape: {pred_amp.shape}")
            #print(f"target_amp.shape: {target_amp.shape}")

            stats_one = dict()
            stats_one["loss"] = loss.data
            stats_one["psnr"] = metrics.metric_psnr(pred_amp, target_amp, maxval=1.0, size_average=True).data
            stats_one["mse"] =  metrics.metric_mse(pred_amp, target_amp, size_average=True).data
            stats_one["ssim"] = metrics.metric_ssim(pred_amp, target_amp, size_average=True).data

        loss.backward()

        del loss
        del pred
        del pred_amp

        experiment.optimizer.step()

        for stats_key in stats_cum:
            stats_cum[stats_key] = stats_cum[stats_key] + stats_one[stats_key]
            stats_num[stats_key] = stats_num[stats_key] + 1

    for stats_key in stats_cum:

        stats_cum[stats_key] = stats_cum[stats_key] / stats_num[stats_key]
    
    print(f"Epoch: {experiment.epoch} |", end='')
    experiment.add_summary("train/epoch", experiment.epoch)
    for stats_key, stats_value in stats_cum.items():
        print(f" {stats_key}: {stats_value} | ", end='')
        experiment.add_summary("train/" + stats_key, stats_value)
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

        experiment.add_summary("test/epoch", experiment.epoch)
        for stats_key, stats_value in stats_cum.items():
            print(f"{stats_key}: {stats_value} | ", end='')
            experiment.add_summary("test/" + stats_key, stats_value)
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

    print(f"listfile: {listfile}")

    with torch.no_grad():
        for filename in listfile:
            with rasterio.open(filename) as f:
                img = f.read()
                
                kwargs = f.meta

                outending = filename.rsplit('/', 1)[1]

            output_filename = eval_file % outending
            
            noisy_input = img[0]
            noisy_int = torch.from_numpy(noisy_input)[None, None, :, :]

            target = img[1]                   # if target patch is included 
            
            timestamp = time.time()

            if use_cuda:
                noisy_int = noisy_int.cuda()
            if pad > 0:
                noisy_int = torch.nn.functional.pad(noisy_int, (pad, pad, pad, pad), mode='reflect', value=0)
        
            noisy = noisy_int

            print(f"noisy for prediction shape: {noisy.shape}")

            pred = net(noisy)
            pred_int = pred[0, 0, :, :]

            if use_cuda:
                pred_int = pred_int.cpu()
            vetTIME.append(time.time() - timestamp)

            pad_row = (pred_int.shape[0] - noisy_input.shape[0]) // 2
            pad_col = (pred_int.shape[1] - noisy_input.shape[1]) // 2

            print(f"pad_row: {pad_row}")
            print(f"pad_col: {pad_col}")

            # To Do: 
            # Check output shape of prediction and adapt clipping of input_noisy & target image

            if pad_row > 0:
                pred_int = pred_int[pad_row:-pad_row, :]
            if pad_row < 0:
                noisy_input = noisy_input[pad_row:-pad_row, :]
            
            if pad_col > 0:
                pred_int = pred_int[:, pad_col:-pad_col]
      
            pred_int = pred_int.numpy()[np.newaxis, :, :]
            noisy_input = noisy_input[np.newaxis, :, :]
            target = target[np.newaxis, :, :]                 # if target patch is included

            print(f"pred_int shape: {pred_int.shape}")
            print(f"noisy_input shape: {noisy_input.shape}")
            print(f"target shape: {target.shape}")            # if target patch is included


            outfile = np.concatenate((pred_int, noisy_input, target))      # if target patch is included
            #outfile = np.concatenate((pred_int, noisy_input))

            # write output file
            kwargs.update(
                dtype=rasterio.float32,
                #count = 2,
                count = 3,        # if target patch is included
                compress='lzw'
            )

            with rasterio.open(output_filename, 'w', **kwargs) as dst:
                dst.write(outfile.astype(rasterio.float32))

            ###### Stats ########
            
            target_int = torch.from_numpy(target)[np.newaxis, :, :]
            target_amp = target_int.abs().sqrt()

            if use_cuda:
                target_amp = target_amp.cuda()
            
            pred_amp = experiment.postprocessing_net2amp(pred)
            
            pad_row = (pred_amp.shape[2] - target_amp.shape[2]) // 2
            pad_col = (pred_amp.shape[3] - target_amp.shape[3]) // 2

            print(f"pred_amp.shape: {pred_amp.shape}")
            print(f"pad_row: {pad_row}")
            print(f"pad_col: {pad_col}")
            print(f"target_amp.shape: {target_amp.shape}")

            if pad_row > 0:
                pred_amp = pred_amp[:, :, pad_row: -pad_row, :]
            if pad_col > 0:
                pred_amp = pred_amp[:, :, :, pad_col: -pad_col]
            if pad_col != 0 or pad_row != 0:
                print(f"error size {pad_col} {pad_row}")

            print(f"pred_amp.shape: {pred_amp.shape}")
            print(f"target_amp.shape: {target_amp.shape}")

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
    fp = os.path.join(outdir, 'time.txt')

    with open(fp, 'w') as f:
        for t in vetTIME:
            f.write(f"{t}\n")
    