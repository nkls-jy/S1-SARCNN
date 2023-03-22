import os 
import tensorboardX as tbx 
import torch
import DnCNN
import torch.nn as nn
#from . import DnCNN 

class Experiment:
    def __init__(self, basedir, expname=None):
        os.makedirs(basedir, exist_ok=True)

        if expname is None:
            self.expname = utils.get_exp_dir(basedir)
        else:
            self.expname = expname
        self.expdir = os.path.join(basedir, self.expname)

    def create_network(self):
        sizearea = self.args.sizearea
        network_weights = DnCNN.make_backnet(1,
                                            sizearea=sizearea,
                                            bn_momentum=0.1,
                                            padding=False
                                            )
        
        net = DnCNN.NlmCNN(network_weights, sizearea=sizearea, padding=False)

        return net

    def preprocessing_amp2net(self, img):
        return img.square()

    def postprocessing_net2amp(self, img):
        return img.abs().sqrt()

    def create_loss(self):
        def criterion(pred, targets):
            
            #loss = nn.MSELoss()
            #loss = loss(pred, targets)
            loss = ((pred + targets) / 2.0).abs().log() - (pred.abs().log() + targets.abs().log()) / 2 #glrt 
            
            loss = loss.view(pred.shape[0], -1)

            return loss
        return criterion

    def create_optimizer(self):
        args = self.args
        #assert(args.optimizer == 'adam')
        parameters = utils.parameters_by_module(self.net)
        if args.optimizer == 'sgd':
            self.base_lr = args.baseLr
            optimizer = torch.optim.SGD(parameters, lr=self.base_lr, momentum=0.9, weight_decay=0.3)
        else:
            self.base_lr = args.adam["lr"]
            optimizer = torch.optim.Adam(parameters, lr=self.base_lr, weight_decay=args.adam["weightdecay"],
                                        betas=(args.adam["beta1"], args.adam["beta2"]), eps=args.adam["eps"])
            # home machine path
            # bias parameters do not get weight decay
            for pg in optimizer.param_groups:
                if pg["name"] == "bias":
                    pg["weight_decay"] = 0

        return optimizer

    def learning_rate_decay(self, epoch):
        if epoch < 15:
            return 1
        elif epoch < 25:
            return 0.1
        elif epoch < 35:
            return 0.01
        else:
            return 0
        """
        if epoch < 20:
            return 1
        elif epoch < 40:
            return 0.1
        elif epoch < 50:
            return 0.01
        else:
            return 0
        """

    def setup(self, args=None, use_gpu=True):
        print(self.expname, self.expdir)
        os.makedirs(self.expdir, exist_ok=True)

        if args == None:
            self.args = utils.load_args(self.expdir)
        else:
            self.args = utils.args2obj(args)
            utils.save_args(self.expdir, self.args)

        writer_dir = os.path.join(self.expdir, 'train')
        os.makedirs(writer_dir, exist_ok=True)
        self.writer = tbx.SummaryWriter(log_dir = writer_dir)
        self.use_cuda = torch.cuda.is_available() and use_gpu
        self.net = self.create_network()
        self.optimizer = self.create_optimizer()
        self.criterion = self.create_loss()

        print(self.net)
        print(f"#Parameters {utils.parameter_count(self.net)}")

        self.epoch = 0

        if self.use_cuda:
            self.net.cuda()
    
    def add_summary(self, name, value, epoch=None):
        if epoch is None:
            epoch = self.epoch
        try:
            self.writer.add_scalar(name, value, epoch)
        except:
            pass

def main_sar(args):
    print(f"args: {args}")
    exp_basedir = args.exp_basedir
    patchsize = args.patchsize

    if args.weights:
        pass
        #from experiment_utility import load_checkpoint, test_list_weights
    elif args.eval:
        from experiment_utility import load_checkpoint, test_list
        from paths import list_testfiles
        
        assert(args.exp_name is not None)
        experiment = Experiment(exp_basedir, args.exp_name)
        experiment.setup(use_gpu=args.use_gpu)
        load_checkpoint(experiment, args.eval_epoch)
        outdir = os.path.join(experiment.expdir, f"results{args.eval_epoch}")
        test_list(experiment, outdir, list_testfiles, pad=22)
    else:
        from experiment_utility import trainloop
        from dataloader import PreprocessingInt as Preprocessing
        from dataloader import create_train_realsar_dataloaders as create_train_dataloaders
        from dataloader import create_valid_realsar_dataloaders as create_valid_dataloaders

        experiment = Experiment(exp_basedir, args.exp_name)
        experiment.setup(args, use_gpu=args.use_gpu)

        trainloader = create_train_dataloaders(patchsize, args.batchsize, args.trainsetiters)
        #validloader = create_valid_dataloaders(args.patchsizevalid, args.batchsizevalid)
        # without validation data
        trainloop(experiment, trainloader, Preprocessing(), log_data=False, validloader=None)
        # with validation data
        #trainloop(experiment, trainloader, Preprocessing(), log_data=False, validloader=validloader)

if __name__ == '__main__':
    import argparse
    import os
    import utils
    #from . import utils
    import utils
    import torch
    
    parser = argparse.ArgumentParser(description='NLM for SAR image denoising')

    parser.add_argument("--sizearea", type=int, default=25) #default=31) #default=25)

    # Optimizer
    parser.add_argument('--optimizer', default="adam", choices=["adam", "sgd"]) # which optimizer to use
    # parameters for SGD
    parser.add_argument("--baseLr", type=float, default=0.000001)
    # parameters for Adam
    parser.add_argument("--adam.beta1", type=float, default=0.9)
    parser.add_argument("--adam.beta2", type=float, default=0.999)
    parser.add_argument("--adam.eps", type=float, default=1e-8)
    parser.add_argument("--adam.weightdecay", type=float, default=0.01) #default=1e-4)
    parser.add_argument('--adam.lr', type=float, default=0.001) # original=0.001

     # Eval mode
    parser.add_argument('--eval', default=False) #False) # action='store_false')
    parser.add_argument('--weights', default=False) # action='store_false')
    parser.add_argument('--eval_epoch', type=int, default=35) #default=50

     # Training options
    parser.add_argument("--batchsize"     , type=int, default= 16) # for home machine: 16
    parser.add_argument("--patchsize"     , type=int, default=48)# 60)#default=48)
    parser.add_argument("--batchsizevalid", type=int, default=8)
    parser.add_argument("--patchsizevalid", type=int, default=256) # original: default=256) but currently no big valid patches available

     # Misc
    utils.add_commandline_flag(parser, "--use_gpu", "--use_cpu", True)
    parser.add_argument("--exp_name", default=None) #'exp0008_SLC_d10_k531_lr001_area35_300iterations') #None)

    # base experiment dir
    base_expdir = "/home/niklas/Documents/mySARCNN_Experiment"
    # base_expdir = "/home/niklas/Documents/Checkpoints/SLC"
    parser.add_argument("--exp_basedir", default=base_expdir)
    parser.add_argument("--trainsetiters", type=int, default=100) # original: 640
    args = parser.parse_args()
    main_sar(args)

# execute on command line to access tensorboard interface
# change exp000% each time 
# tensorboard --logdir=/home/niklas/Documents/mySARCNN_Experiment/exp0000/train/


