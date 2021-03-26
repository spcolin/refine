import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        # ----------------added args---------------------
        # parser.add_argument('--anno_pack_path', help='Path to package containing annotation json files',default="E:/anno")
        parser.add_argument('--anno_pack_path', help='Path to package containing annotation json files',default="E:/rd/datasets/NYUDV2/annotations")

        parser.add_argument('--refine', default=False, help='start refine')
        # ----------------added end----------------------


        parser.add_argument('--dataroot',  help='Path to images',default="datasets/NYUDV2")
        parser.add_argument('--batchsize', type=int, default=2, help='Batch size')
        parser.add_argument('--cfg_file', default='lib/configs/resnext101_32x4d_nyudv2_class',
                            help='Set model and dataset config files')
        parser.add_argument('--dataset', default='nyudv2', help='Path to images')
        parser.add_argument('--load_ckpt', default=False, help='Checkpoint path to load')
        parser.add_argument('--resume', default=False, help='Resume to train')
        parser.add_argument('--epoch', default=10, type=int, help='Set training epochs')
        parser.add_argument('--start_epoch', default=0, type=int, help='Set training epochs')
        parser.add_argument('--start_step', default=0, type=int, help='Set training steps')
        parser.add_argument('--thread', default=0, type=int, help='Thread for loading data')
        parser.add_argument('--use_tfboard', default=True, help='Tensorboard to log training info')
        parser.add_argument('--results_dir', type=str, default='./evaluation', help='Output dir')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt
