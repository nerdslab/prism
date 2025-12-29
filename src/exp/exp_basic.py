import os
import torch
#from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
#    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
#    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
#    WPMixer, MultiPatchFormer

# from src.arch.god_dynamight import god_dynamight
# from src.arch.exp_dynamight import ExpDynamight
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.args = args
        # self.device = self._acquire_device()
        try:
            self.model = self._build_model().to(args.rank)
        except:
            self.model = self._build_model().cuda()

    def _build_model(self):
        raise NotImplementedError
    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
