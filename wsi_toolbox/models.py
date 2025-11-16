import os
import torch
import timm
from timm.layers import SwiGLUPacked


MODEL_LABELS = {
    'uni':  'UNI',
    'gigapath':  'Prov-Gigapath',
    'virchow2':  'Virchow2',
}
MODEL_NAMES_BY_LABEL = {v: k for k, v in MODEL_LABELS.items()}
MODEL_NAMES = list(MODEL_LABELS.keys())


def get_model_label(model_name) -> str:
    return MODEL_LABELS.get(model_name, model_name)

def create_model(model_name):
    if model_name == 'uni':
        return timm.create_model('hf-hub:MahmoodLab/uni',
                                 pretrained=True,
                                 dynamic_img_size=True,
                                 init_values=1e-5)

    if model_name == 'gigapath':
        return timm.create_model('hf_hub:prov-gigapath/prov-gigapath',
                                 pretrained=True,
                                 dynamic_img_size=True)

    if model_name == 'virchow2':
        return timm.create_model('hf-hub:paige-ai/Virchow2',
                                  pretrained=True,
                                  mlp_layer=SwiGLUPacked,
                                  act_layer=torch.nn.SiLU)

    raise ValueError('Invalid model_name', model_name)

