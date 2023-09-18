
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""
from models.model_plain import ModelPlain

def select_Model(opt):
    model = opt['model']

    if model == 'plain':
        m = ModelPlain(opt)
    elif model == 'gan':     # one input: L
        # from models.model_gan import ModelGAN as M
        pass
    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    print('Training model [{:s}] is created.'.format(model))
    return m
