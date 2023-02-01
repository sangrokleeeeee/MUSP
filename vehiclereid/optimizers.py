from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn


def init_optimizer(model,
                   optim='adam',  # optimizer choices
                   lr=0.003,  # learning rate
                   weight_decay=5e-4,  # weight decay
                   momentum=0.9,  # momentum factor for sgd and rmsprop
                   sgd_dampening=0,  # sgd's dampening for momentum
                   sgd_nesterov=False,  # whether to enable sgd's Nesterov momentum
                   rmsprop_alpha=0.99,  # rmsprop's smoothing constant
                   adam_beta1=0.9,  # exponential decay rate for adam's first moment
                   adam_beta2=0.999,  # # exponential decay rate for adam's second moment
                   staged_lr=False,  # different lr for different layers
                   new_layers=None,  # new layers use the default lr, while other layers's lr is scaled by base_lr_mult
                   base_lr_mult=0.1,  # learning rate multiplier for base layers
                   ):
    if staged_lr:
        assert new_layers is not None
        base_params = []
        base_layers = []
        new_params = []
        if isinstance(model, nn.DataParallel):
            model = model.module
        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
        param_groups = [
            {'params': base_params, 'lr': lr * base_lr_mult},
            {'params': new_params},
        ]
        print('Use staged learning rate')
        print('* Base layers (initial lr = {}): {}'.format(lr * base_lr_mult, base_layers))
        print('* New layers (initial lr = {}): {}'.format(lr, new_layers))
    else:
        param_groups = model.parameters()
    params = []
    protos = []
    for i, p in model.named_parameters():
        if 'prototype' not in i:
            params.append(p)
        else:
            protos.append(p)

    param_groups = [{'params': params}, {'params': protos, 'weight_decay': 0.,}]
    # Construct optimizer
    if optim == 'adam':
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2))

    elif optim == 'amsgrad':
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2), amsgrad=True)

    elif optim == 'sgd':
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay,
                               dampening=sgd_dampening, nesterov=sgd_nesterov)

    elif optim == 'rmsprop':
        return torch.optim.RMSprop(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                   alpha=rmsprop_alpha)

    else:
        raise ValueError('Unsupported optimizer: {}'.format(optim))


def init_recal_optimizer(model,
                   optim='adam',  # optimizer choices
                   lr=0.003,  # learning rate
                   weight_decay=5e-4,  # weight decay
                   momentum=0.9,  # momentum factor for sgd and rmsprop
                   sgd_dampening=0,  # sgd's dampening for momentum
                   sgd_nesterov=False,  # whether to enable sgd's Nesterov momentum
                   rmsprop_alpha=0.99,  # rmsprop's smoothing constant
                   adam_beta1=0.9,  # exponential decay rate for adam's first moment
                   adam_beta2=0.999,  # # exponential decay rate for adam's second moment
                   staged_lr=False,  # different lr for different layers
                   new_layers=None,  # new layers use the default lr, while other layers's lr is scaled by base_lr_mult
                   base_lr_mult=0.1,  # learning rate multiplier for base layers
                   ):
    if staged_lr:
        assert new_layers is not None
        base_params = []
        base_layers = []
        new_params = []
        if isinstance(model, nn.DataParallel):
            model = model.module
        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
        param_groups = [
            {'params': base_params, 'lr': lr * base_lr_mult},
            {'params': new_params},
        ]
        print('Use staged learning rate')
        print('* Base layers (initial lr = {}): {}'.format(lr * base_lr_mult, base_layers))
        print('* New layers (initial lr = {}): {}'.format(lr, new_layers))
    else:
        param_groups = model.parameters()

    bn_and_decoder = []
    encoder = []

    for name, param in model.named_parameters():
        if 'bn' in name or 'decoder' in name:
            bn_and_decoder.append(param)
        else:
            encoder.append(param)

    # Construct optimizer
    if optim == 'adam':
        return [torch.optim.Adam(encoder, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2)), 
                torch.optim.Adam(bn_and_decoder, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2))]

    elif optim == 'amsgrad':
        return [torch.optim.Adam(encoder, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2), amsgrad=True),
                torch.optim.Adam(bn_and_decoder, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2), amsgrad=True)]

    elif optim == 'sgd':
        return [torch.optim.SGD(encoder, lr=lr, momentum=momentum, weight_decay=weight_decay,
                               dampening=sgd_dampening, nesterov=sgd_nesterov),
                torch.optim.SGD(bn_and_decoder, lr=lr, momentum=momentum, weight_decay=weight_decay,
                               dampening=sgd_dampening, nesterov=sgd_nesterov)]

    elif optim == 'rmsprop':
        return [torch.optim.RMSprop(encoder, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                   alpha=rmsprop_alpha),
                torch.optim.RMSprop(bn_and_decoder, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                   alpha=rmsprop_alpha)]

    else:
        raise ValueError('Unsupported optimizer: {}'.format(optim))


def init_recal_two_optimizer(model,
                   optim='adam',  # optimizer choices
                   lr=0.003,  # learning rate
                   weight_decay=5e-4,  # weight decay
                   momentum=0.9,  # momentum factor for sgd and rmsprop
                   sgd_dampening=0,  # sgd's dampening for momentum
                   sgd_nesterov=False,  # whether to enable sgd's Nesterov momentum
                   rmsprop_alpha=0.99,  # rmsprop's smoothing constant
                   adam_beta1=0.9,  # exponential decay rate for adam's first moment
                   adam_beta2=0.999,  # # exponential decay rate for adam's second moment
                   staged_lr=False,  # different lr for different layers
                   new_layers=None,  # new layers use the default lr, while other layers's lr is scaled by base_lr_mult
                   base_lr_mult=0.1,  # learning rate multiplier for base layers
                   ):
    if staged_lr:
        assert new_layers is not None
        base_params = []
        base_layers = []
        new_params = []
        if isinstance(model, nn.DataParallel):
            model = model.module
        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
        param_groups = [
            {'params': base_params, 'lr': lr * base_lr_mult},
            {'params': new_params},
        ]
        print('Use staged learning rate')
        print('* Base layers (initial lr = {}): {}'.format(lr * base_lr_mult, base_layers))
        print('* New layers (initial lr = {}): {}'.format(lr, new_layers))
    else:
        param_groups = model.parameters()

    bn_and_decoder = []
    encoder = []

    for name, param in model.named_parameters():
        if 'decoder' in name or 'regular' in name:
            bn_and_decoder.append(param)
        else:
            encoder.append(param)

    # Construct optimizer
    if optim == 'adam':
        return [torch.optim.Adam(encoder, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2)), 
                torch.optim.Adam(bn_and_decoder, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2))]

    elif optim == 'amsgrad':
        return [torch.optim.Adam(encoder, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2), amsgrad=True),
                torch.optim.Adam(bn_and_decoder, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2), amsgrad=True)]

    elif optim == 'sgd':
        return [torch.optim.SGD(encoder, lr=lr, momentum=momentum, weight_decay=weight_decay,
                               dampening=sgd_dampening, nesterov=sgd_nesterov),
                torch.optim.SGD(bn_and_decoder, lr=lr, momentum=momentum, weight_decay=weight_decay,
                               dampening=sgd_dampening, nesterov=sgd_nesterov)]

    elif optim == 'rmsprop':
        return [torch.optim.RMSprop(encoder, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                   alpha=rmsprop_alpha),
                torch.optim.RMSprop(bn_and_decoder, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                   alpha=rmsprop_alpha)]

    else:
        raise ValueError('Unsupported optimizer: {}'.format(optim))


def init_bn_optimizer(model,
                   optim='adam',  # optimizer choices
                   lr=0.003,  # learning rate
                   weight_decay=5e-4,  # weight decay
                   momentum=0.9,  # momentum factor for sgd and rmsprop
                   sgd_dampening=0,  # sgd's dampening for momentum
                   sgd_nesterov=False,  # whether to enable sgd's Nesterov momentum
                   rmsprop_alpha=0.99,  # rmsprop's smoothing constant
                   adam_beta1=0.9,  # exponential decay rate for adam's first moment
                   adam_beta2=0.999,  # # exponential decay rate for adam's second moment
                   staged_lr=False,  # different lr for different layers
                   new_layers=None,  # new layers use the default lr, while other layers's lr is scaled by base_lr_mult
                   base_lr_mult=0.1,  # learning rate multiplier for base layers
                   ):
    if staged_lr:
        assert new_layers is not None
        base_params = []
        base_layers = []
        new_params = []
        if isinstance(model, nn.DataParallel):
            model = model.module
        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)
        param_groups = [
            {'params': base_params, 'lr': lr * base_lr_mult},
            {'params': new_params},
        ]
        print('Use staged learning rate')
        print('* Base layers (initial lr = {}): {}'.format(lr * base_lr_mult, base_layers))
        print('* New layers (initial lr = {}): {}'.format(lr, new_layers))
    else:
        param_groups = model.parameters()

    bn = []
    
    for name, param in model.named_parameters():
        # if 'bn' in name:
        #     bn.append(param)
        # if 'bn' in name:#'layer4' in name or 'layer3':# in name or 'layer2' in name or 'layer1' in name:
        #     bn.append(param)

        if 'bn' in name:
            bn.append(param)
    
    lr = lr * 0.01
    # Construct optimizer
    if optim == 'adam':
        return torch.optim.Adam(bn, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2))

    elif optim == 'amsgrad':
        return torch.optim.Adam(bn, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2), amsgrad=True)

    elif optim == 'sgd':
        return torch.optim.SGD(bn, lr=lr, momentum=momentum, weight_decay=weight_decay,
                               dampening=sgd_dampening, nesterov=sgd_nesterov)

    elif optim == 'rmsprop':
        return torch.optim.RMSprop(bn, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                   alpha=rmsprop_alpha)

    else:
        raise ValueError('Unsupported optimizer: {}'.format(optim))