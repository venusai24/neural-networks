# This file contains functions to initialize and load models, set up classifiers, and define loss functions. 
# It includes functions to handle different model architectures (ResNet for PyTorch and CIFAR) and initialize classifiers
# with specific parameters for training.

import torch
import torch.nn as nn
try:
    import resnet_pytorch
    import resnet_cifar_modified as resnet_cifar
    import resnet_original
    import custom
except ModuleNotFoundError:
    from modifications import resnet_pytorch_modified 
    from modifications import resnet_cifar_modified as resnet_cifar
    from modifications import resnet_pytorch
    from modifications import resnet_cifar
    from modifications import resnet_original
    from modifications import custom

def _mismatched_classifier(model,pretrained):
    classifier_name, old_classifier = model._modules.popitem()
    classifier_input_size = old_classifier[1].in_features
    
    pretrained_classifier = nn.Sequential(
                nn.LayerNorm(classifier_input_size),
                nn.Linear(classifier_input_size, 1000)
            )
    model.add_module(classifier_name, pretrained_classifier)
    state_dict = torch.load(pretrained, map_location='cpu')
    model.load_state_dict(state_dict['model'],strict=False)

    classifier_name, new_classifier = model._modules.popitem()
    model.add_module(classifier_name, old_classifier)
    return model

def get_model(args,num_classes):
    try:
        print(f'resnet_pytorch.{args.model}(num_classes={num_classes},use_norm="{args.classif_norm}",use_gumbel={args.use_gumbel_se},use_gumbel_cb={args.use_gumbel_cb},pretrained="{args.pretrained}")')
        model = eval(f'resnet_pytorch.{args.model}(num_classes={num_classes},use_norm="{args.classif_norm}",use_gumbel={args.use_gumbel_se},use_gumbel_cb={args.use_gumbel_cb},pretrained="{args.pretrained}")')
    except AttributeError:
        try:
            try:
                model = eval(f'resnet_cifar.{args.model}(num_classes={num_classes},use_norm="{args.classif_norm}",use_gumbel={args.use_gumbel_se},use_gumbel_cb={args.use_gumbel_cb})')
            except AttributeError:
                model = eval(f'resnet_original.{args.model}(num_classes={num_classes})')
        except TypeError:
            model = eval(f'resnet_cifar.{args.model}(num_classes={num_classes},use_norm="{args.classif_norm}")')
            
    model = initialise_classifier(args,model,num_classes)
    return model

def get_weights(dataset):
    per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
    per_cls_weights = per_cls_weights.sum()/per_cls_weights
    return per_cls_weights

def get_criterion(args,dataset,model=None):
    if args.deffered:
        weight=get_weights(dataset)
    else:
        weight=None
    if args.criterion =='ce':
        return torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing,weight=weight)
    elif args.criterion =='iif':
        return custom.IIFLoss(dataset,weight=weight,variant=args.iif,label_smoothing=args.label_smoothing)
    elif args.criterion =='bce':
        return custom.BCE(label_smoothing=args.label_smoothing,reduction=args.reduction)
        


def initialise_classifier(args,model,num_classes):
    num_classes = torch.tensor([num_classes])
    if args.criterion == 'bce':
        if args.dset_name.startswith('cifar'):
            torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
        else:
            torch.nn.init.normal_(model.fc.weight.data,0.0,0.001)
        try:
            if args.dset_name.startswith('cifar'):
                torch.nn.init.constant_(model.linear.bias.data,-6.0)
            else:
                torch.nn.init.constant_(model.fc.bias.data,-6.0)
        except AttributeError:
            print('no bias in classifier head')
            pass
    return model
        
    
