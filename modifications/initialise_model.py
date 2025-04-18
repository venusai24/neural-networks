import torch
import torch.nn as nn
try:
    import resnet_cifar
    import custom
except ModuleNotFoundError:
    from modifications import resnet_cifar
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

def get_model(args, num_classes):
    if args.model == 'resnet20_apa':
        from resnet_cifar import resnet20_apa
        model = resnet20_apa(num_classes=num_classes)
    else:
        if args.model == 'resnet20':
            from resnet_cifar import resnet20
            model = resnet20(num_classes=num_classes)
        elif args.model == 'resnet32':
            from resnet_cifar import resnet32
            model = resnet32(num_classes=num_classes)
        elif args.model == 'resnet44':
            from resnet_cifar import resnet44
            model = resnet44(num_classes=num_classes)
        elif args.model == 'resnet56':
            from resnet_cifar import resnet56
            model = resnet56(num_classes=num_classes)
        elif args.model == 'resnet110':
            from resnet_cifar import resnet110
            model = resnet110(num_classes=num_classes)
        elif args.model == 'resnet1202':
            from resnet_cifar import resnet1202
            model = resnet1202(num_classes=num_classes)
        else:
            raise ValueError("Unknown model type: {}".format(args.model))
    return model

def get_weights(dataset):
    per_cls_weights = torch.tensor(dataset.get_cls_num_list()).to('cuda')
    per_cls_weights = per_cls_weights.sum()/per_cls_weights
    return per_cls_weights

def get_criterion(args,dataset,model=None):
    if args.deffered:
        weight=get_weights(dataset)
    else:
        weight=None
    if args.criterion =='ce':
        return torch.nn.CrossEntropyLoss(label_smoothing=0.1,weight=weight)
    elif args.criterion =='ldam':
         return custom.LDAMLoss(dataset.get_cls_num_list(),max_m=0.5,weight=weight,s=30)
       
        
def initialise_classifier(args, model, num_classes):
    """
    Initializes the classifier layer weights and bias for APA Focal Loss.
    """
    if args.criterion == 'APAFocalLoss':
        # For CIFAR models, classifier is usually model.linear; otherwise model.fc
        try:
            if args.dset_name.startswith('cifar'):
                torch.nn.init.normal_(model.linear.weight.data, 0.0, 0.001)
                if hasattr(model.linear, 'bias') and model.linear.bias is not None:
                    torch.nn.init.constant_(model.linear.bias.data, 0.0)
            else:
                torch.nn.init.normal_(model.fc.weight.data, 0.0, 0.001)
                if hasattr(model.fc, 'bias') and model.fc.bias is not None:
                    torch.nn.init.constant_(model.fc.bias.data, 0.0)
        except AttributeError:
            print('No classifier head found or no bias in classifier head')
    return model