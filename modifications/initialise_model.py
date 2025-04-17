def get_model(args, num_classes):
    if args.model == 'resnet20_apa':
        from modifications.resnet_cifar import resnet20_apa
        model = resnet20_apa(num_classes=num_classes)
    else:
        if args.model == 'resnet20':
            from modifications.resnet_cifar import resnet20
            model = resnet20(num_classes=num_classes)
        elif args.model == 'resnet32':
            from modifications.resnet_cifar import resnet32
            model = resnet32(num_classes=num_classes)
        elif args.model == 'resnet44':
            from modifications.resnet_cifar import resnet44
            model = resnet44(num_classes=num_classes)
        elif args.model == 'resnet56':
            from modifications.resnet_cifar import resnet56
            model = resnet56(num_classes=num_classes)
        elif args.model == 'resnet110':
            from modifications.resnet_cifar import resnet110
            model = resnet110(num_classes=num_classes)
        elif args.model == 'resnet1202':
            from modifications.resnet_cifar import resnet1202
            model = resnet1202(num_classes=num_classes)
        else:
            raise ValueError("Unknown model type: {}".format(args.model))
    return model