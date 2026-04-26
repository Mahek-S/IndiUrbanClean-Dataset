from torchvision import transforms

def get_transforms(strategy):
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]

    if strategy == "rotation":
        aug = [transforms.RandomRotation(30)]
        return transforms.Compose(aug + base)

    elif strategy == "crop":
        aug = [transforms.RandomResizedCrop(224)]
        return transforms.Compose(aug + base)

    elif strategy == "color":
        aug = [transforms.ColorJitter(brightness=0.5, contrast=0.5)]
        return transforms.Compose(aug + base)

    elif strategy == "cutout":
        return transforms.Compose(base + [transforms.RandomErasing(p=0.5)])

    else:
        return transforms.Compose(base)