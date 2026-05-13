"""Transforms para os três cenários experimentais."""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_transforms(scenario: str, split: str) -> transforms.Compose:
    """
    Retorna transforms para o split e cenário dados.

    scenario: 'A' (baseline), 'B' (augmentation clássica), 'C' (com sintéticas)
    split: 'train', 'val', 'test'
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split != "train" or scenario == "A":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])

    # Cenários B e C — augmentação clássica no train
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        ),
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.ToTensor(),
        normalize,
    ])
