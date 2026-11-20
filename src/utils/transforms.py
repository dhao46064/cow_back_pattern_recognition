from torchvision import transforms


# def build_transforms(img_size=224):
#     return transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#         ),
#     



def build_transforms(img_size=224, train=False):

    if train:
        return transforms.Compose([

            # 旋转 + 平移 + 缩放
            transforms.RandomAffine(
                degrees=5,                # ±5°
                translate=(0.05, 0.05),   # 平移 ±5%
                scale=(0.95, 1.05)        # 缩放
            ),

            # 亮度变化
            transforms.ColorJitter(
                brightness=0.2
            ),

            transforms.Resize((img_size, img_size)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    else:
        return transforms.Compose([

            transforms.Resize((img_size, img_size)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])