from .timm_model import build_backbone as build_backbone_timm
from .vgg import vgg19
# from .convnext import convnext_nano_ols
def build_backbone(args):
    if args.name in ["vgg19"]:
        return vgg19()
    # if args.name.startswith("convnext_nano"):
    #     return convnext_nano_ols(pretrained=args.pretrained)
    # else:
    #     return build_backbone_transformer(args)
    return build_backbone_timm(args)