from .densemap import DrawDenseMap
from .locationmap import DrawLocationMap    
from .densemap_large import DrawLargeDenseMap
from .densemap_b3 import DrawDenseMapB3


def build_label_processing(args):
    if args.type == "dmap":
        return DrawDenseMap(args)
    if args.type == "lmap":
        return DrawLocationMap(args)
    if args.type == "large_dmap":
        return DrawLargeDenseMap(args)
    if args.type == "dmap_b3":
        return DrawDenseMapB3(args)

    raise Exception("Type {} not supported".format(args.type))