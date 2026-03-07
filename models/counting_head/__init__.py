from .uppernet import build_counting_head as build_head_uppernet
from .simple_offset import build_counting_head as build_head_simple_offset
from .p2p import build_counting_head as build_head_p2p
from .steerer_48 import build_counting_head as build_head_steerer_48
def build_counting_head(args):
    if args.name == "simple_offset":
        return build_head_simple_offset(args)
    elif args.name == "p2p":
        return build_head_p2p(args)
    elif args.name == "steerer_48":
        return build_head_steerer_48(args)
    raise NotImplementedError("{} is not supported".format(args.name))