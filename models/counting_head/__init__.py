
from .base_no_offset import build_counting_head as build_head_no_offset
from .base import build_counting_head as build_head
def build_counting_head(args):
    if args.name == "base":
        return build_head(args)
    elif args.name == "base_no_offset":
        return build_head_no_offset(args)
    raise NotImplementedError("{} is not supported".format(args.name))