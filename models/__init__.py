
from .counting_model import build_counting_model
def build_model(args):
    if args.name == "counting":
        return build_counting_model(args)
