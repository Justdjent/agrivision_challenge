from models import make_model
from params import args
from tqdm import tqdm
tqdm.monitor_interval = 0


def convert():
    model = make_model((None, None, 3))
    model.load_weights(args.weights)
    model.save(args.weights.replace(".h5", "_full.h5"))


if __name__ == '__main__':
    convert()