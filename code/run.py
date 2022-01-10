import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d',default='citeseer')
parser.add_argument('--gpu', '-g',default=0)
parser.add_argument('--ratio', '-r',default=1)
parser.add_argument('--rho', '-rh',default='0.3,0.6,0.8')
args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    os.system("python prepareData.py --ratio %s --dataset %s" % (args.ratio, args.dataset))
    os.system("python train.py --dataset %s --rho %s" % (args.dataset, args.rho))
    os.system("python auc.py")
