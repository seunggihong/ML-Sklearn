import argparse
from utils.selection import select_model
from utils.load import load_params

parser = argparse.ArgumentParser()
parser.add_argument('--prob', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

params = load_params(args.prob, args.model)

print("\nProblem > %s \tModel > %s\n" % (args.prob, args.model))

if args.prob == 'reg':
    print("Dataset > winequality-red.csv\n")
    score, estimator = select_model(args.model, args.prob, params)

    print("Best Parameters  : {}\n".format(estimator))
    print("mse  : {:10.4f}\nrmse : {:10.4f}\nr2   : {:10.4f}".format(
        score[0], score[1], score[2]))

elif args.prob == 'class':
    print("Dataset > heart.csv\n")
    evaluation, estimator = select_model(args.model, args.prob, params)

    print("Best Parameters  : {}\n".format(estimator))
    print("accuracy  : {:10.4f}\nprecision : {:10.4f}\nrecall    : {:10.4f}\nf1        : {:10.4f}\n".format(
        evaluation[0], evaluation[1], evaluation[2], evaluation[3]))
