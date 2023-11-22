import argparse
from utils.selection import select_model

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

print("\nProblem > %s \tModel > %s\n" % (args.problem, args.model))

if args.problem == 'regression':
    print("Dataset > winequality-red.csv\n")
    score = select_model(args.model, args.problem)
    print("mse  : {:10.4f}\nrmse : {:10.4f}\nr2   : {:10.4f}".format(
        score[0], score[1], score[2]))

elif args.problem == 'classification':
    print("Dataset > heart.csv\n")
    evaluation = select_model(args.model, args.problem)
    print("accuracy  : {:10.4f}\nprecision : {:10.4f}\nrecall    : {:10.4f}\nf1        : {:10.4f}".format(
        evaluation[0], evaluation[1], evaluation[2], evaluation[3]))
