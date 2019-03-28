from Coach import Coach
from p4.P4Game import P4Game as Game
from p4.keras.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 80,
    'numEps': 200,
    'tempThreshold': 15,
    'updateThreshold': 0.51,
    'maxlenOfQueue': 1000000,
    'numMCTSSims': 25,
    'arenaCompare': 50,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
