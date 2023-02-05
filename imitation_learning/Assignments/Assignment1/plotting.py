import matplotlib.pyplot as plt
import torch
import argparse 
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool
import numpy as np
import racer

def plotting(heading_errors, dist_errors):
    plt.title("Cross Tracking Error")
    n = len(heading_errors)
    plt.plot(range(1, n+1), heading_errors, label="Heading Errors")
    plt.plot(range(1, n+1), dist_errors, label="Distance Errors")
    plt.xlabel("Number of Dagger Iterations")
    plt.ylabel("Cross-Track Error")
    plt.legend(loc="best")
    plt.show()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", help="directory in which to save the expert's data", default='./dataset/train')
    parser.add_argument("--save_expert_actions", type=str2bool, help="save the images and expert actions in the training set",
                        default=False)
    
    parser.add_argument("--expert_drives", type=str2bool, help="should the expert steer the vehicle?", default=False)
    parser.add_argument("--run_id", type=int, help="Id for this particular data collection run (e.g. dagger iterations)", default=0)
    parser.add_argument("--timesteps", type=int, help="timesteps of simulation to run, up to one full loop of the track", default=100000)
    parser.add_argument("--learner_weights", type=str, help="filename from which to load learner weights for the steering network",
                        default='')
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=21)
    parser.add_argument("--dagger_iterations", help="", default=10)
    args = parser.parse_args()
    
    dist_errors = []
    heading_errors = []
    steering_network = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    for iter in range(args.dagger_iterations):
        args.learner_weights = f"./weights/learner_{iter}.weights"
        steering_network.load_weights_from(args.learner_weights)

        # calculate the error
        heading_error, dist_error = racer.run(steering_network, args)
        print(heading_error, dist_error)
        heading_errors.append(heading_error)
        dist_errors.append(dist_error)

    plotting(heading_errors, dist_errors)