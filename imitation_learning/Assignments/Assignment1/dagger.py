import train_policy
import racer
import argparse
import os

from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool
from full_state_car_racing_env import FullStateCarRacingEnv
import imageio
import numpy as np

def run(steering_network, args, beta):
    
    env = FullStateCarRacingEnv()
    env.reset()
    
    learner_action = np.array([0.0, 0.0, 0.0])
    expert_action = None
    
    # bernoulli distribution for the timesteps
    flag = np.random.binomial(1, beta, size=args.timesteps)
    for t in range(args.timesteps):
        
        env.render()
        
        state, expert_action, reward, done, _ = env.step(learner_action) 
        if done:
            break
        
        expert_steer = expert_action[0]  # [-1, 1]
        expert_gas = expert_action[1]    # [0, 1]
        expert_brake = expert_action[2]  # [0, 1]

        if flag[t]:
            learner_action[0] = expert_steer
        else:
            learner_action[0] = steering_network.eval(state, device=DEVICE)
            
        learner_action[1] = expert_gas
        learner_action[2] = expert_brake

        if args.save_expert_actions:
            imageio.imwrite(os.path.join(args.out_dir, 'expert_%d_%d_%f.jpg' % (args.run_id, t, expert_steer)), state)

    env.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", help="", default=10)
    parser.add_argument("--weighted_loss", help="", default=False)
    parser.add_argument("--timesteps", type=int, help="timesteps of simulation to run, up to one full loop of the track", default=100000)
    parser.add_argument("--save_expert_actions", type=str2bool, help="save the images and expert actions in the training set",
                        default=True)
    parser.add_argument("--run_id", type=int, help="Id for this particular data collection run (e.g. dagger iterations)", default=0)
    parser.add_argument("--out_dir", help="directory in which to save the expert's data", default='./dataset/train')
    args = parser.parse_args()

    #####
    ## Enter your DAgger code here
    ## Reuse functions in racer.py and train_policy.py
    ## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where 
    #####
    
    print ('TRAINING LEARNER ON INITIAL DATASET')
    args.weights_out_file = f"./weights/learner_0.weights"
    policy = train_policy.main(args)
    steering_network = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    for iter in range(args.dagger_iterations):
        weights = policy.state_dict()
        steering_network.load_state_dict( {k:v for k,v in weights.items()}, strict=True)
    
        if iter == 0:
            beta = 1
        else:
            p = 0.5 # as mentioned in the paper, best experimental results were obtained for p = 0.5
            beta = p**(iter)

        print ('GETTING EXPERT DEMONSTRATIONS')
        args.run_id = iter
        run(steering_network, args, beta)


        args.weights_out_file = f"./weights/learner_{iter}.weights"
        print(args.weights_out_file)
        print ('RETRAINING LEARNER ON AGGREGATED DATASET')
        policy = train_policy.main(args)
