#!/usr/bin/env python

"""
Behavorial cloning of an expert policy using a simple Feedforward Neural Network.
Example usage:
    python behavior_cloning.py experts/RoboschoolHumanoid-v1.py RoboschoolHumanoid-v1_20_data.pkl \
    --render --num_rollouts 20
"""

import pickle
import numpy as np
import tensorflow as tf
import gym
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
import importlib

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    return data

def build_policy_network(inputs):
    a3 = tf.contrib.slim.stack(inputs, tf.contrib.slim.fully_connected, [128, 128, 128], scope='fc') 
    y_hat = tf.contrib.slim.fully_connected(a3, 17, scope='fc/fc4', activation_fn=None)  # linear activation
    return y_hat

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('data_file', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument("--max_timesteps", type=int)
    args = parser.parse_args()

    print('loading expert policy')
    module_name = args.expert_policy_file.replace('/', '.')
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    policy_module = importlib.import_module(module_name)
    print('loaded')

    env, policy = policy_module.get_env_and_policy()
    max_steps = args.max_timesteps or env.spec.timestep_limit
    print('loaded and built')

    # Set Parameters
    # task = 'Humanoid-v1'
    # task = 'HalfCheetah-v1'
    # task = 'Hopper-v1' 
    # task = 'Reacher-v1'
    # task = 'Ant-v1'
    # task = 'Walker2d-v1'
    task_data = args.data_file

    X = tf.placeholder(tf.float32, shape=[None, 44])
    Y = tf.placeholder(tf.float32, shape=[None, 17])

    # build network and make an inference  
    y_hat = build_policy_network(X)
    # calculate loss(MSE)
    loss = tf.contrib.slim.losses.mean_squared_error(Y, y_hat)

    optimizer = tf.train.AdamOptimizer() # adam default params are usually solid
    train_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
        
    #  imitation learning
    with tf.Session() as sess:
        # Load in expert policy observation data
        last_checkpoint = tf.train.latest_checkpoint('checkpoints/')
        if last_checkpoint: 
            saver.restore(sess, last_checkpoint)
        data = load_data(task_data)
        obs_data = np.array(data['observations'])
        act_data = np.array(data['actions'])

        # Split data into train and test set
        n = obs_data.shape[0]
        obs_data, act_data = shuffle(obs_data, act_data, random_state=0)
        split_val = int(n*0.8)
        X_train = obs_data[:split_val]
        X_test = obs_data[split_val:]
        Y_train = act_data[:split_val]
        Y_test = act_data[split_val:]
        sess.run(init)
        for i in range(3):
            print('epoch: '+str(i))
            mini_batch_sz = 256
            mini_batches = [[np.array(X_train[j:j+n]), np.array(Y_train[j:j+n])] \
                                   for j in range(0, X_train.shape[0], mini_batch_sz)]
            for batch_i, (mini_batch_X, mini_batch_Y) in enumerate(mini_batches):
                _, train_mse_loss = sess.run([train_op, loss], feed_dict={X: mini_batch_X, Y: mini_batch_Y})
                if batch_i % 10 == 0:
                        print('   batch_i:'+str(batch_i) + ' train mse:' + str(train_mse_loss))
            test_mse_loss = sess.run([loss], feed_dict={X:X_test, Y: Y_test}) 
            print('epoch: '+str(i) + ' test mse:' + str(test_mse_loss))
        saver.save(sess, 'checkpoints/'+args.expert_policy_file.split('/')[-1].replace('.py', ''))
         
    with tf.Session() as sess:
        sess.run(init)
        last_checkpoint = tf.train.latest_checkpoint('checkpoints/')
        saver.restore(sess, last_checkpoint)
        
        returns = []
        new_observations = []
        new_exp_actions = []

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                exp_action = policy.act(obs)
                action = sess.run(y_hat, feed_dict={X: obs.reshape(-1, 44)}).ravel()
                new_observations.append(obs)
                new_exp_actions.append(exp_action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

if __name__ == '__main__':
    main()
