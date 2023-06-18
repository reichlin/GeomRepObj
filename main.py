import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

from dataloader_2d import DatasetSim2D
from dataloader_3d import DatasetSim3D

from representation import Haptic_Repr, Transporter_Repr, VAE_Repr

from PPO import PPO
import mlagents
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


def sim(agent, env, T, r_tot, device, mode=0):

    obs = env.reset()

    for t in range(T):

        st = torch.from_numpy(np.expand_dims(np.transpose(obs[1], (2, 0, 1)), 0)).to(device)

        at, logprob, sigma = agent.get_action(st, test=(mode == 1))
        obs, reward, done, _ = env.step(at[0].detach().cpu().numpy())

        r_tot += reward/T

        if mode == 0:
            if t % T == (T - 1):
                done = True

            agent.push_batchdata(st.detach().cpu(), at.detach().cpu(), logprob.detach().cpu(), reward, done)

        if done:
            break

    return r_tot


def train_rl(env, algo, device):

    EPOCHS = 5000
    test_frq = 10
    test_epochs = 10
    batch_size = 4
    T = 20

    test_reward = []

    z_dim = 2+2
    a_dim = 2
    a_max = 1

    agent = PPO(algo, z_dim, a_dim, a_max, seed, device)

    for epoch in range(EPOCHS):

        r_tot = 0
        for i in range(batch_size):
            r_tot = sim(agent, env, T, r_tot, device, mode=0)

        v_loss, h_loss = agent.update()

        agent.clear_batchdata()

        if epoch % test_frq == (test_frq-1):

            r_tot = 0
            for test_epoch in range(test_epochs):
                r_tot = sim(agent, env, T, r_tot, device, mode=1)

            test_reward.append(r_tot / test_epochs)
            f_name = "./saved_results/rl_"+algo+".npy"
            np.save(f_name, np.array(test_reward))

            agent.save_model("./saved_models/", "rl_"+algo)


def train(dataloader, model, EPOCHS, name_exp):
    relative_err = []

    for e in range(0, EPOCHS):

        rel_err = 0

        for i, batch in enumerate(dataloader):

            model.net.train()

            metrics_logs = model.fit_Rep(batch, i, e)
            rel_err += metrics_logs['avg_isometry_err']

        relative_err.append(rel_err / len(dataloader))
        f_name = "./saved_results/" + name_exp + ".npy"
        np.save(f_name, np.array(relative_err))

    return relative_err, model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default="haptic", type=str, help="haptic, transporter, vae, end-to-end")
    parser.add_argument('--experiment', default="2d", type=str, help="2d, 3d, 3d_pov, rl")
    parser.add_argument('--rot', default=0, type=int, help="0: only position of the object, 1: consider rotations for the object")
    parser.add_argument('--move_bg', default=0, type=int, help="0: black background, 1: move background")

    parser.add_argument('--seed', default=0, type=int, help="seed")
    args = parser.parse_args()

    experiment = args.experiment
    move_bg = args.move_bg
    rotations = args.rot == 1

    channels = 4 if args.experiment == "3d_pov" else 3
    frq = 0.5
    N = 100
    model_type = args.model
    seed = args.seed
    batch_size = 100

    eigen_val1 = 0.01
    eigen_val2 = 5*eigen_val1 if rotations else eigen_val1

    name_exp = experiment+"_"+model_type+"_move_bg="+str(move_bg)+"_rot="+str(rotations)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.experiment == "rl":

        unity_env = UnityEnvironment("./3d_experiment/Unity_envs/Soccer_Final.x86_64")
        env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
        train_rl(env, args.model, device)
        env.close()

    else:

        if experiment == "2d":
            EPOCHS = 100
            dataset = DatasetSim2D(frq=frq, N=N, move_bg=move_bg, segment=True, batch_size=batch_size, rot=rotations)
        elif experiment == "3d":
            EPOCHS = 1000
            dataset = DatasetSim3D("./3d_experiment/dataset/")
        elif experiment == "3d_pov":
            EPOCHS = 1000
            dataset = DatasetSim3D("./3d_experiment/dataset/", pov=True)
        else:
            print("invalid experiment name")
            exit()

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0)

        if model_type == "haptic":
            a_dim = 2
            model = Haptic_Repr(experiment,
                                N,
                                channels,
                                True,
                                a_dim,
                                frq,
                                False,
                                rotations,
                                [eigen_val1, eigen_val2],
                                device)
        elif model_type == "transporter":
            k = 2
            channels = 3
            model = Transporter_Repr(dataset, k, channels, device)
        elif model_type == "vae":
            z_dim = 2
            h_features = 2
            channels = 3
            model = VAE_Repr(N, z_dim, h_features, channels, 1, device)
        else:
            print("invalid model type")
            exit()

        train(dataloader, model, EPOCHS, name_exp)

