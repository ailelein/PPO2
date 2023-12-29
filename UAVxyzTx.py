import copy
import os
from datetime import datetime
from PPO import PPO
from parameters import *
from envplot import guinfo
from gym import Env
from gym.spaces import Box
import numpy as np
import calculations as cal
import torch


class CustomEnv(Env):
    def __init__(self):

        LSUPos = [xMinEnv, yMinEnv, zMinEnv]
        LJUPos = [xMinEnv, yMinEnv, zMinEnv]
        MEUPos = [xMinEnv, yMinEnv, zMinEnv]
        MJUPos = [xMinEnv, yMinEnv, zMinEnv]
        GUallInfo = []
        for _ in range(NumGUs):
            GUallInfo.append(odMin)  # GUoffloadingdata
            for _ in range(2):
                GUallInfo.append(0)  # GUxaxis, GUyaxis
        LSUPosHistory = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        LSUtxHistory = [0,0,0,0,0,0,0,0]

        # LJUPosHistory = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        odhistorySingle = [0 for _ in range(NumGUs*8)]
        # guTXHistory = [0 for _ in range(NumGUs*8)]

        obs1Min = []
        obs1Min.extend(LSUPos)
        obs1Min.extend(LJUPos)
        obs1Min.extend(MEUPos)
        obs1Min.extend(MJUPos)
        obs1Min.extend(GUallInfo)
        obs1Min.extend(LSUPosHistory)
        obs1Min.extend(LSUtxHistory)

        # obs1Min.extend(LJUPosHistory)
        obs1Min.extend(odhistorySingle)
        # obs1Min.extend(guTXHistory)
        obs1low = np.array(obs1Min).astype(np.float32)

        LSUPosUP = [xMaxEnv, yMaxEnv, zMaxEnv]
        LJUPosUP = [xMaxEnv, yMaxEnv, zMaxEnv]
        MEUPosUP = [xMaxEnv, yMaxEnv, zMaxEnv]
        MJUPosUP = [xMaxEnv, yMaxEnv, zMaxEnv]
        GUallInfoUP = []
        for _ in range(NumGUs):
            GUallInfoUP.append(odMax)  # GUoffloadingdata
            for _ in range(2):
                GUallInfoUP.append(xMaxEnv)  # GUxaxis, GUyaxis
        LSUPosHistoryUP = [xMaxEnv, yMaxEnv, zMaxEnv, xMaxEnv, yMaxEnv, zMaxEnv, xMaxEnv, yMaxEnv,
                           zMaxEnv, xMaxEnv, yMaxEnv, zMaxEnv, xMaxEnv, yMaxEnv, zMaxEnv,
                           xMaxEnv, yMaxEnv, zMaxEnv, xMaxEnv, yMaxEnv, zMaxEnv, xMaxEnv, yMaxEnv,
                           zMaxEnv]
        LSUtxHistoryUP = [lsuTxMax, lsuTxMax, lsuTxMax, lsuTxMax, lsuTxMax, lsuTxMax, lsuTxMax, lsuTxMax]

        # LJUPosHistoryUP = [xMaxEnv, yMaxEnv, zMaxEnv, lsuTxMax, xMaxEnv, yMaxEnv, zMaxEnv, lsuTxMax, xMaxEnv, yMaxEnv,
        #                    zMaxEnv, lsuTxMax, xMaxEnv, yMaxEnv, zMaxEnv, lsuTxMax, xMaxEnv, yMaxEnv, zMaxEnv, lsuTxMax,
        #                    xMaxEnv, yMaxEnv, zMaxEnv, lsuTxMax, xMaxEnv, yMaxEnv, zMaxEnv, lsuTxMax, xMaxEnv, yMaxEnv,
        #                    zMaxEnv, lsuTxMax]
        odhistorySingleUP = [odMax for _ in range(NumGUs*8)]
        # guTXHistoryUP = [guTxMax for _ in range(NumGUs*8)]

        obs1Max = []
        obs1Max.extend(LSUPosUP)
        obs1Max.extend(LJUPosUP)
        obs1Max.extend(MEUPosUP)
        obs1Max.extend(MJUPosUP)
        obs1Max.extend(GUallInfoUP)
        obs1Max.extend(LSUPosHistoryUP)
        obs1Max.extend(LSUtxHistoryUP)

        # obs1Max.extend(LJUPosHistoryUP)
        obs1Max.extend(odhistorySingleUP)
        # obs1Max.extend(guTXHistoryUP)
        obs1high = np.array(obs1Max).astype(np.float32)

        self.observation1_space = Box(obs1low, obs1high)

        'lsu_x, lsu_y, lsu_h, lju_x,lsu_tx, lju_y, lju_h, lju_tx + 50 gu tx'
        lsuPosTx = [-1, -1, -1, -1]
        # ljuPosTx = [-1, -1, -1, -1]
        # gutxDOWN = [guTxMin for _ in range(NumGUs)]
        ac1Min = []
        ac1Min.extend(lsuPosTx)
        # ac1Min.extend(ljuPosTx)
        # ac1Min.extend(gutxDOWN)
        ac1low = np.array(ac1Min).astype(np.float32)

        lsuPosTxUP = [1, 1, 1,1]
        # ljuPosTxUP = [1, 1, 1, 1]
        # gutxUP = [guTxMax for _ in range(NumGUs)]
        ac1Max = []
        ac1Max.extend(lsuPosTxUP)
        # ac1Max.extend(ljuPosTxUP)
        # ac1Max.extend(gutxUP)
        ac1high = np.array(ac1Max).astype(np.float32)
        self.action1_space = Box(ac1low, ac1high)

    def reset1obs(self, copy_guinfo, LSUPosHistory, LSUTxHistory, odhistoryFull):

        statereset= []
        lsuPos= [lsuInitPos[0], lsuInitPos[1], lsuInitPos[2]]
        ljuPos = [ljuInitPos[0], ljuInitPos[1], ljuInitPos[2]]
        meupos = [meuPos[0], meuPos[1], meuPos[2]]
        mjupos = [mjuPos[0], mjuPos[1], mjuPos[2]]
        GUallInfo = []
        for i in copy_guinfo.values():
            GUallInfo.append(i[1])
            GUallInfo.append(i[2])
            GUallInfo.append(i[3])
        statereset.extend(lsuPos)
        statereset.extend(ljuPos)
        statereset.extend(meupos)
        statereset.extend(mjupos)
        statereset.extend(GUallInfo)
        statereset.extend(LSUPosHistory)
        statereset.extend(LSUTxHistory)

        # statereset.extend(LJUPosHistory)
        statereset.extend(odhistoryFull)
        # statereset.extend(gutxhistory)
        self.state1obs = np.array(statereset, dtype=np.float32)
        return self.state1obs

    def step1(self, action, copy_guinfo, LSUPosHistory, LSUTxHistory,odhistoryFull ):

        def shiftToactualAction(sampled_action, action_low, action_high):
            scaled_action = (sampled_action + 1) / 2 * (action_high - action_low) + action_low
            return scaled_action

        action1 = np.array(action)
        action1[action1 < -1] = -1
        action1[action1 > 1] = 1
        lsu_x = shiftToactualAction(action1[0], xMinEnv, xMaxEnv)
        lsu_y = shiftToactualAction(action1[1], yMinEnv, yMaxEnv)
        lsu_h = shiftToactualAction(action1[2], zMinEnv, zMaxEnv)
        lsu_tx = int(round(shiftToactualAction(action1[3], TxMin, lsuTxMax)))
        # lju_x = shiftToactualAction(action1[4], xMinEnv, xMaxEnv)
        # lju_y = shiftToactualAction(action1[5], yMinEnv, yMaxEnv)
        # lju_h = shiftToactualAction(action1[6], zMinEnv, zMaxEnv)
        # lju_tx = int(round(shiftToactualAction(action1[7], TxMin, ljuTxMax)))

        lsunewstate = [lsu_x, lsu_y, lsu_h]
        # ljunewstate = [lju_x, lju_y, lju_h, lju_tx]
        # gunewtx = []
        # for j in range(8, NumGUs+8):
        #     gunewtx.append(int(round(shiftToactualAction(action1[j], guTxMin, guTxMax))))
        secrecyrate_info = []
        r_g_s_info =[]
        r_g_e_info = []
        reward = 0
        actualGU = cal.calActualComUAVtoGU(lsu_tx, gunewtx, [lsunewstate[0], lsunewstate[1], lsunewstate[2]])
        for key in actualGU.keys():
            secrate,r_g_s, r_g_e = cal.secrecyRate([actualGU[key][0], actualGU[key][1]],gunewtx, lsunewstate, ljuInitPos, meuPos, mjuPos,lju_tx,mjuTx )
            secrecyrate_info.append(secrate)
            r_g_s_info.append(r_g_s)
            r_g_e_info.append(r_g_e)
            reward += secrate*copy_guinfo[key][1]  # offloading demand
            if secrate!= 0:
               copy_guinfo[key][1] = 0

        GUallInfo = []
        for i in copy_guinfo.values():
            GUallInfo.append(i[1])
            GUallInfo.append(i[2])
            GUallInfo.append(i[3])

        LSUPosHistory = LSUPosHistory[3:]
        LSUPosHistory.extend(lsunewstate)
        LSUTxHistory = LSUTxHistory[1:]
        LSUTxHistory.append(lsu_tx)

        # LJUPosHistory = LJUPosHistory[4:]
        # LJUPosHistory.extend(ljunewstate)
        odhistoryFull = odhistoryFull[NumGUs:]

        odhistory = [copy_guinfo[i][1] for i in range(1, NumGUs + 1)]
        odhistoryFull.extend(odhistory)
        # gutxhistory = gutxhistory[NumGUs:]
        # gutxhistory.extend(gunewtx)
        stateupdate = []
        stateupdate.extend(lsunewstate)
        stateupdate.extend(ljuInitPos)
        stateupdate.extend(meuPos)
        stateupdate.extend(mjuPos)
        stateupdate.extend(GUallInfo)
        stateupdate.extend(LSUPosHistory)
        stateupdate.extend(LSUTxHistory)

        # stateupdate.extend(LJUPosHistory)
        stateupdate.extend(odhistoryFull)
        # stateupdate.extend(gutxhistory)
        state1obs = np.array(stateupdate, dtype=np.float32)

        terminated = False
        return state1obs, reward, terminated, False, {},copy_guinfo, LSUPosHistory,LSUTxHistory,odhistoryFull, secrecyrate_info, r_g_s_info, r_g_e_info


################################### Training ###################################
def train():
    print("============================================================================================")
    env_name = 'my custom environment'
    has_continuous_action_space = True
    max_ep_len = 15  # max timesteps in one episode
    action_std = 0.6

    max_training_timesteps = int(1500000)  # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 10  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.05 # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(50000)  # action_std decay frequency (in num timesteps)
    #####################################################
    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update
    eps_clip = 0.1  # clip parameter for PPO
    gamma = 0.9  # discount factor
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network
    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)
    env = CustomEnv()
    # state space dimension
    state_dim = env.observation1_space.shape[0]
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action1_space.shape[0]
    else:
        action_dim = env.action1_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:
        # prevTime = dict()
        # offTime = dict()
        # FL_val = dict()
        # for i in range(1, NumGUs + 1):
        #     prevTime[i] = 0
        #     offTime[i] = 0
        #     FL_val[i] = 0
        guinfo_copy = copy.deepcopy(guinfo)
        LSUPosHistory = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        LSUTxHistory = [0,0,0,0,0,0,0,0]

        # LJUPosHistory = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        odhistory = [guinfo_copy[i][1] for i in range(1, NumGUs+1)]
        odhistoryFull=[]
        for _ in range(8):
            odhistoryFull.extend(odhistory)
        # gutxHistory = [0 for _ in range(NumGUs * 8)]
        state = env.reset1obs(guinfo_copy, LSUPosHistory,LSUTxHistory, odhistoryFull)
        current_ep_reward = 0
        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _, _, guinfo_copy, LSUPosHistory, LSUTxHistory, odhistoryFull, secrecyrate_info, r_g_s_info, r_g_e_info = env.step1(action,guinfo_copy, LSUPosHistory,LSUTxHistory,odhistoryFull  )
            # saving reward and is_terminals
            if i_episode == 80000:
                print(f'LSUPosHistory info : {LSUPosHistory}')
                print(f'secrecyrate_info info: {secrecyrate_info}')
                print(f'r_g_s_info info: {r_g_s_info}')
                print(f'r_g_e_info info: {r_g_e_info}')
                print(f'reward info: {reward}')

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()







