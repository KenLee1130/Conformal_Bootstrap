import torch
import matplotlib.pyplot as plt

from IPython.display import display

from .Train_and_Predict import Train_and_Predict
from .SAC import save_sac_agent, load_sac_agent
import os
import re
import json
import time

def launch_agent(agent_name):
    agent_name = f"./Agents/{agent_name}"
    if not os.path.exists(agent_name):
        os.makedirs(agent_name)
        os.makedirs(agent_name+'/model')
        os.makedirs(agent_name+'/checkpoints')
        os.makedirs(agent_name+'/traj')

        data=[{'Status':'Start'}]
        with open(os.path.join(agent_name, 'log.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"establish folder: {agent_name}")
    else:
        print(f"Folder {agent_name} has been established")


def complete_theory(raw_theory, info):
    if type(raw_theory)==list:
        result=[]
        for theory in raw_theory:
            theory.update(info)
            result.append(theory)
        return result
    raw_theory.update(info)
    return raw_theory


def written_in_log(agent_name, msg):
    with open(f"{agent_name}/log.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    data.append(msg)

    with open(f"{agent_name}/log.json", 'w', encoding='utf-8') as f:
        formatted_log = json.dumps(data, ensure_ascii=False, indent=4)
        f.write(formatted_log + "\n")


def get_log_info(agent_name):
    with open(f"{agent_name}/log.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('================================')
    for i in range(1, len(data)):
        print(data[i]["Status"])
        for params in data[i]["Params"]:
            print(params["model name"])
        print('================================')
    

class TheoryData:
    def __init__(self, theories=None):
        """
        Initialize a TheoryData object.

        :param theories: (optional) A list of theory dictionaries.
                         Each theory dictionary should contain at least the keys "theory" and "scope".
                         If not provided, it initializes as an empty list.
        """
        self.theories = theories if theories is not None else []
        self.model_names = [self._build_model_name(theory) for theory in self.theories]

    def _build_model_name(self, theory):
        """
        Generate a model name based on the theory.
        The default format is "<theory>_<scope>".

        :param theory: dict, which should contain at least the keys "theory" and "scope"
        :return: str, the model name
        """
        # The format can be adjusted as needed.
        return f"{theory.get('theory', 'unknown')}_{theory.get('scope', 'unknown')}"

    def add_theory(self, theory):
        """
        Add a theory entry or multiple theory entries and automatically update model_names.

        :param theory: either a dict (which should contain at least the keys "theory" and "scope")
                    or a list of such dicts.
        """
        if isinstance(theory, list):
            for t in theory:
                self.theories.append(t)
                self.model_names.append(self._build_model_name(t))
        elif isinstance(theory, dict):
            self.theories.append(theory)
            self.model_names.append(self._build_model_name(theory))
        else:
            raise ValueError("Input must be a dictionary or a list of dictionaries.")

    def update_theory(self, index, new_theory):
        """
        Update the theory at the specified index and update the corresponding model name.

        :param index: int, the index of the theory to update
        :param new_theory: dict, the new theory data
        """
        if 0 <= index < len(self.theories):
            self.theories[index] = new_theory
            self.model_names[index] = self._build_model_name(new_theory)
        else:
            raise IndexError("Index out of range for theories list.")

    def set_theories(self, new_theories):
        """
        Rebuild the theories list and automatically update model_names.

        :param new_theories: list of dict, each dict should contain at least the keys "theory" and "scope"
        """
        self.theories = new_theories
        self.model_names = [self._build_model_name(theory) for theory in new_theories]

    def get_model_names(self):
        """
        Return the list of model names.
        """
        return self.model_names
    

class tools:
    def __init__(self, agent_name):
        launch_agent(agent_name)
        self.agent_name = f"./Agents/{agent_name}"
        self.all_params = None


    def generate_tasks(self, data, device=torch.device("cpu"), max_steps=0, start_steps=-1):
        for theory in data.theories:
            assert "theory" in theory, "theory key must be provided"
            assert "scope" in theory, "scope key must be provided"
            assert "spins" in theory, "spins key must be provided"
            assert "dSigma" in theory, "dSigma key must be provided"
            assert "d_max" in theory, "d_max key must be provided"
            assert "init_state" in theory, "init_state key must be provided"
            assert "bound" in theory, "bound key must be provided"
        theory_name=theory["theory"]
        params = [
            {
            'theory':theory["theory"],
            'scope':theory["scope"],
            'spins':theory["spins"],
            'dSigma':theory["dSigma"],
            'd_max':theory["d_max"],
            'init_state':theory["init_state"],
            'bound':theory["bound"],
            'state_dim':theory.get("state_dim", None),
            'action_dim':theory.get("action_dim", None),
            'max_steps': max_steps if max_steps > 0 else theory.get("max_steps", int(1e6)),
            'max_episode_len':theory.get("max_episode_len", 50),
            'device':theory["device"],
            'start_steps':start_steps if start_steps>0 else theory.get("start_steps", int(1e4)),
            'd_step_size':theory.get("d_step_size", 0.1),
            'reward_scale':theory.get("reward_scale", 10.0),
            'num_envs':theory.get("num_envs", 1),
            'updates_per_step':theory.get("updates_per_step", 1),
            'std_z':theory.get("std_z", 0.1),
            'num_points':theory.get("num_points", 400),
            'w_name':theory.get("w_name", ""),
            'gamma':theory.get("gamma",0.99),
            'tau':theory.get("tau",0.005),
            'batch_size':theory.get("batch_size",256),
            'hidden_size':theory.get("hidden_size",256),
            'replay_capacity':theory.get("replay_capacity",1_000_000),
            'alpha_init':theory.get("alpha_init",0.1),
            'agent_type':theory.get("agent_type", "SAC"),
            'seed':theory.get("seed", 1217),
            'n_states_rew':theory.get("n_states_rew", 2),
            'lr':theory.get("lr", 3e-4),
            'lr_final':theory.get("lr1", 3e-4),
            'checkpoint_dir':theory.get("checkpoint_dir", f"{self.agent_name}/checkpoints"),
            'checkpoint_prefix':f"{self.agent_name[9:]}_{theory['theory']}",
            'checkpoint_interval':theory.get("checkpoint_interval", 1e5)
            } 
            for theory in data.theories]
        
        self.all_params=params
        return [Train_and_Predict(**p) for p in params]


    def train(self, tasks):
        start_time=time.time()
        agent_name = self.agent_name[9:]
        for task in tasks:
            task.train()
        end_time=time.time()
        written_in_log(self.agent_name, msg={'Status':'Train', 'agent name':agent_name, 'Run time':f'{(end_time-start_time)/60} mins', 'Params':self.all_params})
        print("Training status has been written in /.log")
        return tasks


    def save(self, tasks, agent_name):
        model_name_list = []
        for task in tasks:
            agent = task.agent
            save_sac_agent(
                agent, 
                f'{self.agent_name}/model/'+agent_name
            )
            model_name_list.append(agent_name)

        written_in_log(self.agent_name, msg={'Status':'Save', 'agent name':agent_name})
        print("Saving status has been written in /.log")


    def train_and_save(self, tasks, agent_name):
        start_time=time.time()
        # Although it's a for loop but PLEASE only run ONE theory!
        agent_name_list = []
        for task in tasks:
            # Train
            task.train()

            # Save
            agent = task.agent
            save_sac_agent(
                agent, 
                f'{self.agent_name}/model/'+agent_name
            )
            agent_name_list.append(agent_name)
        end_time=time.time()
        written_in_log(self.agent_name, msg={'Status':'Train', 'agent name':agent_name_list, 'Run time':f'{(end_time-start_time)/60} mins', 'Params':self.all_params})
        print("Training and Saving status has been written in /log")
        return tasks


    def reproduce(
        self, 
        tasks, 
        load = load_sac_agent, 
        num_episodes = 100,
        agent_name = "",
    ):
        '''
        One agent try to predict different tasks.
        '''
        start_time=time.time()
        agent_name_list=[]
        if type(agent_name)==list:
            agent_name, file = agent_name
            step=file.split('_')[3][:-4]
            model_saved_path = f'{self.agent_name}/checkpoints/'+file
            agent_name=agent_name+f"_check_{step}"
        else:
            model_saved_path = f'{self.agent_name}/model/'+agent_name
        
        for task in tasks:
            load(task.agent, model_saved_path)
            agent_name = self.predict_trajectories(task, num_episodes=num_episodes, agent_name=agent_name, model_name=task.theory)
            agent_name_list.append(agent_name)
        end_time=time.time()
        #print({'Status':'Test', 'agent names': agent_name_list})
        written_in_log(self.agent_name, msg={'Status':'Test', 'agent names': agent_name_list, 'Run time':f'{(end_time-start_time)/60} mins', 'Params':self.all_params})


    def transfer_learning(self, 
        tasks,
        agent_name,
        load_or_not = False, 
        load = load_sac_agent, 
    ):
        start_time=time.time()

        if load_or_not:
            for task in tasks:
                load(task.agent, f'{self.agent_name}/model/'+agent_name)

        for task in tasks:
            task.train()
        
        end_time=time.time()
        written_in_log(self.agent_name, msg={'Status':'Retrain', 'agent names':agent_name, 'Run time':f'{(end_time-start_time)/60} mins', 'Params':self.all_params})
        print("Retraining and Saving status has been written in /log")
        return tasks


    def transfer_prediction_with_training(
        self, 
        tasks,
        agent_name,
        new_agent_name = None, 
        load = load_sac_agent, 
    ):
        start_time=time.time()
        for task in tasks:
            load(task.agent, f'{self.agent_name}/model/'+agent_name)

        agent_name_list=[]
        for idx in range(len(tasks)):
            task = tasks[idx]
            task.train()
        
            # Save
            agent = task.agent

            if new_agent_name==None:
                new_agent_name = agent_name[9:]+f"_{idx}"
            
            save_sac_agent(
                agent, 
                f'{self.agent_name}/model/'+new_agent_name
            )
            agent_name_list.append(new_agent_name)
        end_time=time.time()
        written_in_log(self.agent_name, msg={'Status':'Retrain', 'agent names':agent_name_list, 'Run time':f'{(end_time-start_time)/60} mins', 'Params':self.all_params})
        print("Retraining and Saving status has been written in /log")
        return tasks


    def predict_trajectories(self, task, num_episodes=1, agent_name="", model_name=""):
        save_trajectories_dir=self.agent_name+'/traj'
        
        if os.path.isfile(f'{save_trajectories_dir}/{agent_name}_delta_trajectories_of_{model_name}.pt'):
            pattern = re.compile(rf"^{re.escape(agent_name)}_delta_trajectories_of_{re.escape(model_name)}(\d*)\.pt$")
            existing_files = [f for f in os.listdir(save_trajectories_dir) if pattern.match(f)]
            numbers = []
            for f in existing_files:
                m = pattern.match(f)
                if m:
                    suffix = m.group(1)
                    if suffix == "":
                        numbers.append(0)
                    else:
                        try:
                            numbers.append(int(suffix))
                        except ValueError:
                            pass
            
            new_number = max(numbers) + 1 if numbers else 1
            model_name = model_name + f"{new_number}"
        task.predict_and_save_traj(num_episodes=num_episodes, save_trajectories_dir=save_trajectories_dir, agent_name=agent_name, model_name=model_name)
        return agent_name


    def load_trajectories_from_file(self, agent_name, model_name):
        """
        Loads trajectories, c_means, rewards, and c_stds saved using torch.save.

        Args:
            save_trajectories_dir (str): The directory where the .pt files are saved.
            model_name (str): The name of the model (expects format like 'someprefix_actualname').
                            The first 8 characters will be stripped for the filename.
            theory (str): The theory identifier used in the filename.
            scope (str): The scope identifier used in the filename.

        Returns:
            tuple: A tuple containing (trajectories, c_means, rews, c_stds).
                Returns None for an item if its corresponding file cannot be loaded.
                Returns (None, None, None, None) if the directory doesn't exist.
        """
        save_trajectories_dir=self.agent_name+'/traj'

        # --- Construct file paths ---
        # Note the addition of the '.pt' extension, crucial for matching saved files
        traj_path = os.path.join(save_trajectories_dir, f'{agent_name}_delta_trajectories_of_{model_name}.pt')
        c_mean_path = os.path.join(save_trajectories_dir, f'{agent_name}_c_mean_trajectories_of_{model_name}.pt')
        rew_path = os.path.join(save_trajectories_dir, f'{agent_name}_rew_trajectories_of_{model_name}.pt')
        c_std_path = os.path.join(save_trajectories_dir, f'{agent_name}_c_std_trajectories_of_{model_name}.pt')

        # --- Load data using torch.load ---
        loaded_data = {}
        paths_to_load = {
            'trajectories': traj_path,
            'c_means': c_mean_path,
            'rews': rew_path,
            'c_stds': c_std_path
        }

        for key, path in paths_to_load.items():
            if os.path.exists(path):
                try:
                    # Use map_location='cpu' to avoid device issues if saved on GPU and loaded on CPU
                    loaded_data[key] = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
                except Exception as e:
                    loaded_data[key] = None # Assign None if loading fails
            else:
                loaded_data[key] = None # Assign None if file doesn't exist

        # Return the loaded data in the specific order
        # Using .get(key) provides None if the key wasn't successfully added (file not found or load error)
        return (
            loaded_data.get('trajectories'),
            loaded_data.get('c_means'),
            loaded_data.get('rews'),
            loaded_data.get('c_stds')
        )
    

    def predict_from_checkpt(self, data, num_episodes, agent_name, model_name):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        folder_path = f"./Agents/{agent_name}/checkpoints"
        pattern = re.compile(rf"^{re.escape(agent_name)}_{re.escape(model_name)}.*\.pth$")
        matching_files = [f for f in os.listdir(folder_path) if pattern.match(f)]
        task = self.generate_tasks(data, device)
        for file in matching_files:
            self.reproduce(tasks=task, num_episodes=num_episodes, agent_name=[agent_name, file])
    

    def plot_checkpt_preform(self, data, agent_name, model_name):
        import numpy as np
        import matplotlib.gridspec as gridspec
        def average_over_steps(input_data):
            final_states = [episode[-1] for episode in input_data]
            final_states_np = np.array(final_states)
            mean_values = np.mean(final_states_np, axis=0)
            std_values = np.std(final_states_np, axis=0)
            
            return mean_values, std_values
        folder_path = f"./Agents/{agent_name}/traj"
        pattern = re.compile(rf"(?=.*{re.escape(agent_name)}_check)(?=.*{re.escape(model_name)}).*\.pt$")
        matching_files = [f for f in os.listdir(folder_path) if pattern.search(f)]
        #print(matching_files)
        d_mean_pt=[]
        d_std_pt=[]
        num_states=0
        num_fix=0
        step_seg=[]
        for file in matching_files:
            step_idx = file.split("_")[3]
            #print(step_idx)
            traj, c_means, rews, c_stds = self.load_trajectories_from_file(agent_name=agent_name+f"_check_{step_idx}", model_name=model_name)
            d_mean, d_std = average_over_steps(traj)
            num_states=data["action_dim"]#len(d_mean)-2
            num_fix=d_mean-num_states
            d_mean_pt.append(d_mean)
            d_std_pt.append(d_std)
            step_seg.append(int(step_idx))
        
        d_mean_pt = np.array(d_mean_pt).T
        d_std_pt = np.array(d_std_pt).T
        step_seg = np.array(step_seg)

        sorted_indices = np.argsort(step_seg)          # 依照 step_seg 值由小到大排序
        step_seg = step_seg[sorted_indices]     # 重新排序後的 x 軸
        d_mean_pt = d_mean_pt[:, sorted_indices]
        d_std_pt = d_std_pt[:, sorted_indices]
        #print(step_seg)
        fig, axs = plt.subplots(num_states, 1, figsize=(12, 3 * num_states))
        fig.suptitle(f"Check point of {agent_name} on {model_name}")

        #Left side graph
        for i, ax in enumerate(axs):
            ax.plot(step_seg, d_mean_pt[i+num_fix], color='black')
            ax.plot(step_seg, [data["init_state"][i+num_fix] for _ in range(len(step_seg))], color='blue')
            ax.fill_between(step_seg, np.array(d_mean_pt[i+num_fix]) - np.array(d_std_pt[i+num_fix]), np.array(d_mean_pt[i+num_fix]) + np.array(d_std_pt[i+num_fix]), alpha=0.2, color='blue')
            
            ax.set_xlabel("Check points")
            ax.set_ylabel(f"Stat. End point of Δ{i+num_fix}")
            #ax.set_xticks(step_seg)
            ax.grid(axis='x', linestyle='--', color='gray', alpha=0.7)
        plt.tight_layout()
        plt.show()

    
    def plot_traj(self, agent_name, theory):
        import numpy as np
        def calc_D_relative_error(delta_T, delta_RL):
            return np.abs(100*(delta_T-delta_RL)/delta_T)#np.sqrt(((delta_T-delta_RL)/delta_T)**2)
        
        import matplotlib.gridspec as gridspec
        model_name=theory["theory"]
        result = self.load_trajectories_from_file(agent_name, model_name)
        if result is None:
            return print(f"You have to run {model_name} on reproduce function first.")
        else:
            traj, c_means, rews, c_stds = result
        episodes=len(traj)
        num_fix=len(traj[0][0])-theory["action_dim"]
        num_states=len(traj[0][0])-num_fix # Fixed two states
        # True spectrum
        true_deltas=np.array(theory["init_state"])
        true_cs=theory["cs"]

        # Data preprocess
        episode_len=[]
        rewards=[]
        Dist_Drl_DT=[[] for _ in range(num_states)]
        traj_c_mean=[[] for _ in range(num_states)]
        traj_c_std=[[] for _ in range(num_states)]
        num_steps=0
        for ep in range(episodes):
            ep_len=0
            for step in range(len(traj[ep])):
                ep_len+=1
                num_steps+=1
                rewards.append(rews[ep][step])
                for state in range(num_states):
                    ed_deltas = calc_D_relative_error(true_deltas[state], traj[ep][step][state])
                    Dist_Drl_DT[state].append(ed_deltas)
                    traj_c_mean[state].append(c_means[ep][step][state])
                    traj_c_std[state].append(c_stds[ep][step][state])
            episode_len.append(num_steps)

        step_axis = [i for i in range(num_steps)]

        plt.figure(figsize=(18, 9))
        plt.plot(step_axis, rewards, color='blue')
        plt.title("Reward-steps")
        plt.xlabel("step")
        plt.ylabel("Reward")
        plt.xticks(episode_len)
        plt.grid(axis='x', linestyle='--', color='gray', alpha=0.7)
        plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

        fig = plt.figure(figsize=(18, 15))
        outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

        # Left side layout
        n_states_rew=int(self.all_params[0]["n_states_rew"])
        left_gs = outer[0].subgridspec(n_states_rew, 1, hspace=0.3)
        ax_left = [fig.add_subplot(left_gs[i]) for i in range(n_states_rew)]#fig.add_subplot(left_gs[1])
        
        # Right side layout
        right_gs = outer[1].subgridspec(num_states, 1, hspace=0.4)
        ax_right = [fig.add_subplot(right_gs[i]) for i in range(num_states)]

        
        # Left side graph
        for i, ax in enumerate(ax_left):
            ax.plot(step_axis, Dist_Drl_DT[i], label=f"Δ {i}")
            ax.legend()
            ax.set_title("Euclidean distance of (Δrl, Δtrue)-steps")
            ax.set_xlabel("step")
            ax.set_ylabel("Distance")
            ax.set_xticks(episode_len)
            ax.grid(axis='x', linestyle='--', color='gray', alpha=0.7)

        #Right side graph
        for i, ax in enumerate(ax_right):
            ax.plot(step_axis, [true_cs[i] for _ in step_axis], color='black')
            ax.plot(step_axis, traj_c_mean[i], color='blue')
            ax.fill_between(step_axis, np.array(traj_c_mean[i]) - np.array(traj_c_std[i]), np.array(traj_c_mean[i]) + np.array(traj_c_std[i]), alpha=0.2, color='blue')
            ax.set_title(f"C prediction-steps")
            ax.set_xlabel("steps")
            ax.set_ylabel("C prediction")
            ax.set_xticks(episode_len)
            ax.grid(axis='x', linestyle='--', color='gray', alpha=0.7)

    
    def predict_from_diff_model(self, task, num_episodes, agent_ver_list):
        agent_name_list = [f'Agent_r{i}' for i in agent_ver_list]
        for agent_name in agent_name_list:
            self.reproduce(tasks=task, num_episodes=num_episodes, agent_name=agent_name)
        
    
    def model_evol(self, high_trained_tasks, high_untrained_tasks, low_tasks, agent_ver):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        import seaborn as sns
        import pandas as pd

        def linear_reg(all_ep_rewards):
            episode_slopes = []
            episode_intercepts = []
            for rewards in all_ep_rewards:
                x = np.arange(len(rewards)).reshape(-1, 1)
                y = np.array(rewards)
                model = LinearRegression()
                model.fit(x, y)
                episode_slopes.append(model.coef_[0])
                episode_intercepts.append(model.intercept_)
            return episode_slopes
        
        def get_data(tasks):
            df_list=[]
            for task_idx in range(len(tasks)):
                model_name = tasks[task_idx].theory
                traj, c_means, rews, c_stds = self.load_trajectories_from_file(agent_name=f'Agent_r{agent_ver}', model_name=model_name)
                data = linear_reg(rews)
                df_part = pd.DataFrame({'theory':[model_name]*len(data), 'value':data})
                df_list.append(df_part)
            return pd.concat(df_list, ignore_index=True)
        
        df_ht = get_data(high_trained_tasks)
        df_hut = get_data(high_untrained_tasks)
        df_l = get_data(low_tasks)

        df_list = [df_ht, df_hut, df_l]

        fig, axs = plt.subplots(len(df_list), 1, figsize=(15, 8), sharex=True)
        # 如果 bins、範圍需要統一，可以先算 min、max 或自己定義
        x_min, x_max = -3, 5
        for i in range(len(df_list)):
            axs[i].axvline(x=0, color='black', linewidth=2, linestyle='--')
            sns.histplot(
                data=df_list[i],
                x="value", hue="theory",  # 以 'Theory' 做分類
                binrange=(x_min, x_max),
                bins=20,
                multiple="dodge",
                shrink=0.8,
                edgecolor='black',
                ax=axs[i]
            )
            axs[i].set_xlim(x_min, x_max)
            axs[i].set_xlabel("Slope of reward evolve in each episode")
        plt.tight_layout()
        plt.show()


    def delta_evol(self, high_trained_tasks, high_untrained_tasks, low_tasks, agent_ver):
        import numpy as np
        import matplotlib.pyplot as plt
        num_states = len(high_trained_tasks[0].init_state)

        def rela_error(deltaRL, deltaT):
            return np.abs(100*(deltaT-deltaRL)/deltaT)

        def get_data(tasks):
            num_states = len(tasks[0].init_state)
            model_name_list=[]
            delta_cap_list=[[[[] for state in range(num_states)] for _ in range(len(tasks))] for ver in agent_ver]
            delta_mean_list=[[[[] for state in range(num_states)] for _ in range(len(tasks))] for ver in agent_ver]
            delta_std_list=[[[[] for state in range(num_states)] for _ in range(len(tasks))] for ver in agent_ver]
            for task_idx in range(len(tasks)):
                delta_T = tasks[task_idx].init_state
                model_name = tasks[task_idx].theory
                model_name_list.append(model_name)
                for i, ver in enumerate(agent_ver):
                    traj_list=[[] for state in range(num_states)]
                    traj_cap=[[] for state in range(num_states)]
                    traj, c_means, rews, c_stds = self.load_trajectories_from_file(agent_name=f'Agent_r{ver}', model_name=model_name)
                    for ep in range(len(rews)):
                        for state in range(num_states):
                            delta_T_state = delta_T[state].cpu().detach().numpy()
                            traj_list[state].append(rela_error(traj[ep][-1][state], delta_T_state))
                            traj_cap[state].append(rela_error(delta_T_state+0.5, delta_T_state))
                    
                    delta_mean_list[i][task_idx] = np.mean(traj_list, axis=1)
                    delta_std_list[i][task_idx] = np.std(traj_list, axis=1)
                    delta_cap_list[i][task_idx] = [traj_cap[state][-1] for state in range(num_states)]

            return model_name_list, np.array(delta_mean_list), np.array(delta_std_list), np.array(delta_cap_list)
        
        ht_m_name, ht_d_mean, ht_d_std, ht_d_cap = get_data(high_trained_tasks)
        hut_m_name, hut_d_mean, hut_d_std, hut_d_cap = get_data(high_untrained_tasks)
        l_m_name, l_d_mean, l_d_std, l_d_cap = get_data(low_tasks)
        
        m_name = ht_m_name+hut_m_name+l_m_name
        d_mean = np.concatenate((ht_d_mean, hut_d_mean, l_d_mean), axis=1)#[[ht_d_mean[i][:][j] + hut_d_mean[i][:][j] + l_d_mean[i][:][j] for j in range(num_states)] for i in range(len(agent_ver))]
        d_std = np.concatenate((ht_d_std, hut_d_std, l_d_std), axis=1)#[[ht_d_std[i][:][j]  + hut_d_std[i][:][j]  + l_d_std[i][:][j] for j in range(num_states)]  for i in range(len(agent_ver))]
        d_cap = np.concatenate((ht_d_cap, hut_d_cap, l_d_cap), axis=1)

        ht_m_len=len(ht_m_name)
        hut_m_len=len(hut_m_name)
        l_m_len=len(l_m_name)
        num_tasks=ht_m_len+hut_m_len+l_m_len

        fig, axs = plt.subplots(len(agent_ver)*num_states, 1, figsize=(12, 25), sharex=True)
        x = np.arange(num_tasks)
        # 分段索引
        ht_idx = np.arange(0, ht_m_len)
        hut_idx = np.arange(ht_m_len, ht_m_len + hut_m_len)
        l_idx  = np.arange(ht_m_len + hut_m_len, num_tasks)
        # 指定每個區段的顏色與 label
        colors = ['red', 'orange', 'blue']
        seg_labels = ['High Trained', 'High Untrained', 'Low']

        graph = 0
        for ver in [0, 1]:
            for state in range(num_states):
                ax = axs[graph]
                # 對每個區段（ht、hut、l）畫圖
                for idx_seg, color, label in zip([ht_idx, hut_idx, l_idx],
                                                colors,
                                                seg_labels):
                    ax.errorbar(
                        x[idx_seg],
                        np.array(d_mean[ver])[idx_seg, state],
                        yerr=np.array(d_std[ver])[idx_seg, state],
                        fmt='o', capsize=5,
                        color=color,
                        label=label
                    )
                    ax.plot(x[idx_seg], np.array(d_cap[ver])[idx_seg, state], 'o-', color=color)
                    
                #ax.set_ylim(0, 10)
                ax.set_title(f"Agent {ver}, Δ{state}")
                ax.set_xlabel("Models")
                ax.set_ylabel("Relative Error of Δ (%)")
                ax.legend()
                ax.grid(True)
                graph += 1
            
        # 設定最後一個子圖的 x 軸標籤為模型名稱（自動依照 x 座標順序排序）
        axs[-1].set_xticks(x)
        axs[-1].set_xticklabels(m_name, rotation=45, fontsize=10)
        axs[-1].set_xlabel("Model Name")

        fig.suptitle(f"Δ Evolution", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return 
    

    def potential_plot(self, tasks, agent_ver, ii, jj):
        figs=[]
        for task in tasks:   
            data = self.load_trajectories_from_file(agent_name=f'Agent_r{agent_ver}', model_name=task.theory)
            fig=task.plot_trajectories(data, ii=ii, jj=jj)
            figs.append(fig)
        display(figs)


    def compare_T_RL(self, tasks, agent_ver):
        import numpy as np
        relative_rew={}
        for task in tasks:   
            delta_T = task.init_state
            traj, rews, c_means, c_stds = self.load_trajectories_from_file(agent_name=f'Agent_r{agent_ver}', model_name=task.theory)
            
            task_relative_rew=[]
            num_episodes = len(traj)
            delta_T = delta_T.unsqueeze(0)
            collect_rew_T=[]
            for _ in range(50):
                rews_T, c_mean_T, c_std_T = task.calc_rew_c(delta_T.to(device=torch.device("cuda:0")))
                rews_T = rews_T.cpu().detach().numpy()
                collect_rew_T.append(rews_T)
            print("True delta Reward: ", {task.theory:np.mean(collect_rew_T)})
            rews_T = np.mean(collect_rew_T)
            #rews_T=10
            for ep in range(num_episodes):
                delta_rl = torch.tensor(np.array([traj[ep][-1]]), dtype=torch.float64, device=torch.device("cuda:0"))
                rews_rl, c_mean_rl, c_std_rl = task.calc_rew_c(delta_rl)
                rews_rl=rews_rl.cpu().detach().numpy()
                task_relative_rew.append(rews_T/(rews_rl+1e-8))
            relative_rew.update({task.theory:np.mean(task_relative_rew)})
        print("relative Reward (rews_T/rews_rl): ", relative_rew)
        #return relative_rew


    def test_truncatibility(self, tasks, ii, jj, save=False, result_folder="./Truncatibility results/"):
        figs=[]
        for task in tasks:
            fig=task.plot_potential(ii, jj)
            if save==True:
                fig.savefig(result_folder+f"{task.theory}-process={task.dSigma}_{len(task.init_state)}states_std={task.std_z}_N={task.num_points}.png")
            figs.append(fig)
        display(figs)