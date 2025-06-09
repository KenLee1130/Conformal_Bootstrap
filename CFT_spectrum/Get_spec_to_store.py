# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 20:15:50 2025

@author: User
"""
import os
import json
import numpy as np
import ast

data_ver="_v3"

def mathematica_data_2_json(CFTdata):
    pq_val=CFTdata[0]
    central_charge=CFTdata[1]
    hexp=CFTdata[2]
    maxDelta=CFTdata[3]
    spec=np.array(ast.literal_eval(CFTdata[4]))
    
    delta_spin_pair = spec[:, [1, 2]]
    vacuum = np.all(delta_spin_pair == [0, 0], axis=1)
    stress_tensor = np.all(delta_spin_pair == [2, 2], axis=1)
    mask = ~(vacuum | stress_tensor)

    filtered_data = spec[mask]
    
    store_data={
        "theory": f"(p, q)={pq_val}",
        "central charge": central_charge, 
        "dSigma": 2*hexp,
        "d_max": maxDelta,
        "spec":[spec[stress_tensor][0].tolist()]+filtered_data.tolist()
        }
    return store_data

def store_data(new_data, label='low'):
    file_path = f"/{label}_minimal_model_data{data_ver}.json"
    # print(file_path)
    # 如果檔案存在且不為空，讀取原有資料；否則使用空列表
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []  # 若檔案內容格式錯誤，則重置為空列表
    else:
        data = []

    # 若原有資料不是列表，可以視需求調整，這裡我們假設以列表存放每筆資料
    if not isinstance(data, list):
        data = [data]

    # 將新的資料追加到列表中
    data.append(new_data)

    # 將更新後的資料寫回檔案
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
        
    return f"Successfully saved {new_data['theory']}-process: {new_data['dSigma']}!"

def load_minimal_pq(label='low'):
    import math
    def is_coprime(pq_val):
        """Return True if a and b are coprime, i.e., their GCD is 1."""
        a, b = pq_val
        return math.gcd(a, b) == 1
    
    with open(f"./CFT_spectrum/{label}_minimal_model_data{data_ver}.json", "r") as f:
        data_list = json.load(f)
    pq_val=[]
    process=[]
    for data in data_list:
        pq = ast.literal_eval(data["theory"][7:])
        one_process = data["dSigma"]
        if is_coprime(pq):
            pq_val.append(pq)
            process.append(one_process)

    pq_w_idx={}
    for idx in range(len(pq_val)):
        pq_w_idx[idx]=f"theory: {pq_val[idx]}-process: {process[idx]}"
    return pq_w_idx

def RL_input(json_type_data, num_states=None):
    spec = np.array(json_type_data["spec"])
    N = spec.shape[0]

    if num_states is None or num_states>N:
        num_states=N
    chosen_spec = spec[:num_states]
    deltas = chosen_spec[:, 1]
    spins = chosen_spec[:, 2]
    modified_c = np.array([0.5 if s == 0 else 1 for s in spins])
    cs = modified_c*chosen_spec[:, 0]

    store_data={
        "theory": f"(p, q)={json_type_data['theory'].split('=')[1]}",
        "central charge": json_type_data["central charge"],
        "spins": spins[:num_states].tolist(),
        "dSigma": json_type_data["dSigma"],
        "d_max": deltas.max(),
        "init_state":deltas.tolist(),
        "cs":cs.tolist(),
        "bound": [[ele-0.5, ele+0.5] for ele in spec[:num_states, 1]]
        }
    return store_data

def from_json_2_RL_input(theory_info, num_states, label='low'):
    import re
    import ast
    import json
    import numpy as np
    def match_pattern(s):
        pattern = r"theory:\s*(\[[^\]]*\])\s*-process:\s*([-+]?\d*\.?\d+)"
        match = re.search(pattern, s)
        if match:
            theory_str = match.group(1)      # '[4, 3]'
            process_str = match.group(2)     # '0.125'
            theory_data = ast.literal_eval(theory_str)
            process_value = float(process_str)
            return theory_data, process_value
    
    with open(f"./CFT_spectrum/{label}_minimal_model_data{data_ver}.json", "r") as f:
        data_list = json.load(f)
    
    pq_val, process = match_pattern(theory_info)
    for data in data_list:
        if data['theory'] == f"(p, q)={pq_val}" and data['dSigma'] == process:
            return RL_input(data, num_states)

def get_non_identity_global_p(spec, num_states):
    identity_delta = [2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0, 10.0, 10.0, 12.0, 12.0, 14.0, 16.0]
    identity_spin = [2.0, 4.0, 0.0, 6.0, 2.0, 8.0, 4.0, 0.0, 6.0, 2.0, 4.0, 0.0, 2.0, 0.0]
    identity_spec = set(zip(identity_delta, identity_spin))
    
    delta = spec[0]
    spin = spec[1]
    cs = spec[2]
    dSigma = spec[3]
    d_max = spec[4]

    selected_idx = [(d.item(), s.item()) not in identity_spec for d, s in zip(delta, spin)]
    return [delta[selected_idx][:num_states], spin[selected_idx][:num_states], dSigma, int(max(delta[selected_idx][:num_states]))], cs[selected_idx][:num_states]

def load_fusion_rule():
    dir_path = "./CFT_spectrum/minimal_model_data"
    with open(os.path.join(dir_path, "fusion_rule_data.json"), "r") as f:
        contents = json.load(f)
        fusion_dict={}
        idx=0
        for i in contents:
            fusion_dict[idx] = i
            idx+=1
        return fusion_dict

def process_data(num_states, process_info, output_num_states=None):
    dir_path = "./CFT_spectrum/minimal_model_data"
    file_names=os.listdir(dir_path)
    for file_name in file_names:
        Name = file_name.split(".")[0]
        if Name[-2:] == str(num_states):
            with open(os.path.join(dir_path, file_name), "r") as f:
                contents = f.read().split("*")
                for content in contents:
                    data = eval(content)
                    if process_info == [data[0], data[2]]:
                        return RL_input(mathematica_data_2_json(data), num_states=output_num_states)

def RL_input_Vversion(theory, json_type_data, num_states=None):
    spec = np.array(json_type_data["data"])
    N = spec.shape[0]

    if num_states is None or num_states>N:
        num_states=N
    chosen_spec = spec[:num_states]
    deltas = chosen_spec[:, 1]
    spins = chosen_spec[:, 2]
    modified_c = np.array([0.5 if s == 0 else 1 for s in spins])
    cs = modified_c*chosen_spec[:, 0]

    store_data={
        "theory": f"(p, q)={theory}",
        "central charge": json_type_data["central charge"],
        "spins": spins[:num_states].tolist(),
        "dSigma": 2*json_type_data["hext"],
        "d_max": deltas.max(),
        "init_state":deltas.tolist(),
        "cs":cs.tolist(),
        "bound": [[ele-0.5, ele+0.5] for ele in spec[:num_states, 1]]
        }
    return store_data

def process_data_sep(num_states, process_info, num_primaries=-1, combinatory=None, output_num_states=None):
    dir_path = "./CFT_spectrum/minimal_model_data"
    file_names=os.listdir(dir_path)
    test_pq_val, test_hext = process_info
    store_data={}
    for file_name in file_names:
        Name = file_name.split(".")[0]
        model_pq = Name.split("_")[1]
        try:
            pq_val = [int(model_pq[1]), int(model_pq[-2])]
        except:
            continue
        if Name[-6:] == str(num_states)+"_sep" and pq_val==test_pq_val:
            with open(os.path.join(dir_path, file_name), "r") as f:
                contents = json.load(f)
                for content in contents:
                    hext = content[0]["hext"]
                    if hext == test_hext:
                        combined_spec = []
                        for operator in content[-num_primaries:]:
                            spectra = RL_input_Vversion(pq_val, operator)
                            dtemp = spectra["init_state"]
                            stemp = spectra["spins"]
                            ctemp = spectra["cs"]
                            btemp = spectra["bound"]
                            for d, s, c, b in zip(dtemp, stemp, ctemp, btemp):
                                combined_spec.append((d, s, c, b))
                        combined_spec.sort(key=lambda t: t[0])
                        deltas_sorted, spins_sorted, cs_sorted, bounds_sorted = zip(*combined_spec)
    store_data["theory"]=spectra["theory"]
    store_data["central charge"]=spectra["central charge"]
    store_data["spins"]=list(spins_sorted[:output_num_states])
    store_data["dSigma"]=spectra["dSigma"]
    store_data["init_state"]=list(deltas_sorted[:output_num_states])
    store_data["d_max"]=max(deltas_sorted[:output_num_states])
    store_data["cs"]=list(cs_sorted[:output_num_states])
    store_data["bound"]=list(bounds_sorted[:output_num_states])
    return store_data

if __name__=='__main__':
    CFTdata = [[5, 4], 7/10, 3/2, 8, "[[1., 0., 0.], [6.428571428571428, 2., 2.], \
[3.642857142857142, 4., 4.], [41.32653061224489, 4., 0.], \
[1.7176870748299284, 6., 6.], [23.418367346938766, 6., 2.], \
[0.444305694305692, 8., 8.], [11.04227405247811, 8., 4.], \
[13.2704081632653, 8., 0.]]"]
        
    # label='low'
#     # print(load_minimal_pq(label))
    # print(mathematica_data_2_json(CFTdata))
#     # print(store_data(mathematica_data_2_json(CFTdata), label))
#     #print(load_minimal_pq(label=label))
#     #model_idx=10
    # total_states=8
    # data = from_json_2_RL_input(load_minimal_pq(label)[0], total_states, label=label)
    # print(data)
    # print(RL_input(mathematica_data_2_json(CFTdata), 8))

#     with open(f"./CFT_spectrum/all_model_specs.json", "r") as f:
#         data_list = json.load(f)

#     for data in data_list:
#         store_data(mathematica_data_2_json(eval(data)), label='all')
    # print(load_fusion_rule())
    # print(process_data(num_states=20, process_info=[[4, 3], 0.5]))
    print(process_data_sep(num_states=25, process_info=[[5, 4], 0.0375], num_primaries=-1, output_num_states=3))
    # import torch
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # num_states=20
    # picked_theory = process_data(num_states, [[4, 3], 0.0625], output_num_states=None)
    # delta = torch.tensor(picked_theory["init_state"], device=device)
    # spin = torch.tensor(picked_theory["spins"], device=device)
    # cs = np.array(picked_theory["cs"])
    # dSigma = picked_theory["dSigma"]
    # central_charge = picked_theory["central charge"]
    # d_max = picked_theory["d_max"]

    # spec = [delta, spin, cs, dSigma, d_max]
    # print(spec)
    # spec, cs = get_non_identity_global_p(spec, num_states)
    # print(spec)