import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme()

"""
The density value used for each setting is: 
     - Fresh Water = 1000kg/m³ 
     - EN13319 = 1020 kg/m³ 
     - Salt Water = 1030 kg/m³
"""
rho_water = 1030.0
p_atm_sea_level = 101325.0
g = 9.81

def read_M_values_from_file(file_path):
    df_M_Values = pd.read_csv(file_path, sep = ";")
    M_values = {}
    for i in range(len(df_M_Values)):
        cpt = df_M_Values.iloc[i]['Cpt']
        HT = df_M_Values.iloc[i]['Half-time [sec]']
        M0 = df_M_Values.iloc[i]['M0']
        M_slope = df_M_Values.iloc[i]['M_slope']
        M_values[cpt] = {}
        M_values[cpt]["HT"] = HT
        M_values[cpt]["M0"] = M0
        M_values[cpt]["M_Slope"] = M_slope       
    return M_values



def is_below_cpt_mvalue_line(cpt, Pamb, P):
    # P : Inert gas tissue pressure
    # Pamb : Ambiant pressure
    return P < (M_values[cpt]['M0'] + M_values[cpt]['M_Slope'] * (Pamb - 1.))
    
    
def shreiner(k, R, P0, Pi0, t):
    # k : half-time constant = ln2/half-time [sec**-1]
    # R : rate of change in inspired gas pressure with change in ambient pressure [bar / sec]
    # Pi0 : initial inspired (alveolar) inert gas pressure [bar]
    # P0 : initial compartment inert gas pressure [bar]
    P = Pi0 + R * (t - (1/k)) - (Pi0 - P0 - (R/k)) * np.exp(-k*t)
    return P
    
    
    
def define_dive_depth_profile(depth, time):
    time *= 60
    descent_time = depth / descent_speed
    total_diving_time = int(descent_time + time)   
    start = pd.to_datetime('00:00:00', format = "%H:%M:%S")
    times = [start + pd.Timedelta(seconds = i) for i in range(total_diving_time + 1)]
    phases = []
    depths = []
    for t in times:
        if ((t - times[0]).total_seconds() <= descent_time):
            depth = (t - times[0]).total_seconds() * descent_speed
            phase = "Descent"
        else:
            depth = depth
            phase = "Bottom"
        depths.append(depth)
        phases.append(phase)
        
    output = pd.DataFrame({'Temps [sec]' : times, 'Profondeur [m]' : depths, 'Phase' : phases})
    return output



def calculate_cpt_saturation_along_dive(diving_profile):
    nb_steps = len(diving_profile)
    time_steps = list(diving_profile["Temps [sec]"])
    depth_steps = list(diving_profile["Profondeur [m]"])

    for cpt in M_values.keys():
        #print(" > Cpt ° " + cpt)
        if (cpt == "6"):
            print("wazaa")
        HT = M_values[cpt]['HT']
        k = np.log(2) / HT # k = half-time constant = ln2/half-time
        tissue_time_sat = []
        for i in range(nb_steps):
            if (i == 0):
                tissue_time_sat.append(N2Part)
            else:
                P0 = tissue_time_sat[-1] # initial compartment inert gas pressure [bar]
                Pamb = (p_atm_sea_level + (depth_steps[i] * rho_water * g)) / 1E5
                Pi0 = N2Part * Pamb # initial inspired (alveolar) inert gas pressure [bar]
                depth_delta = depth_steps[i] - depth_steps[i-1]
                time_delta = (time_steps[i] - time_steps[i-1]).total_seconds()
                desc_speed = depth_delta / time_delta
                R = N2Part * desc_speed * ((rho_water * g) / 1E5) # R = rate of change in inspired gas pressure with change in ambient pressure
                t = time_delta # time
                tissue_time_sat.append(shreiner(k, R, P0, Pi0, t))
            
        diving_profile['pN2_cpt_' + cpt + ' [bar]'] = tissue_time_sat
    return diving_profile
    
    
    
def get_ascent_duration(current_depth, final_depth, ascent_speed):
    return (final_depth - current_depth) / -ascent_speed




def get_next_deco_stop(N2Part, ascent_speed, curr_depth, pN2_cpts, idx_deco_stop = 0):      
    is_above_mvalue_line = True  
    while (is_above_mvalue_line):
        final_depth_candidate = possible_deco_stops[idx_deco_stop]
        t = get_ascent_duration(curr_depth, final_depth_candidate, ascent_speed)
        for cpt in M_values.keys():
            k = np.log(2) / M_values[cpt]['HT']
            R = - N2Part * ascent_speed * ((rho_water * g) / 1E5)
            Pi0 = N2Part * (p_atm_sea_level + (curr_depth * rho_water * g)) / 1E5
            P0 = pN2_cpts["pN2_cpt_" + cpt + " [bar]"]
            P = shreiner(k, R, P0, Pi0, t)
            Pamb = (p_atm_sea_level + (final_depth_candidate * rho_water * g)) / 1E5                
            if (is_below_cpt_mvalue_line(cpt, Pamb, P)):
                is_above_mvalue_line = False            
            else:
                is_above_mvalue_line = True
                break
        if (not is_above_mvalue_line):
            return final_depth_candidate
        else:
            idx_deco_stop += 1
            
            
            
def calculate_time_at_next_deco_stop(N2Part, ascent_speed, curr_depth, deco_depth, pN2_cpts):
    ascent_time = get_ascent_duration(curr_depth, deco_depth, ascent_speed)
    pN2_cpts_after_ascent = {}
    for cpt in M_values.keys():
        k = np.log(2) / M_values[cpt]['HT']
        R = - N2Part * ascent_speed * ((rho_water * g) / 1E5)
        Pi0 = N2Part * (p_atm_sea_level + (curr_depth * rho_water * g)) / 1E5
        P0 = pN2_cpts["pN2_cpt_" + cpt + " [bar]"]
        P = shreiner(k, R, P0, Pi0, ascent_time)
        pN2_cpts_after_ascent["pN2_cpt_" + cpt + " [bar]"] = P
    stay_at_depth = True
    time_at_stop_min = 1
    while (stay_at_depth):
        pN2_cpts_candidates = {}
        for cpt in M_values.keys():
            k = np.log(2) / M_values[cpt]['HT']
            Pi0 = N2Part * (p_atm_sea_level + (deco_depth * rho_water * g)) / 1E5
            P0 = pN2_cpts_after_ascent["pN2_cpt_" + cpt + " [bar]"]
            P = shreiner(k, 0., P0, Pi0, time_at_stop_min * 60)
            pN2_cpts_candidates["pN2_cpt_" + cpt + " [bar]"] = P
        next_deco_depth = get_next_deco_stop(N2Part, ascent_speed, deco_depth, pN2_cpts_candidates, idx_deco_stop = 0)
        
        if (next_deco_depth > deco_depth):
            print("Problem : Diver above deco ceil !!!")
            
        elif (next_deco_depth < deco_depth):
            stay_at_depth = False
        
        elif (next_deco_depth == deco_depth):
            time_at_stop_min += 1    
    
    return (time_at_stop_min, pN2_cpts_candidates)



def calculate_deco_profile(diving_profile, ascent_speed, ascent_speed_between_stops):
    pN2_cpts = diving_profile[[c for c in diving_profile.columns if ("pN2_cpt_" in c)]].iloc[-1]
    curr_depth = diving_profile["Profondeur [m]"].iloc[-1]
    fin_depth = 0.
    deco_stops = []
    deco_stop = 1
    while (deco_stop > 0):
        if (len(deco_stops) == 0): ascent_speed = ascent_speed
        else: ascent_speed = ascent_speed_between_stops                  
        deco_stop = get_next_deco_stop(N2Part, ascent_speed, curr_depth, pN2_cpts)
        if (deco_stop > 0.):
            (deco_time, pN2_cpts) = calculate_time_at_next_deco_stop(N2Part, ascent_speed, curr_depth, deco_stop, pN2_cpts)
            deco_stops.append({'DECO_DEPTH' : deco_stop, 'DECO_TIME' : deco_time, 'SPEED_ASCENT' : ascent_speed})
            curr_depth = deco_stop
    return deco_stops



def add_deco_to_depth_profile(diving_profile, deco_profile):
    depths = diving_profile['Profondeur [m]'].tolist()
    times = diving_profile['Temps [sec]'].tolist()    
    depth = depths[-1]
    t = times[-1]  
    for deco_stop in deco_profile:
        speed_ascent = deco_stop['SPEED_ASCENT']
        deco_time = deco_stop['DECO_TIME']
        deco_depth = deco_stop['DECO_DEPTH']        
        ascent_seconds = int(np.ceil((depth - deco_depth) / speed_ascent))
        for i in range(1, ascent_seconds + 1):
            depth += - speed_ascent
            t += pd.Timedelta(seconds = 1)
            depths.append(depth)
            times.append(t)
        for j in range(1, deco_time * 60 + 1):
            t += pd.Timedelta(seconds = 1)
            depths.append(depth)
            times.append(t)
    output = pd.DataFrame()
    output['Profondeur [m]'] = depths
    output['Temps [sec]'] = times    
    return output      



def calculate_GF_lines(GF_low, GF_high, Pamb_max, M_values):
    custom_M_values = {}
    for cpt in M_values.keys():
        M0 = M_values[cpt]['M0']
        M_slope = M_values[cpt]['M_Slope']
        custom_M0 = (M0 - 1.) * GF_high + 1.    
        point_1 = np.array([Pamb_max, Pamb_max])
        point_2 = np.array([1. + (Pamb_max - M0)/(M_slope), Pamb_max])
        point_GF_low = point_1 * (1 - GF_low) + point_2 * GF_low
        custom_M_slope = (point_GF_low[1] - custom_M0) / (point_GF_low[0] - 1.)
        custom_M_values[cpt] = {}
        custom_M_values[cpt]['M0'] = custom_M0
        custom_M_values[cpt]['M_Slope'] = custom_M_slope  
        custom_M_values[cpt]['HT'] = M_values[cpt]['HT']
    return custom_M_values


def total_time_to_surface(depth, deco_profile):
    tts = 0
    for deco_stop in deco_profile:
        time_to_stop = (depth - deco_stop['DECO_DEPTH']) / (60 * deco_stop['SPEED_ASCENT'])
        time_at_stop = deco_stop['DECO_TIME']
        tts += time_to_stop + time_at_stop
        depth = deco_stop['DECO_DEPTH']
    tts += deco_profile[-1]['DECO_DEPTH'] / (60 * deco_profile[-1]['SPEED_ASCENT'])
    tts = int(np.ceil(tts))
    return tts

descent_speed = 20 / 60
ascent_speed = 10 / 60
N2Part = 0.79
M_values = read_M_values_from_file("Buhlmann_Zh-L16C_M-values.csv")
M_values_basis = M_values
diving_profile = define_dive_depth_profile(depth = 30, time = 59)
Pamb_max = 1 + (diving_profile['Profondeur [m]'].max() / 10)
M_values = calculate_GF_lines(1., 1., Pamb_max, M_values)
diving_profile = calculate_cpt_saturation_along_dive(diving_profile)
possible_deco_stops = [*range(0, 1 + int(np.floor(diving_profile['Profondeur [m]'].max())), 3)]
deco_profile = calculate_deco_profile(diving_profile, ascent_speed = 10 / 60, ascent_speed_between_stops = 3 / 60)
deco_profile.append({'DECO_DEPTH': 0., 'DECO_TIME': 0, 'SPEED_ASCENT': 3 / 60})
depth_before_ascent = diving_profile['Profondeur [m]'].iloc[-1]
diving_profile = add_deco_to_depth_profile(diving_profile, deco_profile)
diving_profile = calculate_cpt_saturation_along_dive(diving_profile)

print(total_time_to_surface(depth_before_ascent, deco_profile))
print(deco_profile)
