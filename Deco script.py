#!/usr/bin/env python
# coding: utf-8

# # DECO SCRIPT :
# - Version : 01 / 04 / 2024 (1st April 2024)
# - Author : Benoit Pierson
# - Mail : benoitmarc.pierson@gmail.com

# ![screen_pic.jpg](attachment:screen_pic.jpg)

# # SCOPE : 
# 
# - The aim of this work is to give an overview of the methodology used by a diving computer, not to replace it or to plan a dive !
# 
# - This is a non-certifies work. This work has not been checked properly (no peer review process) as it has to be for a real usage
# 
# - This notebook is useful if you want to visualize tissues saturations along time for a given dive (see at the end of this notebook)

# # HYPOTHESIS : 
# - We consider Buhlmann ZHL16-C model for M-values and half-times of 16 tissues
# - We let the user choose the proper gradient factor he wants to use
# - For the first tissue, we consider 1b
# - Speed at descent is 20m/s
# - Speed ascent is 10m/s to the first deco stop, 3m/s between deco stops, 3m/s to the surface (high considering common used values ~ 1m/s in last meters of ascent

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme()


# # 1 - Definition of constants

# In[2]:


"""
The density value used for each setting is: 
     - Fresh Water = 1000kg/m³ 
     - EN13319 = 1020 kg/m³ 
     - Salt Water = 1030 kg/m³
"""

rho_water = 1030.0 # Water density [kg/m**3]
p_water_vapor = 0.0627 * 1E5 # Water vapor pressure in lungs [Pa]
p_atm_sea_level = 101325.0 # Ambiant pressure at sea level [Pa]
g = 9.81 # gravity [m*s**-2]


# # 2 - Definition of functions

# In[5]:


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
    return P < (M_values[cpt]['M0'] + M_values[cpt]['M_Slope'] * (Pamb - p_atm_sea_level/1E5))

    
    
def shreiner(k, R, P0, Pi0, t):
    # k : half-time constant = ln2/half-time [sec**-1]
    # R : rate of change in inspired gas pressure with change in ambient pressure [bar / sec]
    # Pi0 : initial inspired (alveolar) inert gas pressure [bar]
    # P0 : initial compartment inert gas pressure [bar]
    P = Pi0 + R * (t - (1/k)) - (Pi0 - P0 - (R/k)) * np.exp(-k*t)
    return P
    
    
    
def define_dive_depth_profile(depth, time):
    time *= 60
    descent_time = depth / des_speed
    total_diving_time = int(descent_time + time)   
    start = pd.to_datetime('00:00:00', format = "%H:%M:%S")
    times = [start + pd.Timedelta(seconds = i) for i in range(total_diving_time + 1)]
    phases = []
    depths = []
    for t in times:
        if ((t - times[0]).total_seconds() <= descent_time):
            depth = (t - times[0]).total_seconds() * des_speed
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
        HT = M_values[cpt]['HT']
        k = np.log(2) / HT # k = half-time constant = ln2/half-time
        tissue_time_sat = []
        for i in range(nb_steps):
            if (i == 0):
                tissue_time_sat.append(0.79 * ((p_atm_sea_level - p_water_vapor) / 1E5))
            else:
                P0 = tissue_time_sat[-1] # initial compartment inert gas pressure [bar]
                Pamb = p_atm_sea_level + depth_steps[i] * rho_water * g
                Pi0 = N2Part * ((Pamb - p_water_vapor) / 1E5) # initial inspired (alveolar) inert gas pressure [bar]
                depth_delta = depth_steps[i] - depth_steps[i-1]
                time_delta = (time_steps[i] - time_steps[i-1]).total_seconds()
                desc_speed = depth_delta / time_delta
                R = N2Part * desc_speed * ((rho_water * g) / 1E5) # R = rate of change in inspired gas pressure with change in ambient pressure
                t = time_delta # time
                tissue_time_sat.append(shreiner(k, R, P0, Pi0, t))
            
        diving_profile['pN2_cpt_' + cpt + ' [bar]'] = tissue_time_sat
    return diving_profile
    

    
def calculate_cpt_saturation_after_dive(diving_profile, time_after_dive_min):
    start = diving_profile['Temps [sec]'].tail(1).values[0]
    time_steps = [start + pd.Timedelta(seconds = i) for i in range(1, time_after_dive_min*60 + 1)]
    depth_steps = [0. for i in range(1, time_after_dive_min*60 + 1)]
    diving_profile_surf = pd.DataFrame()
    
    diving_profile_surf['Profondeur [m]'] = depth_steps
    diving_profile_surf['Temps [sec]'] = time_steps
    
    for cpt in M_values.keys():
        #print(" > Cpt ° " + cpt)
        HT = M_values[cpt]['HT']
        k = np.log(2) / HT # k = half-time constant = ln2/half-time
        tissue_time_sat = []
        for i in range(len(time_steps)):
            if (i == 0):
                P0 = diving_profile['pN2_cpt_' + cpt + ' [bar]'].tail(1).values[0] # initial compartment inert gas pressure [bar]
            else:
                P0 = tissue_time_sat[-1]
                
            Pamb = p_atm_sea_level + depth_steps[i] * rho_water * g
            Pi0 = 0.79 * ((Pamb - p_water_vapor) / 1E5) # initial inspired (alveolar) inert gas pressure [bar]
            depth_delta = depth_steps[i] - depth_steps[i-1]
            if (i == 0):
                time_delta = (time_steps[i] - start).total_seconds()
            else:
                time_delta = (time_steps[i] - time_steps[i-1]).total_seconds()
            R = 0.
            t = time_delta # time
            tissue_time_sat.append(shreiner(k, R, P0, Pi0, t))
           
        diving_profile_surf['pN2_cpt_' + cpt + ' [bar]'] = tissue_time_sat
    return diving_profile_surf
    
    
    
    
def get_ascent_duration(current_depth, final_depth, ascent_speed):
    return (final_depth - current_depth) / -ascent_speed




def get_next_deco_stop(N2Part, ascent_speed, curr_depth, pN2_cpts, idx_deco_stop = 0):      
    is_above_mvalue_line = True
    directing_cpt = None
    while (is_above_mvalue_line):
        final_depth_candidate = possible_deco_stops[idx_deco_stop]            
        t = get_ascent_duration(curr_depth, final_depth_candidate, ascent_speed)
        for cpt in M_values.keys():
            k = np.log(2) / M_values[cpt]['HT']
            R = - N2Part * ascent_speed * ((rho_water * g) / 1E5)
            Pamb = p_atm_sea_level + curr_depth * rho_water * g
            Pi0 = N2Part * ((Pamb - p_water_vapor) / 1E5)
            P0 = pN2_cpts["pN2_cpt_" + cpt + " [bar]"]
            P = shreiner(k, R, P0, Pi0, t)
            Pamb = (p_atm_sea_level + (final_depth_candidate * rho_water * g)) / 1E5                     
            if (is_below_cpt_mvalue_line(cpt, Pamb, P)):
                is_above_mvalue_line = False   
            else:
                is_above_mvalue_line = True
                directing_cpt = cpt
                break               
        if (not is_above_mvalue_line):
            return final_depth_candidate, directing_cpt       
        else:
            idx_deco_stop += 1
            
            
            
def calculate_time_at_next_deco_stop(N2Part, ascent_speed, curr_depth, deco_depth, pN2_cpts):
    ascent_time = get_ascent_duration(curr_depth, deco_depth, ascent_speed)
    pN2_cpts_after_ascent = {}
    for cpt in M_values.keys():
        k = np.log(2) / M_values[cpt]['HT']
        R = - N2Part * ascent_speed * ((rho_water * g) / 1E5)
        Pamb = p_atm_sea_level + curr_depth * rho_water * g
        Pi0 = N2Part * (Pamb - p_water_vapor) / 1E5
        P0 = pN2_cpts["pN2_cpt_" + cpt + " [bar]"]
        P = shreiner(k, R, P0, Pi0, ascent_time)
        pN2_cpts_after_ascent["pN2_cpt_" + cpt + " [bar]"] = P
        
    stay_at_depth = True
    time_at_stop_min = 1
    while (stay_at_depth):
        pN2_cpts_candidates = {}
        for cpt in M_values.keys():
            k = np.log(2) / M_values[cpt]['HT']
            Pamb = p_atm_sea_level + (deco_depth * rho_water * g)
            Pi0 = N2Part * (Pamb - p_water_vapor) / 1E5
            P0 = pN2_cpts_after_ascent["pN2_cpt_" + cpt + " [bar]"]
            P = shreiner(k, 0., P0, Pi0, time_at_stop_min * 60)
            pN2_cpts_candidates["pN2_cpt_" + cpt + " [bar]"] = P
        next_deco_depth, _ = get_next_deco_stop(N2Part, ascent_speed, deco_depth, pN2_cpts_candidates, idx_deco_stop = 0)           
        
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
        deco_stop, directing_cpt = get_next_deco_stop(N2Part, ascent_speed, curr_depth, pN2_cpts)
        if (deco_stop > 0.):
            (deco_time, pN2_cpts) = calculate_time_at_next_deco_stop(N2Part, ascent_speed, curr_depth, deco_stop, pN2_cpts)
            deco_stops.append({'DECO_DEPTH [m]' : deco_stop, 'DECO_TIME [min]' : deco_time, 'SPEED_ASCENT [m/sec]' : ascent_speed, 'DIRECTING CPT' : directing_cpt})
            curr_depth = deco_stop            
    return deco_stops



def add_deco_to_depth_profile(diving_profile, deco_profile):
    depths = diving_profile['Profondeur [m]'].tolist()
    times = diving_profile['Temps [sec]'].tolist()    
    depth = depths[-1]
    t = times[-1]  
    for deco_stop in deco_profile:
        speed_ascent = deco_stop['SPEED_ASCENT [m/sec]']
        deco_time = deco_stop['DECO_TIME [min]']
        deco_depth = deco_stop['DECO_DEPTH [m]']        
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
        time_to_stop = (depth - deco_stop['DECO_DEPTH [m]']) / (60 * deco_stop['SPEED_ASCENT [m/sec]'])
        time_at_stop = deco_stop['DECO_TIME [min]']
        tts += time_to_stop + time_at_stop
        depth = deco_stop['DECO_DEPTH [m]']
    tts += deco_profile[-1]['DECO_DEPTH [m]'] / (60 * deco_profile[-1]['SPEED_ASCENT [m/sec]'])
    tts = int(np.ceil(tts))
    return tts



def get_limit_depth_for_deco_mix(O2Part_Deco, O2Part_admissible = 1.6):
    return 10 * ((O2Part_admissible / O2Part_Deco) - 1)
    
    
    
def calculate_cpt_saturation_with_deco_mix(diving_profile, O2Part_adm = 1.6):
    max_depth_deco_mix = get_limit_depth_for_deco_mix(O2Part_Deco, O2Part_admissible = O2Part_adm)
    nb_steps = len(diving_profile)
    time_steps = list(diving_profile["Temps [sec]"])
    depth_steps = list(diving_profile["Profondeur [m]"])
    
    for i in range(len(diving_profile)):
        depth = depth_steps[len(diving_profile)-i-1]    
        if (depth > max_depth_deco_mix):
            first_timestep_deco = time_steps[len(diving_profile)-i-1]
            break
        
    for cpt in M_values.keys():
        #print(" > Cpt ° " + cpt)
        HT = M_values[cpt]['HT']
        k = np.log(2) / HT # k = half-time constant = ln2/half-time
        tissue_time_sat = []
        for i in range(nb_steps):
            if (i == 0):
                tissue_time_sat.append(0.79 * ((p_atm_sea_level - p_water_vapor) / 1E5))
            else:
                P0 = tissue_time_sat[-1] # initial compartment inert gas pressure [bar]
                Pamb = p_atm_sea_level + depth_steps[i] * rho_water * g
                depth_delta = depth_steps[i] - depth_steps[i-1]
                time_delta = (time_steps[i] - time_steps[i-1]).total_seconds()
                desc_speed = depth_delta / time_delta
                
                if ((time_steps[i] < first_timestep_deco) and (depth_steps[i] > 0.001)):
                    Pi0 = N2Part * ((Pamb - p_water_vapor) / 1E5) # initial inspired (alveolar) inert gas pressure [bar]
                    R = N2Part * desc_speed * ((rho_water * g) / 1E5) # R = rate of change in inspired gas pressure with change in ambient pressure
                
                elif ((time_steps[i] >= first_timestep_deco) and (depth_steps[i] > 0.001)):
                    Pi0 = (1.0 - O2Part_Deco) * ((Pamb - p_water_vapor) / 1E5) # initial inspired (alveolar) inert gas pressure [bar]
                    R = (1.0 - O2Part_Deco) * desc_speed * ((rho_water * g) / 1E5) # R = rate of change in inspired gas pressure with change in ambient pressure
                
                else:
                    Pi0 = 0.79 * ((Pamb - p_water_vapor) / 1E5) # initial inspired (alveolar) inert gas pressure [bar]
                    R = 0.79 * desc_speed * ((rho_water * g) / 1E5) # R = rate of change in inspired gas pressure with change in ambient pressure

                t = time_delta # time
                tissue_time_sat.append(shreiner(k, R, P0, Pi0, t))
            
        diving_profile['pN2_cpt_' + cpt + ' [bar]_deco'] = tissue_time_sat
    return diving_profile
    


# # 3 - Diving deco profile calculation

# In[6]:


des_speed = 1 / 60 #20 / 60 # unit : m/sec
asc_speed = 1.5 / 60 #10 / 60 # unit : m/sec
asc_speed_between_stops = 1.5 / 60 # unit : m/sec MUST BE GREATER THAN (OR EQUAL) asc_speed !!!!!
N2Part = 0.79 # unit : %
O2Part_Deco = 1.0 # unit : % O2 in deco mix
GF_low = 0.9 # unit : %
GF_high = 0.9 # unit : %

M_values = read_M_values_from_file("Buhlmann_Zh-L16C_M-values_bak.csv")
M_values_basis = M_values
diving_profile = define_dive_depth_profile(depth = 20, time = 120)
Pamb_max = (p_atm_sea_level + (rho_water * g * diving_profile['Profondeur [m]'].max()))/1E5
M_values = calculate_GF_lines(GF_low, GF_high, Pamb_max, M_values)
diving_profile = calculate_cpt_saturation_along_dive(diving_profile)
possible_deco_stops = [*range(0, 1 + int(np.floor(diving_profile['Profondeur [m]'].max())), 3)]
deco_profile = calculate_deco_profile(diving_profile, ascent_speed = asc_speed, ascent_speed_between_stops = asc_speed_between_stops)
deco_profile.append({'DECO_DEPTH [m]': 0., 'DECO_TIME [min]': 0, 'SPEED_ASCENT [m/sec]': asc_speed_between_stops})
depth_before_ascent = diving_profile['Profondeur [m]'].iloc[-1]
diving_profile = add_deco_to_depth_profile(diving_profile, deco_profile)
diving_profile = calculate_cpt_saturation_along_dive(diving_profile)
diving_profile_post_dive = calculate_cpt_saturation_after_dive(diving_profile, time_after_dive_min = 15)
diving_profile = pd.concat([diving_profile, diving_profile_post_dive], ignore_index = True)
diving_profile = calculate_cpt_saturation_with_deco_mix(diving_profile, O2Part_adm = 2.2)


# In[7]:


for deco_stop in deco_profile:
    print(deco_stop)


# In[8]:


total_time_to_surface(depth_before_ascent, deco_profile)


# # 4 - Diving profile

# In[9]:


fig, ax = plt.subplots(1, 1, figsize = (12, 7))
ax.plot(diving_profile['Temps [sec]'], -diving_profile['Profondeur [m]'], color = 'black', linewidth = 4)
xticks = ax.get_xticklabels(minor=False, which=None)
ax.set_xticklabels(diving_profile['Temps [sec]'], rotation = 45)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

for deco_stop in deco_profile:
    depth = deco_stop['DECO_DEPTH [m]']
    time_at_stop = deco_stop['DECO_TIME [min]']
    if (depth == 0.): break    
    dir_cpt = deco_stop['DIRECTING CPT']
    deco_time = diving_profile.loc[(diving_profile['Profondeur [m]'] >= (depth * 0.99)) & (diving_profile['Profondeur [m]'] <= (depth * 1.01)), "Temps [sec]"].tail(1).values[0]
    txt = " " + str(depth) + ' m - '+ str(time_at_stop) + ' min' + ' - cpt n°' + str(dir_cpt) 
    ax.text(deco_time, -depth - 0.5, txt)

plt.title('DEPTH-DIVING PROFILE', fontsize = 14)
plt.xlabel("TIME\n[ hh:mm:ss ]", fontsize = 12)
plt.ylabel("DEPTH [m]", fontsize = 12)
plt.show()


# # 5 - Tissues saturation along time

# In[10]:


diving_profile['Pamb'] = (p_atm_sea_level + (diving_profile['Profondeur [m]'] * rho_water * g)) / 1E5    

cpts = list(M_values.keys())
fig, axs = plt.subplots(len(cpts), 2, figsize = (14, 7 * len(cpts)))
i = 0
for cpt in cpts:
    col_name = "pN2_cpt_" + cpt + " [bar]"
    M0 = M_values_basis[cpt]['M0']
    M_slope = M_values_basis[cpt]['M_Slope']
    x = [i for i in range(1, 8)]
    M_line = [(p - 1) * M_slope + M0 for p in x]
    M0_gf = M_values[cpt]['M0']
    M_slope_gf = M_values[cpt]['M_Slope']
    GF_line = [(p - 1) * M_slope_gf + M0_gf for p in x]
    axs[i, 0].plot(x, M_line, color = 'red', linewidth = 3, label = "M-value line")
    axs[i, 0].plot(x, GF_line, color = 'red', linewidth = 3, linestyle = 'dotted', label = "GF line")   
    axs[i, 0].plot([1., 1.], [0., 7.], linestyle = '--', color = 'black', label = "Surface")
    axs[i, 0].plot(diving_profile['Pamb'], diving_profile[col_name], color = 'blue', label = "Deco profile")
    axs[i, 0].plot(diving_profile['Pamb'], diving_profile["pN2_cpt_" + cpt + " [bar]_deco"], color = 'green', linestyle = 'dotted', label = "Deco profile with deco mix - O2 : " + str(np.round(O2Part_Deco * 100)) + " %")
    axs[i, 0].set_xlabel('Ambiant pressure [bar]')
    axs[i, 0].set_ylabel('Inert gas tension [bar]')
    axs[i, 0].set_xlim([0., 7.])
    axs[i, 0].set_ylim([0., 7.])
    axs[i, 0].plot([0., 7.], [0., 7.], color = 'black', linestyle = '--', linewidth = 2)
    axs[i, 0].fill_between(x, GF_line, 7., color = 'red', alpha = 0.2)
    axs[i, 0].fill_between(x, M_line, 7., color = 'red', alpha = 0.3)
    axs[i, 0].set_title("TISSUE " + cpt.upper(), fontsize = 18)
    ttl = axs[i, 0].title
    ttl.set_position([1.1, 1.2])
    axs[i, 0].legend()
    
    Pamb_max = diving_profile['Pamb'].max()
    x1 = np.array([Pamb_max, Pamb_max])
    x2 = np.array([1+((Pamb_max-M0)/M_slope), Pamb_max])
    axs[i, 0].plot([x1[0], x2[0]], [x1[1], x2[1]], color = 'black', linewidth = 3, marker = '|', markersize = 10, markeredgewidth = 5)
    for k in range(0, 11, 1):
        x1_marker = x1 + (x2 - x1) * (k/10)
        if (k%2 == 0):
            axs[i, 0].scatter([x1_marker[0]], [x1_marker[1]], marker = '|', color = 'black', s = 50)
            axs[i, 0].text(x1_marker[0], x1_marker[1] + 0.25, str(k*10) + "%", fontsize = 10, rotation = 45)
        else:
            axs[i, 0].scatter([x1_marker[0]], [x1_marker[1]], marker = '|', color = 'black', s = 30)

    x1 = np.array([1., 1.])
    x2 = np.array([1, M0])
    axs[i, 0].plot([x1[0], x2[0]], [x1[1], x2[1]], color = 'black', linewidth = 3, marker = '_', markersize = 10, markeredgewidth = 5)
    for z in range(0, 11, 1):
        x1_marker = x1 + (x2 - x1) * (z/10)
        if (z%2 == 0):
            axs[i, 0].scatter([x1_marker[0]], [x1_marker[1]], marker = '_', color = 'black', s = 50)
            axs[i, 0].text(x1_marker[0] - 0.75, x1_marker[1], str(z*10) + "%", fontsize = 10)        
        else:
            axs[i, 0].scatter([x1_marker[0]], [x1_marker[1]], marker = '_', color = 'black', s = 30)
       
    M_val_lim = (diving_profile['Pamb'] - 1.) * M_slope + M0
    M_val_lim_gf = (diving_profile['Pamb'] - 1.) * M_slope_gf + M0_gf
    axs[i, 1].plot(diving_profile['Temps [sec]'], M_val_lim, color = 'red', linewidth = 3, label = "Max admissible inert gas tension")
    axs[i, 1].plot(diving_profile['Temps [sec]'], M_val_lim_gf, color = 'red', linewidth = 3, linestyle = 'dotted', label = "Max admissible inert gas tension - GF")
    axs[i, 1].plot(diving_profile['Temps [sec]'], diving_profile['Pamb'], color = 'black', label = "Ambiant pressure")
    axs[i, 1].plot(diving_profile['Temps [sec]'], diving_profile[col_name], color = 'blue', label = "Inert gas tension")
    axs[i, 1].plot(diving_profile['Temps [sec]'], diving_profile["pN2_cpt_" + cpt + " [bar]_deco"], color = 'green', linestyle = 'dotted', label = "Inert gas tension with deco mix - O2 : " + str(np.round(O2Part_Deco * 100, 0)) + " %")
    axs[i, 1].set_xticklabels(diving_profile['Temps [sec]'], rotation = 30)
    axs[i, 1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axs[i, 1].fill_between(diving_profile['Temps [sec]'], M_val_lim_gf, 100., color = 'red', alpha = 0.2)
    axs[i, 1].fill_between(diving_profile['Temps [sec]'], M_val_lim, 100., color = 'red', alpha = 0.3)
    axs[i, 1].set_xlabel('Time [hh:mm:ss]')
    axs[i, 1].set_ylabel('Pressure [bar]')
    axs[i, 1].set_ylim([0., 7.])
    axs[i, 1].legend()   
    i += 1

plt.savefig("deco_profile.pdf")


# In[11]:


plt.figure(figsize = (10, 10))
x = [*np.arange(1.0, 7.0, 0.2)]

for cpt in M_values_basis.keys():
    y = [(i-1.) * M_values_basis[cpt]['M_Slope'] + M_values_basis[cpt]['M0'] for i in x]
    plt.plot(x, y, label = 'Cpt n°' + cpt)

plt.plot([1., 1.], [0., 7.], label = "Surface pressure", linestyle = '--', color = 'black', linewidth = 3)
plt.plot([0., 7.], [0., 7.], label = "Ambiant pressure line", linestyle = 'dotted', color = 'black', linewidth = 3)
plt.legend()
plt.xlabel('PRESSURE\n[bar]', fontsize = 14)
plt.ylabel('INERT GAS TENSION\n[bar]', fontsize = 14)
plt.title('M-VALUES LINES', fontsize = 18)
plt.xlim([0, 7])
plt.ylim([0, 7])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




