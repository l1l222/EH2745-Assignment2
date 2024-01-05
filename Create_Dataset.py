import pandapower as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl

def build_network():
    """
    Build a network based on the given data.
    Input: none
    Output: pandapower network
    """

    net = pp.create_empty_network()
    
    # create buses
    bus1 = pp.create_bus(net, name="CLARK-Region1", vn_kv=110, type="b")
    bus2 = pp.create_bus(net, name="AMHERST-Region1", vn_kv=110, type="b")
    bus3 = pp.create_bus(net, name="WINLOCK-Region1", vn_kv=110, type="b")
    bus4 = pp.create_bus(net, name="BOWMAN-Region2", vn_kv=110, type="b")
    bus5 = pp.create_bus(net, name="TROY-Region2", vn_kv=110, type="b")
    bus6 = pp.create_bus(net, name="MAPLE-Region2", vn_kv=110, type="b")
    bus7 = pp.create_bus(net, name="GRAND-Region3", vn_kv=110, type="b")
    bus8 = pp.create_bus(net, name="WAUTAGA-Region3", vn_kv=110, type="b")
    bus9 = pp.create_bus(net, name="CROSS-Region3", vn_kv=110, type="b")
    
    # create lines
    pp.create_line(net, bus1, bus4, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 1-4")
    pp.create_line(net, bus4, bus9, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 4-9")
    pp.create_line(net, bus8, bus9, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 8-9")
    pp.create_line(net, bus2, bus8, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 2-8")
    pp.create_line(net, bus7, bus8, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 7-8")
    pp.create_line(net, bus6, bus7, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 6-7")
    pp.create_line(net, bus3, bus6, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 3-6")
    pp.create_line(net, bus5, bus6, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 5-6")
    pp.create_line(net, bus4, bus5, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 4-5")

    # create generators
    pp.create_gen(net, bus1, p_mw=0, slack=True, name="Generator 1") # slack bus
    pp.create_sgen(net, bus2, p_mw=163, q_mvar=0, name="Generator 2")
    pp.create_sgen(net, bus3, p_mw=85, q_mvar=0, name="Generator 3")
    
    # create loads
    pp.create_load(net, bus5, p_mw=90, q_mvar=30, name="Load1")
    pp.create_load(net, bus7, p_mw=100, q_mvar=35, name="Load2")
    pp.create_load(net, bus9, p_mw=125, q_mvar=50, name="Load3")
    
    return net

# Timeseries powerflow create
def create_datasource(net,operating_states,no_time_steps=60,no_test_time_steps=15):
    '''
    high load = 1
    low load = 2
    generator disconnection = 3
    line disconnection = 4
    '''
    training_set = pd.DataFrame()
    test_set = pd.DataFrame()
    if operating_states == 1:
    
        # High Load state
        for i in range(3):
            
            #Training set
            training_set['load{}_P'.format(str(i+1))] = 1.2 * net.load.p_mw[i] + (0.05 * np.random.random(no_time_steps) * net.load.p_mw[i])
            training_set['load{}_Q'.format(str(i+1))] = 1.2 * net.load.q_mvar[i] + (0.05 * np.random.random(no_time_steps) * net.load.q_mvar[i])
                        
            # Test set
            test_set['load{}_P'.format(str(i+1))] = 1.2 * net.load.p_mw[i] + (0.05 * np.random.random(no_test_time_steps) * net.load.p_mw[i])
            test_set['load{}_Q'.format(str(i+1))] = 1.2 * net.load.q_mvar[i] + (0.05 * np.random.random(no_test_time_steps) * net.load.q_mvar[i])
                                                          
    elif operating_states == 2:
         
        #Low Load state       
        for i in range(3):
            
            #Training set
            training_set['load{}_P'.format(str(i+1))] = 0.8 * net.load.p_mw[i] + (0.05 * np.random.random(no_time_steps) * net.load.p_mw[i])
            training_set['load{}_Q'.format(str(i+1))] = 0.8 * net.load.q_mvar[i] + (0.05 * np.random.random(no_time_steps) * net.load.q_mvar[i])
                        
            # Test set
            test_set['load{}_P'.format(str(i+1))] = 0.8 * net.load.p_mw[i] + (0.05 * np.random.random(no_test_time_steps) * net.load.p_mw[i])
            test_set['load{}_Q'.format(str(i+1))] = 0.8 * net.load.q_mvar[i] + (0.05 * np.random.random(no_test_time_steps) * net.load.q_mvar[i])
        
    elif operating_states == 3:
        
        # Generator Disconnection state
        for i in range(3):
            
            #Training set
            training_set['load{}_P'.format(str(i+1))] = 0.8 * net.load.p_mw[i] + (0.05 * np.random.random(no_time_steps) * net.load.p_mw[i])
            training_set['load{}_Q'.format(str(i+1))] = 0.8 * net.load.q_mvar[i] + (0.05 * np.random.random(no_time_steps) * net.load.q_mvar[i])
                        
            # Test set
            test_set['load{}_P'.format(str(i+1))] = 0.8 * net.load.p_mw[i] + (0.05 * np.random.random(no_test_time_steps) * net.load.p_mw[i])
            test_set['load{}_Q'.format(str(i+1))] = 0.8 * net.load.q_mvar[i] + (0.05 * np.random.random(no_test_time_steps) * net.load.q_mvar[i])
       
        training_set['generator1_P'] = [0]*no_time_steps
        training_set['generator1_Q'] = [0]*no_time_steps
        
        test_set['generator1_P'] = [0]*no_test_time_steps
        test_set['generator1_Q'] = [0]*no_test_time_steps
        
    elif operating_states == 4:
                
        # Line Disconnection state
        for i in range(3):
            
            #Training set
            training_set['load{}_P'.format(str(i+1))] = net.load.p_mw[i] + (0.05 * np.random.random(no_time_steps) * net.load.p_mw[i])
            training_set['load{}_Q'.format(str(i+1))] = net.load.q_mvar[i] + (0.05 * np.random.random(no_time_steps) * net.load.q_mvar[i])
                        
            # Test set
            test_set['load{}_P'.format(str(i+1))] = net.load.p_mw[i] + (0.05 * np.random.random(no_test_time_steps) * net.load.p_mw[i])
            test_set['load{}_Q'.format(str(i+1))] = net.load.q_mvar[i] + (0.05 * np.random.random(no_test_time_steps) * net.load.q_mvar[i])
            
        pp.create_switch(net, bus=5, closed=False, element=6, et='l', type='CB')
    
    DataSource_training = DFData(training_set)
    DataSource_test = DFData(test_set)
    
    return DataSource_training,DataSource_test

#Create controllers
def create_controller(net, DataSource, generator_disconnection=False):
    
    ConstControl(net, element='load', variable='p_mw', element_index=[0],
                     data_source=DataSource, profile_name=["load1_P"])
    ConstControl(net, element='load', variable='q_mvar', element_index=[0],
                     data_source=DataSource, profile_name=['load1_Q'])
    
    ConstControl(net, element='load', variable='p_mw', element_index=[1],
                     data_source=DataSource, profile_name=["load2_P"])
    ConstControl(net, element='load', variable='q_mvar', element_index=[1],
                     data_source=DataSource, profile_name=['load2_Q'])
    
    ConstControl(net, element='load', variable='p_mw', element_index=[2],
                     data_source=DataSource, profile_name=["load3_P"])
    ConstControl(net, element='load', variable='q_mvar', element_index=[2],
                     data_source=DataSource, profile_name=['load3_Q'])
    
    if generator_disconnection == True:
        ConstControl(net, element='sgen', variable='p_mw', element_index=[0],
                         data_source=DataSource, profile_name=["generator1_P"])
        ConstControl(net, element='sgen', variable='q_mvar', element_index=[0],
                         data_source=DataSource, profile_name=["generator1_Q"])
        
    return net

#Save training sets results
def save_training_results(net, operating_states, no_time_steps=60):
    if operating_states == 1:
        output_dir = os.path.join(os.getcwd(), "time_series_training", "high_load")
    elif operating_states == 2:
        output_dir = os.path.join(os.getcwd(), "time_series_training", "low_load")
    elif operating_states == 3:
        output_dir = os.path.join(os.getcwd(), "time_series_training", "generator_disconnection")
    elif operating_states == 4:
        output_dir = os.path.join(os.getcwd(), "time_series_training", "line_disconnection")
            
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ow = OutputWriter(net, no_time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())
    # these variables are saved during the time series loop
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    
    print("Current network data is saved in: {}".format(output_dir)) 
    
    # Plotting voltage magnitude and angle
    # Voltage angle
    vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
    vm_pu_tot = pd.read_excel(vm_pu_file)
    vm_pu = vm_pu_tot[[0,1,2,3,4,5,6,7,8]]
    vm_pu.plot(label="vm_pu")
    plt.xlabel("time step")
    plt.ylabel("voltage mag. [p.u.]")
    plt.title("Voltage Magnitude")
    plt.grid()
    plt.show()
    
    # Voltage angle
    va_degree_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
    va_degree_tot = pd.read_excel(va_degree_file)
    va_degree = va_degree_tot[[0,1,2,3,4,5,6,7,8]]
    va_degree.plot(label="va_degree")
    plt.xlabel("time step")
    plt.ylabel("voltage angle. [degree]")
    plt.title("Voltage Angle")
    plt.grid()
    plt.show()
    
    return ow

#Save test sets results
def save_test_results(net, operating_states, no_time_test_steps=15):
    if operating_states == 1:
        output_dir = os.path.join(os.getcwd(), "time_series_test", "high_load")
    elif operating_states == 2:
        output_dir = os.path.join(os.getcwd(), "time_series_test", "low_load")
    elif operating_states == 3:
        output_dir = os.path.join(os.getcwd(), "time_series_test", "generator_disconnection")
    elif operating_states == 4:
        output_dir = os.path.join(os.getcwd(), "time_series_test", "line_disconnection")
            
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ow = OutputWriter(net, no_time_test_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())
    # these variables are saved during the time series loop
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    
    print("Current network data is saved in: {}".format(output_dir)) 
    return ow
    
#Create data for training and test
def create_dataset():
    
    #Generate training data
    #high load
    net1 = build_network()
    training_set1, test_set1 = create_datasource(net1,1)   
    net1 = create_controller(net1, training_set1)
    save_training_results(net1, operating_states=1)
    run_timeseries(net1, no_time_steps=60, calculate_voltage_angles=True)
    
    
    #low load
    net2 = build_network()
    training_set2, test_set2 = create_datasource(net2,2)
    net2 = create_controller(net2, training_set2)
    save_training_results(net2,operating_states=2)
    run_timeseries(net2, no_time_steps=60, calculate_voltage_angles=True)
    
    #generator disconnection
    net3 = build_network()
    training_set3, test_set3 = create_datasource(net3,3)
    net3 = create_controller(net3, training_set3,generator_disconnection=True)
    save_training_results(net3, operating_states=3)
    run_timeseries(net3, no_time_steps=60, calculate_voltage_angles=True)
    
    #line disconnection
    net4 = build_network()
    training_set4, test_set4 = create_datasource(net4,4)
    net4 = create_controller(net4, training_set4)
    save_training_results(net4, operating_states=4)
    run_timeseries(net4, no_time_steps=60, calculate_voltage_angles=True)
    
    
    
    # Generate test data
    # high load
    net1 = build_network()  
    net1 = create_controller(net1, test_set1)
    save_test_results(net1, operating_states=1)
    run_timeseries(net1, no_time_steps=15, calculate_voltage_angles=True)
    
    
    # low load
    net2 = build_network()
    training_set2, test_set2 = create_datasource(net2,2)
    net2 = create_controller(net2, test_set2)
    save_test_results(net2, operating_states=2)
    run_timeseries(net2, no_time_steps=15, calculate_voltage_angles=True)
    
    # generator disconnection
    net3 = build_network()
    training_set3, test_set3 = create_datasource(net3,3)
    net3 = create_controller(net3, test_set3,generator_disconnection=True)
    save_test_results(net3, operating_states=3)
    run_timeseries(net3, no_time_steps=15, calculate_voltage_angles=True)
    
    # line disconnection
    net4 = build_network()
    net4 = create_controller(net4, test_set4)
    save_test_results(net4, operating_states=4)
    run_timeseries(net4, no_time_steps=15, calculate_voltage_angles=True)
    
     
if __name__ == "__main__":
    create_dataset()

