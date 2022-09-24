import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time
import cvxopt
from collections import (namedtuple, defaultdict)
import os.path
import os
#from mpl_toolkits.mplot3d import Axes3D
#from memory_profiler import memory_usage


#from BloodManagementParsAndInitialState import *
from DriverAssignmentNetwork import *
from DriverAssignmentModel import *
from DriverAssignmentPolicy import initLPMatrices,Policy

def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.get_memory_info().rss


def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before,
            elapsed_time))
        return result
    return wrapper


def printParams(params):
    print(params) 
        

def loadParams(filename):

    parDf = pd.read_excel(filename, sheet_name = 'Parameters')
    parDict=parDf.set_index('Index').T.to_dict('list')
    params = {key:v for key, value in parDict.items() for v in value}

    params['PRINT']=False
    params['PRINT_ALL']=False
    params['OUTPUT_FILENAME'] = 'DetailedOutput.xlsx'

    params['SHOW_PLOTS']=False




def Main():    
    
    t_global_init = time.time()
    print("********************Started Main*****************\n")
    params = loadParams('Parameters.xlsx')
    alpha = params['ALPHA']
    
    #ite_TRA=np.arange(0, params['NUM_TRAINNING_ITER'], 1)
    #selectedIte = list(set([0,5,10,19]) & set(ite_TRA))
    
    

    # initializing the random seed for trainning iterations
    np.random.seed(params['SEED_TRAINING'])

    # initializing the blood network
    Dri_Net = create_dri_net(params)

    if (params['LOAD_VFA'] and os.path.exists(params['NAME_LOAD_VFA_PICKLE'])):
        pickle_off = open(params['NAME_LOAD_VFA_PICKLE'],"rb")
        Other_Bld_Net = pickle.load(pickle_off)
        Dri_Net.varr = Other_Bld_Net.varr.copy()
        Dri_Net.parallelarr = Other_Bld_Net.parallelarr.copy()

    # initializing the model
    state_names = ['DriversFleet', 'Demand']
    decision_names = ['Hold', 'Contribution']

    # initializing the lists that will store the all the info/decisions/states/slopes along the iterations for printing purposes
    demandExoList, donationExoList, supplyPreList, supplyPostList, slopesList, solDemList, solHoldList,  simuList, updateVfaList = [],[],[],[],[],[],[],[],[]

    #initializing the policy
    P = Policy(params,Dri_Net)
    
    iteration = 0
    obj = []

    if (params['NUM_TRAINNING_ITER']>0):
        print("\n Starting training iterations\n")

    while iteration < params['NUM_ITER']:  
        IS_TRAINING = (iteration<params['NUM_TRAINNING_ITER'])
        if (iteration==params['NUM_TRAINNING_ITER']):
            print("Starting testing iterations! Currently at iteration ",iteration)
            print("Reseting random seed!")
            np.random.seed(params['SEED_TESTING'])
            
        t_init = time.process_time()
        print('Iteration = ', iteration)
        
        # Initial fleet
        if (params['SAMPLING_DIST'] == 'P'):
            drifle_init = [int(np.random.poisson(params['MAX_DON_BY_BLOOD'][dri[0]])*.9) if dri[1]=='0' else int(np.random.poisson(params['MAX_DON_BY_BLOOD'][dri[0]])*(0.1/(params['MAX_AGE']-1))) for dri in Dri_Net.drivernodes]
        else:
            drifle_init = [round(np.random.uniform(0, params['MAX_DON_BY_BLOOD'][dri[0]])*.9) if dri[1]=='0' else round(np.random.uniform(0, params['MAX_DON_BY_BLOOD'][dri[0]])*(0.1/(params['MAX_AGE']-1))) for dri in Dri_Net.drivernodes]
    
        # Initial exogenous information
        if (params['SAMPLING_DIST'] == 'P'):
            exog_info_init = generate_exog_info_by_bloodtype_p(0, Dri_Net, params)
        else:
            exog_info_init = generate_exog_info_by_bloodtype(0, Dri_Net, params)

        #initial state - only the initial fleet counts
        init_state = {'DriversFleet': drifle_init, 'Demand': exog_info_init.demand}

        M = Model(state_names, decision_names, init_state, Dri_Net,params)
        #print("Initial blood supply across {} types and {} ages is {}".format(params['NUM_BLD_TYPES'],params['MAX_AGE'],sum(M.bld_inv)))
        #print("Initial demand across {} types and {} urgency states and {} substitution states is {}".format(params['NUM_BLD_TYPES'],params['NUM_SUR_TYPES'],len(params['Substitution']),sum(M.demand)))

        
        t = 0        
        obj.append(0)

        #Steping forward in time
        while t < params['MAX_TIME']:
          
            
            #Compute the solution for time period t - return the solution, the value, the dual and the updated lists
            sol,val,x,hld,d,solDemList,solHoldList=P.getLPSol(params,M,iteration,t,solDemList,solHoldList,IS_TRAINING)
            obj[iteration] += val
            
            
            #Grabbing exogenous data to construct data frame
            recordDemandExo = (iteration,t,M.Bld_Net.demandamount.copy())
            demandExoList.append(recordDemandExo)       
            if (t==0):   
                recordDonationExo = (iteration,0,list(np.array(M.bld_inv)[::params['MAX_AGE']]))
                donationExoList.append(recordDonationExo)
            
            if (t<params['MAX_TIME']-1):
                recordDonationExo = (iteration,t+1,M.donation.copy())
                donationExoList.append(recordDonationExo)
                
            #Grabbing pre-decision state to construct data frame
            recordSupplyPre = (iteration,t,M.bld_inv.copy())
            supplyPreList.append(recordSupplyPre)
                
            
            if IS_TRAINING:    
                alpha,slopesList,updateVfaList = P.updateVFAs(params,M,iteration,t,d, slopesList,updateVfaList)
                            
            # build decision
            dcsn = M.build_decision({'Hold': hld, 'Contribution': val})
                    
            M.transition_fn(dcsn)
            
            #Grabbing post-decision state to construct data frame
            recordSupplyPost = (iteration,t,M.bld_inv.copy())
            supplyPostList.append(recordSupplyPost)
            
            t += 1
            # generate/read exogenous information 
            if (params['SAMPLING_DIST'] == 'P'):
                exog_info = generate_exog_info_by_bloodtype_p(t, Bld_Net, params)
            else:
                exog_info = generate_exog_info_by_bloodtype(t, Bld_Net, params)
            M.exog_info_fn(exog_info)
            
        
        
        # copy v to the parallel links
        for t in params['Times']:
            for hld in M.Bld_Net.holdnodes:
                parArr = 1 * M.Bld_Net.varr[(t,hld, M.Bld_Net.supersink)]
                M.Bld_Net.add_parallel(t,hld, M.Bld_Net.supersink, parArr)
        
          

        t_end = time.process_time()
        recordSimu = (iteration,int(t_end-t_init),alpha,obj[iteration],(iteration<params['NUM_TRAINNING_ITER']))
        simuList.append(recordSimu)
       
        print("***Finishing iteration {} in {:.2f} secs. Total contribution: {:.2f}***\n".format(recordSimu[0],recordSimu[1],recordSimu[3]))
        
        
        iteration += 1
        
    #End of iterations
    ###########################################################################################################################################


    if (params['SAVE_VFA']):
        pickling_on = open(params['NAME_SAVE_VFA_PICKLE'],"wb")
        pickle.dump(M.Bld_Net, pickling_on)
        pickling_on.close()


#End Main
###############################################################################################################################################
     

    
###############################################################################################################################################
if __name__ == "__main__":
    Main()
    
    #mem = max(memory_usage(proc=Main))
    #print("Maximum memory used: {0} MiB".format(str(mem)))