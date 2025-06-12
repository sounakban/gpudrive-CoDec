"""
    This part of pipeline 1 deals with the inference part of the logic. It computes the likelihood that different construed agents produced 
    observed trajectories (generalet in first part of the pipeline), then generates estimates of lambda parameters.
"""


# |Set parent to current working directory for imports
from genericpath import isfile
import os
import sys
from pathlib import Path

working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)
sys.path.append(str(working_dir))

# |Import everything
from examples.CoDec_Research.code.Experiment_1.exp1_imports import *





#####################################################
################ PARAMETER INFERENCE ################
#####################################################




def compute_construal_likelihoods(
                                    train_loader: SceneDataLoader,
                                    out_dir: str,
                                    processID: str,
                                    exp_config: Dict,
                                    device: str,
                                    synthData_params: Dict = None,
                                 ):
    """
    First part of inference logic. Compute likelihoods of given data under each construal.
    """
    ### Compute Construal Log Likelihoods ###
    curr_data_batch = [env_path2name(scene_path_) for scene_path_ in train_loader.dataset]
    construal_action_likelihoods = None

    # |Check if saved data is available
    intermediate_results_files = get_files_at_loc(out_dir)
    for srFile in intermediate_results_files:
        if "log_likelihood_dict" in srFile:
            with open(srFile, 'rb') as opn_file:
                construal_action_likelihoods = pickle.load(opn_file)
            #2# |Ensure the correct file is being loaded
            if all(env_path2name(scene_path_) in construal_action_likelihoods.keys() for scene_path_ in train_loader.dataset) and \
                    construal_action_likelihoods["params"] == synthData_params:
                print(f"Using log-likelihood dictionary from file: {srFile}")
                break
            else:
                construal_action_likelihoods = None

    # Loop through multiple synthetic data files which (combined) contains data for all current scenes
    print("Could not find saved likelihood estimations for this dataset, now computing.")
    if construal_action_likelihoods is None:
        construal_action_likelihoods = {}
        for srFile in intermediate_results_files:
            if "baseline_state_action_pairs" in srFile:
                state_action_pairs = ObservationDataManager.load_data(srFile, decompress_data=exp_config['compress_synthetic_data'])
                #2# |Ensure the correct file is being loaded
                fileParams = state_action_pairs.pop("params")
                state_action_pairs.pop('dict_structure')
                # print(fileParams)
                if set(state_action_pairs.keys()).issubset(set(curr_data_batch)) and fileParams == synthData_params:
                    print(f"Using synthetic baseline data from file: {srFile}")
                    construal_action_likelihoods.update(evaluate_construals(state_action_pairs, construal_size, sim_agent, device=device))
                # |Clear memory for large variable
                del state_action_pairs
                gc.collect()

        if construal_action_likelihoods == {}:
            raise FileNotFoundError("Compatible baseline data not found. Please run part 1 of pipiline on current scenes.")

        # |Save data
        savefl_path = out_dir+processID+"_"+"log_likelihood_dict_"+str(datetime.now())+".pickle"
        construal_action_likelihoods["params"] = synthData_params # Save parameters for data generation
        save_pickle(savefl_path, construal_action_likelihoods, "Log Likelihood")

    # |Use parameters only for matching data identity (not needed beyond this point)
    construal_action_likelihoods.pop("params")



    ### Convert Results to Pandas Dataframe and Save ###
    construal_action_likelihoods_df = None

    # |Current process parameters for file name
    heuristic_params_str = '_'.join([heur_+str(param_) for heur_, param_ in synthData_params.items()])

    # |Check if data is already available
    for srFile in intermediate_results_files:
        if 'log_likelihood_DF' in srFile and heuristic_params_str in srFile and processID in srFile:
            construal_action_likelihoods_df = pd.read_csv(srFile, sep='\t')
            construal_action_likelihoods_df['base_construal'] = construal_action_likelihoods_df['base_construal'].map(eval) # String to tuple
            construal_action_likelihoods_df['test_construal'] = construal_action_likelihoods_df['test_construal'].map(eval) # String to tuple
            print(f"Using log-likelihood dataframe from file: {srFile}")
            break        

    if construal_action_likelihoods_df is None:
        # |If data not already available, convert to pandas dataframe and save
        construal_action_likelihoods_df = {(scene,baseC,testC,sample): construal_action_likelihoods[scene][baseC][sample][testC]['log_likelihood'].item()
                                                for scene in construal_action_likelihoods.keys() 
                                                for baseC in construal_action_likelihoods[scene].keys() 
                                                for sample in construal_action_likelihoods[scene][baseC].keys()
                                                for testC in construal_action_likelihoods[scene][baseC][sample].keys()
                                                }

        multi_index = pd.MultiIndex.from_tuples(construal_action_likelihoods_df.keys(), names=['scene', 'base_construal', 'test_construal', 'sample'])
        construal_action_likelihoods_df = pd.DataFrame(construal_action_likelihoods_df.values(), index=multi_index)
        construal_action_likelihoods_df.columns = ['-log_likelihood']
        construal_action_likelihoods_df = construal_action_likelihoods_df.reset_index()
        construal_action_likelihoods_df.to_csv(out_dir + processID + "_" + f"log_likelihood_DF_{heuristic_params_str}.tsv", sep="\t", index=False, header=True)

        # construal_action_likelihoods_summarydf = construal_action_likelihoods_df.groupby(level=(0,1,2)).mean().sort_values(by='-log_likelihood', ascending=True).\
        #                                             groupby(level=(0,1)).head(5).sort_index(level=(0,1), sort_remaining=False)
        # construal_action_likelihoods_summarydf.to_csv(intermediate_results_path + processID + "_" + f"log_likelihood_DF_summary_{heuristic_params_str}.tsv", sep="\t", index=True, header=True)

    # return (construal_action_likelihoods, construal_action_likelihoods_df)
    return construal_action_likelihoods_df







def compute_param_likelihoods(
                                target_params: List,
                                train_loader: SceneDataLoader,
                                out_dir: str,
                                processID: str,
                                exp_config: Dict,
                                heuristic_params_vals: Dict = None,
                                synthData_params: Dict = None,
                                construal_action_likelihoods_df: pd.DataFrame = None,
                             ):
    """
    Compute the likelihood of the given data for different parameter values or find the most likely parameter value.
    """
    # |Check for saved construal utility values
    intermediate_results_files = get_files_at_loc(out_dir)
    for srFile in intermediate_results_files:
        if "construal_vals" in srFile:
            with open(srFile, 'rb') as opn_file:
                default_values = pickle.load(opn_file)
            #2# |Ensure the correct file is being loaded
            if all(env_path2name(scene_path_) in default_values.keys() for scene_path_ in train_loader.dataset):
                print(f"Using construal values from file: {srFile}")
                break
            else:
                default_values = None
    if default_values is None:
        raise FileNotFoundError("Could not find saved file for construal values for current scenes")


    # |Define variables for inference process
    curr_heuristic_params = deepcopy(synthData_params)
    get_constral_heurisrtic_values_partial = partial(get_constral_heurisrtic_values, env=env, 
                                                    train_loader=train_loader, default_values=default_values)
        
        
    # |Define the function to compute the likelihood of a lambda value, given the data.
    def get_lambda_likelihood(**kwargs) -> float:
        flag = False
        # if isinstance(lambda_heur[0], list):
        if isinstance(list(kwargs.values())[0], list):
            #4# |If using skopt optimizer, lambda_heur is a list with one element
            lambda_heur = lambda_heur[0]
            flag = True
        #2# |lambda_heur: hyper parameter to optimize for
        for param_, val_ in kwargs.items():
            curr_heuristic_params[param_] = val_
        curr_heuristic_values = get_constral_heurisrtic_values_partial(heuristic_params=curr_heuristic_params)
        #2# |Deep copy the dataframe then perform operations
        curr_construal_likelihoods = deepcopy(construal_action_likelihoods_df)
        #2# |Convert log likelihoods to likelihoods
        curr_construal_likelihoods['traj_constr_likelihoods'] = np.exp(-1*curr_construal_likelihoods['-log_likelihood'])
        #2# |Get construal selection probs under lambda value
        curr_construal_likelihoods['construal_probs'] = [curr_heuristic_values[row['scene']][row['test_construal']] 
                                                            for _, row in curr_construal_likelihoods.iterrows()]
        #2# |Get likelihood for trajectories given construals
        curr_construal_likelihoods['construal_likelihoods'] = curr_construal_likelihoods['construal_probs']*curr_construal_likelihoods['traj_constr_likelihoods']
        #2# |Group by trajectory and add 'construal_likelihoods' values
        traj_log_likelihoods = np.log(curr_construal_likelihoods.groupby(by=['scene','base_construal','sample']).sum()['construal_likelihoods'].to_list())
        #2# |Take product of all likelihoods (log sum)
        if flag:
            #3# skopt optimizer minimizes value. So, positive log likelihood value is returned
            return -1*traj_log_likelihoods.sum().item()
        return traj_log_likelihoods.sum().item()


    # |Perform inference
    if exp_config['lambda_distribution']:    
        #2# |Get probability distribution over all integer lambda values by looping over range

        lamda_inference = {}
        target_param_value_range = {param_:heuristic_params_vals[param_] for param_ in target_params}
        target_param_value_sets = zip(*target_param_value_range.values())       # Assuming equal sample count for all parameters
        for curr_val_set in target_param_value_sets:
            curr_args = tuple(zip(target_param_value_range.keys(), curr_val_set))
            lamda_inference[curr_args] = -1*get_lambda_likelihood(**dict(curr_args))    # multiply by '-1' for negative log likelihood

        # print("lamda_inference: ", lamda_inference)
        resultFile = out_dir + processID + "_" + "inference_distribution_"+str(datetime.now())+".json"
        lamda_inference["TrueParams"] = synthData_params
        with open(resultFile, 'w') as json_file:
            json_file.write(str(lamda_inference))
            # json.dump(lamda_inference, json_file, indent=4)
        return None

    else:
        #2# |Perform bayesian optimization to get best fit lambda value for observed behavior (trajectories)

        #3# |Number of iterations to perform bayes optimization: proportional to number of parameters to be inferred
        iter_ratio = 1 + 0.4*(len(target_params) - 1)
        init_points = int(7*iter_ratio)     # Number of points for first esimate of function
        n_iter = int(8*iter_ratio)          # Number of iterations after first estimation to improve estimation

        #3# Create a BayesianOptimization optimizer, and optimize the given black_box_function.
        # pbounds = {"lambda_heur": [-15, 15]}    # Set range of lambda to optimize for.
        pbounds = {param_:[heuristic_params_vals[param_].min().item(), heuristic_params_vals[param_].max().item()] 
                        for param_ in target_params}    # Set range of lambda to optimize for.
        optimizer = BayesianOptimization(f = get_lambda_likelihood,
                                        pbounds = pbounds, verbose = 0,
                                        random_state = 4)
        optimizer.maximize(init_points = init_points, n_iter = n_iter)

        print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))
        # return (optimizer.max["params"]["lambda_heur"].item(), optimizer.max["target"])
        return (optimizer.max["params"], optimizer.max["target"])

        # #3# Using skopt BayesianOptimization optimizer.
        # pbounds = [Real(-15,15,name="lambda_heur")]
        # res_gp = gp_minimize(get_lambda_likelihood,
        #                         pbounds, n_calls = 15*iter_ratio,
        #                         random_state = 4)
        # print("Best result: {}; f(x) = {}.".format(res_gp.x, res_gp.fun))























if __name__ == "__main__":

    ##### Set Up Environments #####
    
    from examples.CoDec_Research.code.Experiment_1.exp1_config import *

    if len(sys.argv) > 1:
        # |If calling code with arguments
        # |Uses values from configuration file if values are not passed as argument
        try:
            target_param, target_heur_val = sys.argv[1], sys.argv[2]     # Changing global variable
            target_param = target_param.split(',')
            target_heur_val = [float(i) for i in target_heur_val.split(',')]
            for param_, val_ in zip(target_param, target_heur_val):
                active_heuristic_params[param_] = val_
        except Exception as e:
            raise ValueError("could not run python file: invalid arguments."
                                " Please ensure first argument is a comma-separated (no space)"
                                " list of heuristic names and the second is a comma-separated"
                                " (no space) list of values (Ex: 2,3,4)")
        
            
            
    moving_veh_masks = get_mov_veh_masks(
                                        training_config=training_config, 
                                        device=device, 
                                        dataset_path=dataset_path,
                                        max_agents=moving_veh_count,
                                        result_file_loc=intermediate_results_path,
                                        processID=processID
                                        )

    env_config, train_loader, env, sim_agent = get_gpuDrive_vars(
                                                                training_config=training_config,
                                                                device=device,
                                                                num_parallel_envs=num_parallel_envs,
                                                                dataset_path=dataset_path,
                                                                total_envs=total_envs,
                                                                sim_agent_path="daphne-cornelisse/policy_S10_000_02_27",
                                                                )


    ##### Run Main Code #####

    start_time = time.perf_counter()

    construal_action_likelihoods_df = compute_construal_likelihoods(
                                                                    train_loader= train_loader,
                                                                    out_dir= intermediate_results_path,
                                                                    processID= processID,
                                                                    exp_config= curr_config,
                                                                    device= device,
                                                                    synthData_params= active_heuristic_params,
                                                                )
    
    result = compute_param_likelihoods(
                                        target_params= target_param,
                                        train_loader= train_loader,
                                        out_dir= intermediate_results_path,
                                        processID= processID,
                                        exp_config= curr_config,
                                        heuristic_params_vals= heuristic_params_vals,
                                        synthData_params= active_heuristic_params,
                                        construal_action_likelihoods_df= construal_action_likelihoods_df,
                                    )
    

    ##### Write result out to file #####
    if not result is None:
        # |Result is none when distribution of lambda is calculated
        fileExists = True if os.path.isfile(intermediate_results_path+processID+"_results.tsv") else False
        with open(intermediate_results_path+processID+"_results.tsv", "a") as resultFile:
            if not fileExists:
                resultFile.write('\t'.join(['parameter', 'lambda_true', 'lambda_predicted\n']))
            for param_ in target_param:
                resultFile.write( '\t'.join([param_, str(active_heuristic_params[param_]), str(result[0][param_])])+'\n' )
    

    ##### Delete intermediate files #####
    intermediate_results_files = [intermediate_results_path+fl_name for fl_name in listdir(intermediate_results_path)]  # Rerun it to include files created by this script
    files2delete = [currFile for currFile in intermediate_results_files if any(kwrd in currFile for kwrd in keywords2delete)]
    for delFile in files2delete:
        try:
            os.remove(delFile)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File '{delFile}' not found.")
        except OSError as e:
            raise OSError(f"Error: {e}")



    env.close()

    # |Print the execution time
    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {math.floor(execution_time/60)} minutes and {execution_time%60:.1f} seconds")