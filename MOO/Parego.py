from experiment import build_experiment, initialize_experiment, N_BATCH
from MOO import search_space, optimization_config, Models
from ax.service.utils.report_utils import exp_to_df
from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_PAREGO
from ax.modelbridge.modelbridge_utils import observed_hypervolume
import numpy as np
import matplotlib.pyplot as plt
from ax import Data

# Start Parego
parego_experiment = build_experiment("parego_pareto", search_space=search_space,
                                   optimization_config=optimization_config)

parego_data = initialize_experiment(parego_experiment)

# Define the model of Parego
parego_model = None
parego_hv_list = []


# Define each iteration
def ehvi_iteration(parego_data=parego_data):
    parego_model = get_MOO_PAREGO(
        experiment=parego_experiment,
        data=parego_data,
    )

    generator_run = parego_model.gen(1)
    trial = parego_experiment.new_trial(generator_run=generator_run)
    trial.run()
    new_parego_data = Data.from_multiple_data([parego_data, trial.fetch_data()])

    exp_df = exp_to_df(parego_experiment)
    outcomes = np.array(exp_df[['a', 'b']], dtype=np.double)

    try:
        hv = observed_hypervolume(modelbridge=parego_model)
    except:
        hv = 0
        print("Failed to compute hv")

    return hv, new_parego_data


# iteration by iteration
def iteration(max_iter=N_BATCH):
    new_parego_data = None
    for i in range(max_iter):
        if i == 0:
            hv, new_parego_data = ehvi_iteration(parego_data)
        else:
            hv, new_parego_data = ehvi_iteration(new_parego_data)
        print("Iteration: {}, HV: {}".format(i, hv))
        parego_hv_list.append(hv)


if __name__ == "__main__":
    iteration()
    ehvi_outcomes = np.array(exp_to_df(parego_experiment)[['a', 'b']],
                              dtype=np.double)
    print(ehvi_outcomes.shape)

    plt.plot(parego_hv_list)
    plt.show()
