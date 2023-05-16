from experiment import build_experiment, initialize_experiment, N_BATCH
from MOO import search_space, optimization_config, Models
from ax.service.utils.report_utils import exp_to_df
from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_PAREGO
from ax.modelbridge.modelbridge_utils import observed_hypervolume
import numpy as np
import matplotlib.pyplot as plt
from ax import Data

# Start Sobol
ehvi_experiment = build_experiment("ehvi_pareto", search_space=search_space,
                                   optimization_config=optimization_config)
# print(sobol_experiment.__class__)
ehvi_data = initialize_experiment(ehvi_experiment)
print("ehvi data is {}".format(ehvi_data))

# Define the model of Sobol
ehvi_model = None
ehvi_hv_list = []


# Define each iteration
def ehvi_iteration(ehvi_data=ehvi_data):
    ehvi_model = get_MOO_EHVI(
        experiment=ehvi_experiment,
        data=ehvi_data,
    )

    generator_run = ehvi_model.gen(1)
    trial = ehvi_experiment.new_trial(generator_run=generator_run)
    trial.run()
    new_ehvi_data = Data.from_multiple_data([ehvi_data, trial.fetch_data()])

    exp_df = exp_to_df(ehvi_experiment)
    outcomes = np.array(exp_df[['a', 'b']], dtype=np.double)

    try:
        hv = observed_hypervolume(modelbridge=ehvi_model)
    except:
        hv = 0
        print("Failed to compute hv")

    return hv, new_ehvi_data


# iteration by iteration
def iteration(max_iter=N_BATCH):
    new_ehvi_data = None
    for i in range(max_iter):
        if i == 0:
            hv, new_ehvi_data = ehvi_iteration(ehvi_data)
        else:
            hv, new_ehvi_data = ehvi_iteration(new_ehvi_data)
        print("Iteration: {}, HV: {}".format(i, hv))
        ehvi_hv_list.append(hv)


if __name__ == "__main__":
    iteration()
    ehvi_outcomes = np.array(exp_to_df(ehvi_experiment)[['a', 'b']],
                              dtype=np.double)
    print(ehvi_outcomes.shape)

    plt.plot(ehvi_hv_list)
    plt.show()
