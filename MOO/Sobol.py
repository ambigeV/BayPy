from experiment import build_experiment, initialize_experiment, N_BATCH
from MOO import search_space, optimization_config, Models
from ax.service.utils.report_utils import exp_to_df
from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_PAREGO
from ax.modelbridge.modelbridge_utils import observed_hypervolume
import numpy as np
import matplotlib.pyplot as plt

# Start Sobol
sobol_experiment = build_experiment("sobol_pareto", search_space=search_space,
                                    optimization_config=optimization_config)
# print(sobol_experiment.__class__)
sobol_data = initialize_experiment(sobol_experiment)

# Define the model of Sobol
sobol_model = Models.SOBOL(
    experiment=sobol_experiment,
    data=sobol_data
)

sobol_hv_list = []


# Define each iteration
def sobol_iteration():
    generator_run = sobol_model.gen(1)
    trial = sobol_experiment.new_trial(generator_run=generator_run)
    trial.run()

    exp_df = exp_to_df(sobol_experiment)
    outcomes = np.array(exp_df[['a', 'b']], dtype=np.double)

    dummy_model = get_MOO_EHVI(
        experiment=sobol_experiment,
        data=sobol_experiment.fetch_data(),
    )

    try:
        hv = observed_hypervolume(modelbridge=dummy_model)
    except:
        hv = 0
        print("Failed to compute hv")

    return hv


# iteration by iteration
def iteration(max_iter=N_BATCH):
    for i in range(max_iter):
        hv = sobol_iteration()
        print("Iteration: {}, HV: {}".format(i, hv))
        sobol_hv_list.append(hv)


if __name__ == "__main__":
    iteration()
    sobol_outcomes = np.array(exp_to_df(sobol_experiment)[['a', 'b']],
                              dtype=np.double)
    print(sobol_outcomes.shape)

    plt.plot(sobol_hv_list)
    plt.show()
