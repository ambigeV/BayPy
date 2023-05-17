from ax import Experiment
from MOO import Models
from ax.runners.synthetic import SyntheticRunner

# initialize variables
N_INIT = 6
N_BATCH = 25


# experiment builder
def build_experiment(exp_name,
                     search_space,
                     optimization_config):
    experiment = Experiment(
        name=exp_name,
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment


# initialize with sobol samples
def initialize_experiment(experiment):
    # print(experiment.__class__)
    sobol = Models.SOBOL(search_space=experiment.search_space, seed=1234)

    for _ in range(N_INIT):
        experiment.new_trial(sobol.gen(1)).run()

    return experiment.fetch_data()

