import math
import torch
import gpytorch
import matplotlib.pyplot as plt

# prepare input data
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * 2 * math.pi) + torch.randn(train_x.size()) + math.sqrt(0.04)

# prepare the model


class MyKernel(gpytorch.kernels.Kernel):
    is_stationary = True

    def forward(self, x1, x2, **params):
        diff = self.covar_dist(x1, x2, **params)
        diff.where(diff == 0, torch.as_tensor(1e-20))
        return torch.sin(diff).div(diff)
# GP model


class MyGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MyGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = MyKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(model, likelihood, training_iter):
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # define the loss for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # training iter
    for i in range(training_iter):
        optimizer.zero_grad()

        output = model(train_x)

        loss = -mll(output, train_y)
        loss.backward()

        print("Iter:{}/{}, loss:{}, length_scale:{}, noise:{}".format(
            i, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise_covar.noise.item()
        ))

        optimizer.step()

# evaluation step --> returns p(f* | x*, X, y)
# model(test_x) returns model distribution
# likelihood(model(test_x)) returns output distribution


def evaluation(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

        return observed_pred


def plot(observed_pred, test_x):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    lower, upper = observed_pred.confidence_region()
    ax.plot(train_x.numpy(), train_y.numpy(), 'g*')
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

    plt.show()


if __name__ == "__main__":
    # fetch likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MyGPModel(train_x, train_y, likelihood)

    # training loop
    training_iter = 50
    model.train()
    likelihood.train()

    # testing data
    test_x = torch.linspace(0, 1.5, 202)

    train(model, likelihood, training_iter)
    observed_pred = evaluation(model, likelihood, test_x)
    plot(observed_pred, test_x)
