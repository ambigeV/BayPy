import math
import torch
import gpytorch
import matplotlib.pyplot as plt
from gpytorch.constraints import Positive

# prepare input data
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * 2 * math.pi) + torch.randn(train_x.size()) + math.sqrt(0.04)

# prepare the model

class MySimpleKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # apply lengthscale
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        # calculate the distance between inputs
        diff = self.covar_dist(x1_, x2_, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        # return sinc(diff) = sin(diff) / diff
        return torch.sin(diff).div(diff)


class MyKernel(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # register the raw parameter
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()

        # register the constraint
        self.register_constraint("raw_length", length_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if length_prior is not None:
            self.register_prior(
                "length_prior",
                length_prior,
                lambda m: m.length,
                lambda m, v: m._set_length(v),
            )
    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, **params):
        x1_ = x1.div(self.length)
        x2_ = x2.div(self.length)
        diff = self.covar_dist(x1_, x2_, **params)
        diff.where(diff == 0, torch.as_tensor(1e-20))
        return torch.sin(diff).div(diff)
# GP model


class MyGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = MyKernel()
        self.covar_module = MySimpleKernel()

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

        # print("Iter:{}/{}, loss:{}, length_scale:{}, noise:{}".format(
        #     i, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise_covar.noise.item()
        # ))

        optimizer.step()

# evaluation step --> returns p(f* | x*, X, y)
# model(test_x) returns model distribution
# likelihood(model(test_x)) returns output distribution


def evaluation(model, likelihood, test_x):
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

    model.eval()
    likelihood.eval()
    observed_pred = evaluation(model, likelihood, test_x)
    plot(observed_pred, test_x)
