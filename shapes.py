import torch
from torch import nn


class Shape(nn.Module):
    def __init__(self, kernel_size, initial_parameters, num_shapes):
        super().__init__()
        x = torch.arange(kernel_size)
        y = torch.arange(kernel_size)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.kernel_size = kernel_size
        self.register_buffer("grid_x", grid_x.flatten().float())
        self.register_buffer("grid_y", grid_y.flatten().float())

        params = torch.tensor(initial_parameters).unsqueeze(-1).repeat(1, num_shapes)
        self.shape_params = nn.Parameter(params, requires_grad=True)

    def forward(self, kernel_centers, kernel_polarities, normalize=True):
        kernel_x = kernel_centers[:, 0]
        kernel_y = kernel_centers[:, 1]

        dx = kernel_x[None, :] - self.grid_x[:, None]
        dy = kernel_y[None, :] - self.grid_y[:, None]

        W = self.shape_function(dx ** 2 + dy ** 2)
        if normalize:
            W = W / W.norm(dim=0, keepdim=True)

        return W * kernel_polarities

    def shape_function(self, rr):
        raise NotImplementedError


class DifferenceOfGaussianShape(Shape):
    def __init__(self, kernel_size, DoG_version, num_shapes=1):
        self.DoG_version = DoG_version
        if self.DoG_version == 'color':
            init_params = [-4.0, -7.0, 2.0]
        elif self.DoG_version == 'claude':
            init_params = [0.0,0.0,0.0]
        else:
            init_params = [-3,-0.9,2]
        super().__init__(kernel_size, init_params, num_shapes) #What the parameters are initialized to

    def shape_function(self, rr):
        logA, logB, logitC = self.shape_params

        if self.DoG_version == 'original': 
            a = logA.exp()
            b = logB.exp()
            a = a + b  # make the center smaller than the surround
            max_r = self.kernel_size // 4
            logitlogC = self.shape_params[2]
            logC = - (a - b) * max_r ** 2 * logitlogC.sigmoid()
            c = logC.exp()
            
        elif self.DoG_version == 'sigmoid_c':
            a = logA.exp()
            b = logB.exp()
            a = a + b  # make the center smaller than the surround
            c = logitC.sigmoid() #to keep it within (0, 1)
        

        self.a, self.b, self.c = a.detach(), b.detach(), c.detach()

        return torch.exp(-a * rr) - c*torch.exp(-b * rr)



def get_shape_module(type):
    return {
        'difference-of-gaussian': DifferenceOfGaussianShape,
    }[type]

