import torch

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing theh output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # uncomment to run on GPU

# N is batch size: D_in in input dimension;
# H is hidden dimension: D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and ouputs
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these tensors during teh backward pass
x = torch.randn(N, D_in, device=device, dtype=dtype);
y = torch.randn(N, D_out, device=device, dtype=dtype);

# Create random Tensors for weights
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True);
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True);

learning_rate = 1e-6
for t in range(500):
    # to apply our Function, we use Function.appy method
    relu = MyReLU.apply

    # Forward pass: compute predicted y. using custom autograd p[eration
    y_pred = relu(x.mm(w1)).mm(w2)

    # compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if (t % 100 == 99):
        print(t, loss.item())

    # Use autograd to compute the backward pass
    # Afterwards, w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively
    loss.backward()

    # Manually update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating the weights
        w1.grad.zero_()
        w2.grad.zero_()

