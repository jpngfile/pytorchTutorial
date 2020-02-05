import torch

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
    # Forward pass: compute predicted y. We no longer need to keep references 
    # to intermediate values since the backward pass is handled automatically.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # compute and print loss
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss
    loss = (y_pred - y).pow(2).sum()
    if (t % 100 == 99):
        print(t, loss.item())

    # Use autograd to compute the backward pass
    # Afterwards, w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively
    loss.backward()

    # Manually update weights using gradient descent
    # Wrap in torch.no_grad() because weights have requires_grad=True but
    # we don't need to track this in autograd
    # Alternatively, we can operate on weight.data and weight.grad.data
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating the weights
        w1.grad.zero_()
        w2.grad.zero_()

