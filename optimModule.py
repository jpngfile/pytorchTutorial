import torch

# N is batch size: D_in in input dimension;
# H is hidden dimension: D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and ouputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# the nn package also contains d efinitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

# use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # compute and print loss.
    loss = loss_fn(y_pred, y)
    if (t % 100 == 99):
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers (i.e., not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model.
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()

