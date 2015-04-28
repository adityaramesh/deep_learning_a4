require 'torch'
require 'nngraph'

x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Linear(3, 3)()
l1 = nn.CMulTable()({x2, x3})
l2 = nn.CAddTable()({x1, l1})
m = nn.gModule({x1, x2, x3}, {l2})

-- Initialize the parameters for testing.
params = m:parameters()
-- Set the weight matrix of the linear module to the 3x3 identity.
params[1]:copy(torch.eye(3))
-- Set the bias of the linear module to the zero vector.
params[2]:fill(0)

-- Run forward and backward propagation for the sake of updating the
-- annotations.
t1 = torch.Tensor({1, 1, 1})
t2 = torch.Tensor({2, 2, 2})
t3 = torch.Tensor({3, 3, 3})
output = m:updateOutput({t1, t2, t3})
m:updateGradInput({t1, t2, t3}, torch.rand(3))

exp_output = torch.Tensor({7, 7, 7})
assert(torch.eq(output, exp_output):sum() == 3)

graph.dot(m.fg, 'Forward Graph', 'forward_graph')
