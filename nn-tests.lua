torch = require 'torch'
nn = require 'nn'
gp = require 'gnuplot'

l1 = nn.Linear(2,1)
net = nn.Sequential()
cr = nn.MSECriterion()

net:add(l1)

function learn_step(model)
  -- random sample
  local input= torch.rand(2);     -- normally distributed example in 2d
  local output= torch.Tensor(1);

  output[1] = 2 * input[1] + 3 * input[2] - 5

  -- feed it to the neural network and the criterion
  local o = net:forward(input)
  cr:forward(o, output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  model:zeroGradParameters()
  -- (2) accumulate gradients
  model:backward(input, cr:backward(net.output, output))
  -- (3) update parameters with a 0.01 learning rate
  model:updateParameters(0.01)
  return (o[1] - output[1])*(o[1] - output[1])
end

errs = torch.Tensor(10000)
for i=1,1000 do
   local e = learn_step(net)
   errs[i] = e
   print(e)
end
gp.plot("errs", errs, '+')
