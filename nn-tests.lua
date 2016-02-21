torch = require 'torch'
nn = require 'nn'
gp = require 'gnuplot'

function build_net1()
   local l1 = nn.Linear(2,1)
   local net = nn.Sequential()
   net:add(l1)
   return net
end

function learn_step(model, func, cr)
  -- random sample
  local input= torch.rand(2);     -- normally distributed example in 2d
  local output= torch.Tensor(1);

  output[1] = func(input[1], input[2])

  -- feed it to the neural network and the criterion
  local o = model:forward(input)
--  cr:forward(o, output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  model:zeroGradParameters()
  -- (2) accumulate gradients
  model:backward(input, cr:backward(model.output, output))
  -- (3) update parameters with a 0.01 learning rate
  model:updateParameters(0.01)
  return (o[1] - output[1])*(o[1] - output[1])
end

function disp_train(net, func, len)
   local cr = nn.MSECriterion()
   local errs = torch.Tensor(len)

   for i=1,len do
      local e = learn_step(net, func, cr)
      errs[i] = e
   end
   gp.plot("errs", errs, '+')
end

n1 = build_net1()
disp_train(n1, function(x,y) return 3* x - 10*y + 3 end, 2000)
