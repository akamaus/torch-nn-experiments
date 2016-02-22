torch = require 'torch'
nn = require 'nn'
gp = require 'gnuplot'

function build_net(w)
   local net = nn.Sequential()
   net:add(nn.Linear(1,w))
   net:add(nn.Tanh())
   net:add(nn.Linear(w,w))
   net:add(nn.Tanh())
   net:add(nn.Linear(w,1))
   return net
end

function learn_step(model, func, cr)
  -- random sample
  local input= torch.rand(1) * 10 - 5;     -- normally distributed example in 2d
  local output= torch.Tensor(1);

  output[1] = func(input[1])

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

   local ch = 1000
   local ch_e = errs:chunk(ch)

   local es = map(function(cs) return cs:sum() / cs:size(1) end, ch_e)

   gp.plot("errs", torch.Tensor(es), '-')
end

function f(x)
   return -math.sin(3*x) + math.cos(2*x + 1)
end

function map(func, tbl)
     local newtbl = {}
     for i,v in pairs(tbl) do
         newtbl[i] = func(v)
     end
     return newtbl
end

for w=1,10 do
   local nt = build_net(w)

   gp.figure(w)
   gp.raw("set multiplot layout 2,1")
   disp_train(nt, f, 50000)

   xx = torch.linspace(-5,5,100)
   yy1 = xx:clone()
   yy2 = xx:clone()

   yy1 = yy1:apply(function(x) return nt:forward(torch.Tensor({x}))[1] end)
   yy2 = yy2:apply(f)

   gp.plot({"func", xx,yy1}, {"tgt", xx, yy2})
   gp.raw("unset multiplot")
end
