torch = require 'torch'
nn = require 'nn'
gp = require 'gnuplot'

function build_net(w)
   local net = nn.Sequential()
   net:add(nn.Linear(1,w))
   net:add(nn.Tanh())
   net:add(nn.Linear(w,1))
   return net
end

function disp_train(net, func, len, disp_every)
   local dataset = {}
   for i=1, len do
      local p = torch.rand(1) * 16 - 8
      local t = p:clone():apply(func)
      dataset[i] = {p, t}
   end
   function dataset.size()
      return len
   end

   local errs = {}
   local cr = nn.MSECriterion()
   local trainer = nn.StochasticGradient(net, cr)
   trainer.learningRate = 0.01
   trainer.maxIteration = 10
   trainer.hookIteration = function(s, i, e)
      errs[i] = e
   end

   for i=1,10 do
      trainer.learningRate = 0.01 / (1 + i /2 )
      errs = {}
      trainer:train(dataset)

      gp.figure(i)
      gp.raw("set multiplot layout 2,1")
      gp.plot("errs", torch.Tensor(errs), '-')
      disp(net)
      gp.raw("unset multiplot")
   end
end

function f(x)
   return 0.5 * (math.sin(3 * x) + math.cos(2 * x + 1))
end

function disp(net)
   xx = torch.linspace(-10,10,100)
   yy1 = xx:clone()
   yy2 = xx:clone()

   yy1 = yy1:apply(function(x) return net:forward(torch.Tensor({x}))[1] end)
   yy2 = yy2:apply(f)

   gp.plot({"func", xx,yy1}, {"tgt", xx, yy2})
end


-- for w=1,10 do
--    local nt = build_net(w)
--    gp.figure(w)
--    disp_train(nt, f, 200)
-- end

local nt = build_net(10)
disp_train(nt, f, 10000, 200)
