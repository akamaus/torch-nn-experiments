torch = require 'torch'
nn = require 'nn'
gp = require 'gnuplot'

function build_net(i, w, k)
   local net = nn.Sequential()

   net:add(nn.Linear(i,w))
   net:add(nn.Tanh())

   for i=2,k do
      net:add(nn.Linear(w,w))
      net:add(nn.Tanh())
   end

   net:add(nn.Linear(w,1))
   return net
end

function disp_train(net, func, batch, num_epochs)
   local dataset = {}
   for i=1, batch do
      local p = torch.rand(2) * 4 - 2
      local t = torch.Tensor({func(p[1],p[2])})
      dataset[i] = {p, t}
   end
   function dataset.size()
      return batch
   end

   local errs = {}
   local cr = nn.MSECriterion()
   local trainer = nn.StochasticGradient(net, cr)
   trainer.learningRate = 0.01
   trainer.maxIteration = 10
   trainer.hookIteration = function(s, i, e)
      errs[i] = e
   end

   for i=1,num_epochs do
--      trainer.learningRate = 0.01 / (1 + i /2 )
      errs = {}
      trainer:train(dataset)

      gp.figure(i)
      gp.raw("set multiplot layout 2,1")
      gp.plot("errs", torch.Tensor(errs), '-')
      disp2d(net, func)
      gp.raw("unset multiplot")
   end
end

function f(x,y)
   x = x * 1.5
   return math.sin(x*x + y*y) + math.cos(x + y)
end

function disp2d(net,f)
   local n = 20
   local h = 2
   local tgt = torch.Tensor(n,n)
   local res = torch.Tensor(n,n)
   for i=1,n do
      for j=1,n do
         local x = i / (n / h / 2) - 2
         local y = j / (n / h / 2) - 2
         tgt[i][j] = f(x,y)
         res[i][j] = net:forward(torch.Tensor({x,y}))[1]
      end
   end

   gp.splot({"tgt", tgt}, {"net", res})
end


-- for w=1,10 do
--    local nt = build_net(w)
--    gp.figure(w)
--    disp_train(nt, f, 200)
-- end

local net = build_net(2, 6, 2)
--disp2d(net, f)
disp_train(net, f, 10000, 50)
