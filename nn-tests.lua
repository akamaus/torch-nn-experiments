torch = require 'torch'
nn = require 'nn'
gp = require 'gnuplot'

require 'mnist_reader.lua'

function build_net(i, layers)
   local net = nn.Sequential()

   local last = i
   for _,l_size in pairs(layers) do
      net:add(nn.Linear(last,l_size))
--      net:add(nn.Dropout(0.5))
      net:add(nn.Tanh())

      last = l_size
   end

   net:add(nn.Linear(last,10))
   return net
end


function maxi(t)
   local m = -100
   local n = -1
   for i=1,t:size(1) do
      if t[i] > m then
         m = t[i]
         n = i
      end
   end
   return n
end

function fitness(net, dataset_raw)
   local N = 0
   local ok = 0
   for i,v in pairs(dataset_raw) do
      local lbl = torch.zeros(10)
      local res = net:forward(v["image"]:resize(28*28))
      if maxi(res) - 1 == v["label"] then
         ok = ok + 1
      end
      N = N + 1
   end
   return ok / N
end

function disp_train(net, dataset_raw, num_epochs)
   local dataset = {}
   local N = 0
   for i,v in pairs(dataset_raw) do
      local lbl = torch.zeros(10)
      lbl[ v["label"] + 1 ] = 1
      dataset[i] = {v["image"]:resize(28*28), lbl}
      N = N + 1
   end
   function dataset.size()
      return N
   end

   local errs = {}
   local cr = nn.MSECriterion()
   local trainer = nn.StochasticGradient(net, cr)
   trainer.learningRate = 0.01
   trainer.maxIteration = 5
   trainer.hookIteration = function(s, i, e)
      errs[i] = e
   end

   for i=1,num_epochs do
      print("epoch: ", i, " train: ", fitness(net, dataset_raw), " validation: ", fitness(net, test_data))
      if i >= 10 then
         trainer.learningRate = 0.01 / (1 + (i - 10) * 0.2)
      end
      errs = {}
      trainer:train(dataset)

      gp.figure(i)
--      gp.raw("set multiplot layout 2,1")
--      gp.plot("errs", torch.Tensor(errs), '-')
--      disp2d(net, func)
--      gp.raw("unset multiplot")
   end
end

net = build_net(28*28, {50, 25})
--disp2d(net, f)
train_data = read_mnist("train")
test_data = read_mnist("t10k")


disp_train(net, train_data, 100)
