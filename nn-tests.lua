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

function batch_learn(model, input, target, cr)
   local err = 0
   model:zeroGradParameters()
   for k=1,input:size(1) do
      -- feed it to the neural network and the criterion
      local inp = input:narrow(1,k,1)
      local tgt = target:narrow(1,k,1)

      local out = model:forward(inp)
      local e = cr:forward(out, tgt)
      local grad = cr:backward(model.output, tgt)
      model:backward(inp, grad)

      err = err + e
   end
   model:updateParameters(0.0001)

   return err / input:size(1)
end

function disp_train(net, func, len, disp_every)
   local cr = nn.MSECriterion()
   local data_points = torch.rand(len) * 10 - 5
   local tgts = data_points:clone():apply(func)

   local errs = torch.Tensor(len)

   for i=1,len do
      local e = batch_learn(net, data_points, tgts, cr)
      errs[i] = e
      if disp_every and i % disp_every == 0 then
         gp.figure(i)
         disp(net, i)
      end
   end
   gp.raw("set multiplot layout 2,1")
   gp.plot("errs", errs, '-')
   disp(net)
   gp.raw("unset multiplot")
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
