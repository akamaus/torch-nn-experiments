
function read_mnist(file)
   -- pixels
   local tf = torch.DiskFile("mnist_data/" .. file .. "-images-idx3-ubyte", "r"):binary():bigEndianEncoding()
   local magic = tf:readInt()
   assert(magic == 2051, "unknown format " .. tostring(magic))

   local n = tf:readInt()
   local rows = tf:readInt()
   local cols = tf:readInt()

   local images = {}

   for i=1,n do
      local p = tf:readByte(rows * cols)
      local t = torch.Tensor(rows,cols)
      t:storage():copy(p)
      images[i] = {}
      images[i]["image"] = t / 255
   end
   tf:close()

   -- labels
   local lf = torch.DiskFile("mnist_data/" .. file .. "-labels-idx1-ubyte", "r"):binary():bigEndianEncoding()
   local magic = lf:readInt()
   assert(magic == 2049, "unknown format " .. tostring(magic))

   local nl = lf:readInt()
   assert(n == nl, "every labelled")

   for i=1,n do
      local lbl = lf:readByte()
      images[i]["label"] = lbl
   end
   lf:close()

   return images
end
