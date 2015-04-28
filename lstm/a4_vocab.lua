require 'torch'
require 'base'

params = {model_granularity = "character"}
ptb = require('data')

traindata = ptb.traindataset(20)
for k,v in pairs(ptb.vocab_map) do
  print(k .. " " .. v)
end
