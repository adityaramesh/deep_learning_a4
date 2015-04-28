stringx = require('pl.stringx')
require 'io'

opt = {
	task = "evaluate",
	model = "queryable_char_model",
	version = "best_train",
	device = 1
}
require "main"
clear_model_context()

function readline()
	local line = io.read("*line")
	if line == nil then error({code="EOF"}) end
	
	local index = 0
	if ptb.vocab_map[line] ~= nil then
		return ptb.vocab_map[line]
	end

	print("Unknown token: \"" .. line .. "\".")
	error({code="unknown"})
end

print("Please type a single character, <unk>, or <eos>.")
io.write("OK GO\n")
io.flush()

while true do
	local ok, index = pcall(readline)
	if not ok then
		if index.code == "EOF" then
			break
		elseif index.code == "unknown" then
			print("Line contains unknown token; please try again.")
		else
			print("Unknown error occurred; please try again.")
		end
	else
		preds = predict_next_char(index)
		for i = 1, preds:size(1) do
			io.write(preds[i] .. " ")
		end

		--local prob, index = preds:min(1)
		--index = index[1]
		--print(ptb.index_map[index])
		io.write('\n')
		io.flush()
	end
end
