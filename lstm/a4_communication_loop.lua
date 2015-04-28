stringx = require('pl.stringx')
require 'io'

opt = {
	task = "evaluate",
	model = "queryable_word_model",
	version = "best_train",
	device = 1
}
require "main"

function readline()
	local line = io.read("*line")
	if line == nil then error({code="EOF"}) end
	line = stringx.split(line)
	if tonumber(line[1]) == nil then error({code="init"}) end

	local count = tonumber(line[1])
	table.remove(line, 1)

	indices = {}
	for i = 2, #line do
		if ptb.vocab_map[line[i]] ~= nil then
			indices[i - 1] = ptb.vocab_map[line[i]]
		else
			indices[i - 1] = ptb.vocab_map["<unk>"]
		end
	end
	return count, line, indices
end

while true do
	print("Query: len word1 word2 etc")
	local ok, count, line, indices = pcall(readline)
	if not ok then
		if line.code == "EOF" then
			break
		elseif line.code == "init" then
			print("Line must start with a number.")
		else
			print("Unknown error occurred; please try again.")
		end
	else
		process_new_sentence(indices)
		for i = 1, count do
			preds = torch.exp(predict_next_word(indices))
			local prob, index = preds:max(1)
			prob = prob[1]
			index = index[1]
			table.insert(indices, index)
			table.insert(line, ptb.index_map[index])
		end
		print(table.concat(line, " "))
		io.write('\n')
	end
end
