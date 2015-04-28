--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

ok, cunn = pcall(require, 'fbcunn')
if not ok then
	ok,cunn = pcall(require,'cunn')
	if ok then
		print("warning: fbcunn not found. Falling back to cunn") 
		LookupTable = nn.LookupTable
	else
		print("Could not find cunn or fbcunn. Either is required")
		os.exit()
	end
else
	deviceParams = cutorch.getDeviceProperties(1)
	cudaComputeCapability = deviceParams.major + deviceParams.minor/10
	LookupTable = nn.LookupTable
end

-- Train 1 day and gives 82 perplexity.
--[[
params = {
	batch_size=20,
	seq_length=35,
	layers=2,
	decay=1.15,
	rnn_size=1500,
	dropout=0.65,
	init_weight=0.04,
	lr=1,
	vocab_size=10000,
	epochs_to_lr_decay=14,
	max_epochs=55,
	max_grad_norm=10,
	use_default_opt=true
}
]]--

-- Trains 1h and gives test 115 perplexity.
-- params = {
-- 	batch_size=20,
-- 	seq_length=20,
-- 	layers=2,
-- 	decay=2,
-- 	rnn_size=200,
-- 	dropout=0,
-- 	init_weight=0.1,
-- 	lr=1,
-- 	vocab_size=10000,
-- 	epochs_to_lr_decay=4,
-- 	max_epochs=13,
-- 	max_grad_norm=5,
-- 	use_default_opt=true,
-- 	make_queryable_model=false,
-- 	model_granularity="word"
-- }

-- Parameters for sentence query model.
-- params = {
-- 	batch_size=20,
-- 	seq_length=20,
-- 	layers=2,
-- 	decay=2,
-- 	rnn_size=200,
-- 	dropout=0,
-- 	init_weight=0.1,
-- 	lr=1,
-- 	vocab_size=10000,
-- 	epochs_to_lr_decay=4,
-- 	max_epochs=13,
-- 	max_grad_norm=5,
-- 	use_default_opt=true,
-- 	make_queryable_model=true,
-- 	model_granularity="word"
-- }

-- Parameters for character query model.
params = {
	batch_size=20,
	seq_length=100,
	layers=1,
	decay=2,
	rnn_size=400,
	dropout=0,
	init_weight=0.1,
	lr=1,
	vocab_size=50,
	epochs_to_lr_decay=4,
	max_epochs=13,
	max_grad_norm=5,
	use_default_opt=true,
	make_queryable_model=true,
	model_granularity="character"
}

require('nngraph')
require('base')
require('xlua')
ptb = require('data')

-- Ripped out the serialization functionality from run_model.lua and isolated it
-- so that we can quickly integrate it with this script.
require('serialization')

-- TODO: include this in the repository for submission.
--package.path = package.path .. ";./../../torch_utilities/?.lua"
--require "sopt"

model = {}

function transfer_data(x)
	return x:cuda()
end

function perplexity(mean_nll)
	if model.granularity == "word" then
		return torch.exp(mean_nll)
	else
		return torch.exp(5.6 * mean_nll)
	end
end

function lstm(i, prev_c, prev_h)
	local function new_input_sum()
		local i2h = nn.Linear(params.rnn_size, params.rnn_size)
		local h2h = nn.Linear(params.rnn_size, params.rnn_size)
		return nn.CAddTable()({i2h(i), h2h(prev_h)})
	end
	local in_gate     = nn.Sigmoid()(new_input_sum())
	local forget_gate = nn.Sigmoid()(new_input_sum())
	local in_gate2    = nn.Tanh()(new_input_sum())
	local next_c      = nn.CAddTable()({
		nn.CMulTable()({forget_gate, prev_c}),
		nn.CMulTable()({in_gate, in_gate2})
	})
	local out_gate = nn.Sigmoid()(new_input_sum())
	local next_h   = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
	return next_c, next_h
end

function create_network()
	local x      = nn.Identity()()
	local y      = nn.Identity()()
	local prev_s = nn.Identity()()
	local i      = {[0] = LookupTable(params.vocab_size, params.rnn_size)(x)}
	local next_s = {}
	local split  = {prev_s:split(2 * params.layers)}
	for layer_idx = 1, params.layers do
		local prev_c         = split[2 * layer_idx - 1]
		local prev_h         = split[2 * layer_idx]
		local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
		local next_c, next_h = lstm(dropped, prev_c, prev_h)
		table.insert(next_s, next_c)
		table.insert(next_s, next_h)
		i[layer_idx] = next_h
	end
	local h2y     = nn.Linear(params.rnn_size, params.vocab_size)
	local dropped = nn.Dropout(params.dropout)(i[params.layers])
	local pred    = nn.LogSoftMax()(h2y(dropped))
	local err     = nn.ClassNLLCriterion()({pred, y})

	local module = {}
	if not params.make_queryable_model then
		module = nn.gModule({x, y, prev_s}, {err, nn.Identity()(next_s)})
	else
		module = nn.gModule({x, y, prev_s}, {err, nn.Identity()(next_s),
			nn.Identity()(pred)})
	end

	module:getParameters():uniform(-params.init_weight, params.init_weight)
	return transfer_data(module)
end

function setup()
	print("Creating a RNN LSTM network.")
	if not model_info_loaded() then
		local core_network = create_network()
		paramx, paramdx = core_network:getParameters()
		model.s = {}
		model.ds = {}
		model.start_s = {}

		if params.make_queryable_model then
			model.dummy_output_grads = transfer_data(
				torch.zeros(params.batch_size, params.vocab_size))
		end

		for j = 0, params.seq_length do
			model.s[j] = {}
			for d = 1, 2 * params.layers do
				model.s[j][d] = transfer_data(
					torch.zeros(params.batch_size, params.rnn_size))
			end
		end
		for d = 1, 2 * params.layers do
			model.start_s[d] = transfer_data(
				torch.zeros(params.batch_size, params.rnn_size))
			model.ds[d] = transfer_data(
				torch.zeros(params.batch_size, params.rnn_size))
		end
		model.core_network = core_network
		model.rnns = g_cloneManyTimes(core_network, params.seq_length)
		model.norm_dw = 0
		model.err = transfer_data(torch.zeros(params.seq_length))
		set_model_info(core_network, model)
	else
		local core_network, model_buffers = get_model_info()
		paramx, paramdx = core_network:getParameters()
		model = model_buffers
		model.rnns = g_cloneManyTimes(core_network, params.seq_length)
	end
end

function reset_state(state)
	state.pos = 1
	if model ~= nil and model.start_s ~= nil then
		for d = 1, 2 * params.layers do
			model.start_s[d]:zero()
		end
	end
end

function reset_ds()
	for d = 1, #model.ds do
		model.ds[d]:zero()
	end
end

function clear_model_context()
	dummy_state = {}
	reset_state(dummy_state)
	reset_ds()

	for j = 0, params.seq_length do
		for d = 1, 2 * params.layers do
			model.s[j][d]:zero()
		end
	end
end

function fp(state)
	g_replace_table(model.s[0], model.start_s)
	if state.pos + params.seq_length > state.data:size(1) then
		reset_state(state)
	end
	for i = 1, params.seq_length do
		local x = state.data[state.pos]
		local y = state.data[state.pos + 1]
		local s = model.s[i - 1]
		model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
		state.pos = state.pos + 1
	end
	g_replace_table(model.start_s, model.s[params.seq_length])
	return model.err:mean()
end

function bp(state)
	paramdx:zero()
	reset_ds()
	for i = params.seq_length, 1, -1 do
		state.pos = state.pos - 1
		local x = state.data[state.pos]
		local y = state.data[state.pos + 1]
		local s = model.s[i - 1]
		local derr = transfer_data(torch.ones(1))
		local tmp = {}

		if not params.make_queryable_model then
			tmp = model.rnns[i]:backward({x, y, s}, {derr, model.ds})[3]
		else
			tmp = model.rnns[i]:backward({x, y, s}, {derr, model.ds,
				model.dummy_output_grads})[3]
		end

		g_replace_table(model.ds, tmp)
		cutorch.synchronize()
	end

	state.pos = state.pos + params.seq_length
	model.norm_dw = paramdx:norm()

	--if params.use_default_opt then
		if model.norm_dw > params.max_grad_norm then
			local shrink_factor = params.max_grad_norm / model.norm_dw
			paramdx:mul(shrink_factor)
		end
	--end

	if params.use_default_opt then
		paramx:add(paramdx:mul(-params.lr))
	end
end

function run_valid()
	reset_state(state_valid)
	g_disable_dropout(model.rnns)
	local len = (state_valid.data:size(1) - 1) / (params.seq_length)
	local perp = 0
	for i = 1, len do
		perp = perp + fp(state_valid)
	end

	perp = perplexity(perp / len)
	save_test_progress(perp)
	print("Validation set perplexity: " .. g_f3(perp))
	g_enable_dropout(model.rnns)
end

function run_test()
	reset_state(state_test)
	g_disable_dropout(model.rnns)
	local perp = 0
	local len = state_test.data:size(1)
	g_replace_table(model.s[0], model.start_s)
	for i = 1, (len - 1) do
		local x = state_test.data[i]
		local y = state_test.data[i + 1]
		perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
		perp = perp + perp_tmp[1]
		g_replace_table(model.s[0], model.s[1])
	end
	print("Test set perplexity : " .. g_f3(perplexity(perp / (len - 1))))
	g_enable_dropout(model.rnns)
end

--
-- Clear the context of the network, and do a pass over the first n - 1 words in
-- the sentence.
--
function process_new_sentence(indices)
	input = torch.Tensor(#indices, 1):float()
	for i = 1, #indices do
		input[i][1] = indices[i]
	end
	input = input:expand(#indices, params.batch_size)
	transfer_data(input)

	local perp = 0
	clear_model_context()

	for i = 1, #indices - 1 do
		local x = input[i]:cuda()
		local y = input[i + 1]:cuda()
		perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
		perp = perp + perp_tmp[1]
		g_replace_table(model.s[0], model.s[1])
	end
end

--
-- After we have done a forward pass over the first n - 1 words of the sentence,
-- we can use this function to predict consecutive words.
--
function predict_next_word(indices)
	input = torch.Tensor(params.batch_size):float():fill(indices[#indices]):cuda()
	transfer_data(input)

	perp_tmp, model.s[1], preds = unpack(model.rnns[1]:forward({input, input, model.s[0]}))
	g_replace_table(model.s[0], model.s[1])
	return preds[1]
end

--
-- This function should only be used with character-level models.
--
function predict_next_char(index)
	input = torch.Tensor(params.batch_size):float():fill(index):cuda()
	transfer_data(input)

	perp_tmp, model.s[1], preds = unpack(model.rnns[1]:forward({input, input, model.s[0]}))
	g_replace_table(model.s[0], model.s[1])
	return preds[1]
end

if opt.task ~= "evaluate" then
	print("Configuration parameters:")
	print(params)
	print("")

	g_init_gpu("" .. opt.device)
	print("Loading training data.")
	state_train = {data = transfer_data(ptb.traindataset(params.batch_size))}
	print("Loading validation data.")
	state_valid = {data = transfer_data(ptb.validdataset(params.batch_size))}
	--print("Loading testing data.")
	--state_test  = {data = transfer_data(ptb.testdataset(params.batch_size))}

	--local states = {state_train, state_valid, state_test}
	local states = {state_train, state_valid}
	for _, state in pairs(states) do
		reset_state(state)
	end

	setup()
	train_info = get_train_info()

	-- I don't serialize this information right now, because I don't want to deal
	-- with updating times and counts based on prior values.
	total_cases = 0
	start_time = torch.tic()
	epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
	print("\nStarting epoch " .. train_info.epoch .. ".")

	if not params.use_default_opt and not train_info.opt_method then
		train_info.opt_method = sopt.adadelta
		train_info.opt_state = {
			learning_rate = sopt.constant(1),
			momentum = sopt.constant(0.95),
			momentum_type = sopt.none
		}

		objective = function(x)
			if x ~= paramx then
				paramx:copy(x)
			end

			local perp = fp(state_train)
			if train_info.perps == nil then
				train_info.perps = torch.zeros(epoch_size):add(perp)
			end
			train_info.perps[train_info.iter % epoch_size + 1] = perp
			bp(state_train)

			return perp, paramdx
		end
	end

	while true do
		if params.use_default_opt then
			perp = fp(state_train)
			if train_info.perps == nil then
				train_info.perps = torch.zeros(epoch_size):add(perp)
			end
			train_info.perps[train_info.iter % epoch_size + 1] = perp
			bp(state_train)
		else
			train_info.opt_method(objective, paramx, train_info.opt_state)
		end

		total_cases = total_cases + params.seq_length * params.batch_size
		xlua.progress(train_info.iter % epoch_size + 1, epoch_size)
		train_info.iter = train_info.iter + 1

		if train_info.iter % torch.round(epoch_size / 10) == 10 then
			wps = torch.floor(total_cases / torch.toc(start_time))
			since_beginning = g_d(torch.toc(start_time) / 60)
			perp = perplexity(train_info.perps:mean())
			save_train_progress(perp)

			print('epoch = '                  .. g_f3(train_info.epoch) ..
				', train perp. = '        .. g_f3(perp)             ..
				', wps = '                .. wps                    ..
				', dw:norm() = '          .. g_f3(model.norm_dw)    ..
				', lr = '                 .. params.lr        ..
				', since last restart = ' .. since_beginning        .. ' mins.')
		end

		if train_info.iter % epoch_size == 0 then
			run_valid()
			train_info.epoch = train_info.epoch + 1
			if params.use_default_opt and train_info.epoch > params.epochs_to_lr_decay then
				params.lr = params.lr / params.decay
			end

			xlua.progress(epoch_size, epoch_size)
			print("\nStarting epoch " .. train_info.epoch .. ".")
		end

		if train_info.iter % 33 == 0 then
			cutorch.synchronize()
			collectgarbage()
		end
	end

	-- run_test()
	-- print("Training is over.")
else
	print("Configuration parameters:")
	print(params)
	print("")

	g_init_gpu("" .. opt.device)
	print("Loading training data.")
	state_train = {data = transfer_data(ptb.traindataset(params.batch_size))}
	print("Loading validation data.")
	state_valid = {data = transfer_data(ptb.validdataset(params.batch_size))}
	--print("Loading testing data.")
	--state_test  = {data = transfer_data(ptb.testdataset(params.batch_size))}

	--local states = {state_train, state_valid, state_test}
	local states = {state_train, state_valid}
	for _, state in pairs(states) do
		reset_state(state)
	end

	setup()
	g_disable_dropout(model.rnns)
end
