require "torch"
require "optim"
require "xlua"
require "lfs"

if not opt then
	local cmd = torch.CmdLine()
	cmd:text("Select one of the following options:")
	cmd:option("-task", "create", "create | resume | replace | evaluate")
	cmd:option("-model", "test", "The model name.")
	cmd:option("-version", "current", "current | best_train | best_test")
	cmd:option("-device", 0, "GPU device number.")
	opt = cmd:parse(arg or {})
end

local models_dir = "models"
local model_name = opt.model

if string.match(model_name, "^[A-Za-z0-9_]+") == nil then
	error("Invalid model name `" .. model_name .. "`.")
end

local output_dir = paths.concat(models_dir, model_name)

function remove_file_if_exists(file, silent)
	if not paths.filep(file) then
		return
	end

	local success, err = os.remove(file)
	if success and not silent then
		print("Removed file `" .. file .. "`.")
	elseif not success then
		error("Failed to remove file `" .. file .. "`: " .. err .. ".")
	end
end

function remove_empty_directory(dir, silent)
	if not paths.dirp(dir) then
		error("Could not find directory `" .. dir .. "`.")
	end

	local success, err = lfs.rmdir(dir)
	if success and not silent then
		print("Removed directory `" .. dir .. "`.")
	elseif not success then
		error("Failed to remove directory `" .. dir .. "`: " .. err .. ".")
	end
end

function remove_directory(dir, silent)
	for file in lfs.dir(dir) do
		local path = paths.concat(dir, file)
		if lfs.attributes(path, "mode") ~= "directory" then
			remove_file_if_exists(path, silent)
		end
	end

	remove_empty_directory(dir, silent)
	print("")
end

function make_directory(dir, silent)
	if not paths.mkdir(dir) then
		error("Failed to create directory `" .. dir .. "`.")
	elseif not silent then
		print("Created directory `" .. dir .. "`.")
	end
end

function create_hard_link(target, link, silent)
	local success, err = lfs.link(target, link)
	if success and not silent then
		print("Created hard link `" .. link .. "`.")
	elseif not success then
		error("Failed to create hard link `" .. link .. "`: " .. err .. ".")
	end
end

function rename_file_if_exists(old, new, silent)
	if not paths.filep(old) then
		return
	end

	local success, err = os.rename(old, new)
	if success and not silent then
		print("Renamed file `" .. old .. "` to `" .. new .. "`.")
	elseif not success then
		print("Failed to rename file `" .. old .. "` to `" .. new .. "`: " .. err .. ".")
	end
end

function rename_backup(backup, new)
	if not paths.filep(backup) then
		return true
	elseif paths.filep(backup) and paths.filep(new) then
		print("Both `" .. new .. "` and `" .. backup .. "` exist.")
		return false
	end

	rename_file_if_exists(backup, new)
	return true
end

if opt.task == "create" then
	if paths.dirp(output_dir) then
		error("Model `" .. model_name .. "` already exists.")
	end
	make_directory(output_dir)
elseif opt.task == "resume" or opt.task == "evaluate" then
	if not paths.dirp(output_dir) then
		error("Model `" .. model_name .. "` does not exist.")
	end
elseif opt.task == "replace" then
	if paths.dirp(output_dir) then
		remove_directory(output_dir)
		make_directory(output_dir)
	end
else
	error("Invalid task `" .. opt.task .. "`.")
end

-- Define the paths to the output files for serialization.
local cur_model_fn = paths.concat(output_dir, "cur_model.t7")
local best_train_model_fn = paths.concat(output_dir, "best_train_model.t7")
local best_test_model_fn = paths.concat(output_dir, "best_test_model.t7")

local cur_train_info_fn = paths.concat(output_dir, "cur_train_info.t7")
local best_train_train_info_fn = paths.concat(output_dir, "best_train_train_info.t7")
local best_test_train_info_fn = paths.concat(output_dir, "best_test_train_info.t7")

local cur_model_backup_fn = paths.concat(output_dir, "cur_model_backup.t7")
local best_train_model_backup_fn = paths.concat(output_dir, "best_train_model_backup.t7")
local best_test_model_backup_fn = paths.concat(output_dir, "best_test_model_backup.t7")

local cur_train_info_backup_fn = paths.concat(output_dir, "cur_train_info_backup.t7")
local best_train_train_info_backup_fn = paths.concat(output_dir, "best_train_train_info_backup.t7")
local best_test_train_info_backup_fn = paths.concat(output_dir, "best_test_train_info_backup.t7")

local acc_info_fn = paths.concat(output_dir, "acc_info.t7")
local acc_info_backup_fn = paths.concat(output_dir, "acc_info_backup.t7")

-- Deal with the backup files.
local status = true
status = status and rename_backup(cur_model_backup_fn, cur_model_fn)
status = status and rename_backup(best_train_model_backup_fn, best_train_model_fn)
status = status and rename_backup(best_test_model_backup_fn, best_test_model_fn)
status = status and rename_backup(cur_train_info_backup_fn, cur_train_info_fn)
status = status and rename_backup(best_train_train_info_backup_fn, best_train_train_info_fn)
status = status and rename_backup(best_test_train_info_backup_fn, best_test_train_info_fn)
status = status and rename_backup(acc_info_backup_fn, acc_info_fn)

if not status then
	print("Both backup and non-backup versions of certain files exist (see above output).")
	print("There may be data corruption.")
	print("Please carefully inspect the files, and eliminate the duplicates.")
	os.exit(1)
end

-- Determine the files from which we are to restore the model and training info
-- states.
local target_model_fn = ""
local target_train_info_fn = ""

if opt.version == "current" then
	target_model_fn = cur_model_fn
	target_train_info_fn = cur_train_info_fn
elseif opt.version == "best_train" then
	target_model_fn = best_train_model_fn
	target_train_info_fn = best_train_train_info_fn
elseif opt.version == "best_test" then
	target_model_fn = best_test_model_fn
	target_train_info_fn = best_test_train_info_fn
else
	error("Invalid model version `" .. opt.version .. "`.")
end

-- Deserialize the model and training states, or initialize new ones.
local model_info = {}
local train_info = {}
local acc_info = {}

local is_model_info_loaded = false
local is_train_info_loaded = false

if paths.filep(target_model_fn) then
	print("Restoring model from `" .. target_model_fn .. "`.")
	model_info = torch.load(target_model_fn)
	is_model_info_loaded = true
else
	if opt.task == "evaluate" then
		error("Model file `" .. target_model_fn .. "` not found.")
	end
	--print("Creating new model.")
	--model_info = get_model_info()
end

if paths.filep(target_train_info_fn) then
	print("Restoring training info from `" .. target_train_info_fn .. "`.")
	train_info = torch.load(target_train_info_fn)
	is_train_info_loaded = true
else
	if opt.task == "evaluate" then
		error("Train info file `" .. target_train_info_fn .. "` not found.")
	end
	--print("Initializing training state.")
	--train_info = get_training_info()
end

if paths.filep(acc_info_fn) then
	print("Restoring accuracy info from `" .. acc_info_fn .. "`.")
	acc_info = torch.load(acc_info_fn)
else
	print("Initializing accuracy info.")
	acc_info = {
		best_train = 1e10,
		best_test = 1e10
	}
end

-- Note: these functions are new, and are written for easier integration with
-- main.lua.

function model_info_loaded()
	return is_model_info_loaded
end

function get_model_info()
	return model_info.model, model_info.model_buffers
end

function set_model_info(model, model_buffers)
	model_info.model = model
	model_info.model_buffers = model_buffers
end

function get_train_info()
	if not is_train_info_loaded then
		train_info.iter = 0
		train_info.epoch = 1
	end
	return train_info
end

-- For validation, the "test" sample is actually the validation sample.
local do_train = opt.task ~= "evaluate"
local do_test = true
if not do_train and not do_test then
	error("No training or test data specified; nothing to do.")
end

-- local train_log = {}
-- local test_log = {}
-- 
-- if do_train then
-- 	train_log = optim.Logger(paths.concat(output_dir, "train.log"))
-- end
-- if do_test then
-- 	test_log = optim.Logger(paths.concat(output_dir, "test.log"))
-- end

function save_train_progress(train_perp)
	print("Saving current model and training info.")
	rename_file_if_exists(cur_model_fn, cur_model_backup_fn, true)
	rename_file_if_exists(cur_train_info_fn, cur_train_info_backup_fn, true)
	torch.save(cur_model_fn, model_info)
	torch.save(cur_train_info_fn, train_info)

	if train_perp < acc_info.best_train then
		acc_info.best_train = train_perp

		print("New best train perplexity: updating hard links.")
		rename_file_if_exists(best_train_model_fn,
			best_train_model_backup_fn, true)
		rename_file_if_exists(best_train_train_info_fn,
			best_train_train_info_backup_fn, true)
		create_hard_link(cur_model_fn, best_train_model_fn, true)
		create_hard_link(cur_train_info_fn, best_train_train_info_fn, true)
		remove_file_if_exists(best_train_model_backup_fn, true)
		remove_file_if_exists(best_train_train_info_backup_fn, true)

		print("Saving accuracy info.")
		rename_file_if_exists(acc_info_fn, acc_info_backup_fn, true)
		torch.save(acc_info_fn, acc_info)
		remove_file_if_exists(acc_info_backup_fn, true)
	end

	remove_file_if_exists(cur_model_backup_fn, true)
	remove_file_if_exists(cur_train_info_backup_fn, true)
end

function save_test_progress(test_perp)
	if test_perp < acc_info.best_test then
		acc_info.best_test = test_perp

		print("New best test perplexity: updating hard links.")
		rename_file_if_exists(best_test_model_fn,
			best_test_model_backup_fn, true)
		rename_file_if_exists(best_test_train_info_fn,
			best_test_train_info_backup_fn, true)
		create_hard_link(cur_model_fn, best_test_model_fn, true)
		create_hard_link(cur_train_info_fn, best_test_train_info_fn, true)
		remove_file_if_exists(best_test_model_backup_fn, true)
		remove_file_if_exists(best_test_train_info_backup_fn, true)

		print("Saving accuracy info.")
		rename_file_if_exists(acc_info_fn, acc_info_backup_fn, true)
		torch.save(acc_info_fn, acc_info)
		remove_file_if_exists(acc_info_backup_fn, true)
	end
end
