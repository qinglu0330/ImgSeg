require 'image'
--require 'cunn'
dofile 'networks.lua'


cmd = torch.CmdLine()
cmd:option('-model','fcn8s_Gland_100000.bin','model')
cmd:option('-dataset','Gland','dataset name')
cmd:option('-ext','png','only load a specific type of data')
cmd:option('-qt','fixed_point','quantization type')
cmd:option('-frac','32','fraction width')
cmd:option('-int','4','integer width')
cmd:option('-scale', '0.5', 'scale for binary quantization')
opt = cmd:parse(arg or {})
opt.rundir = cmd:string('Quan', opt, {model == true, dataset == true, ext == true})
--local outputDir = '../output/quantization experiment/fixed point/' .. opt.dataset .. '/' ..opt.rundir
if opt.qt == 'fixed_point' then
    outputDir = paths.concat('../output', 'quantization experiment', opt.qt, opt.dataset, string.format('int = %d, frac = %d', opt.int, opt.frac))
elseif opt.qt == 'binary' then
    outputDir = paths.concat('../output', 'quantization experiment', opt.qt, opt.dataset, string.format('scale = %s', opt.scale))
end
local model = 'fcn8s_Gland_200000_v1.bin'
print('output directory is ',outputDir)
if (not paths.dirp(outputDir)) then
    paths.mkdir(outputDir)
end
-- cmd:log(outputDir .. '/log', opt)

local imgFormat = opt.ext

if opt.dataset == 'Gland' then
	local nChannel = 3
	local nClass = 2
	local classMap = torch.Tensor{0, 1}
end


local datasetPath = paths.concat('../dataset', opt.dataset)
if not paths.dirp(datasetPath) then
    print('dataset not existing')
end
testPath = paths.concat(datasetPath, 'test/image')
testFiles = {}
for file in paths.files(testPath) do
    if file:find('.' .. imgFormat) then
        local testFile = paths.concat(testPath, file)
        table.insert(testFiles, testFile)
    end
end

print('number of test files: ', #testFiles)

print('loading: ' .. model)
net = torch.load(model)

parameters,gradParameters = net.net:getParameters()

function quantize(parameters, type, intWidth, fracWidth)
    print('quantizing parameters ...')
    local lastPercentage = -1
    for i = 1, parameters:size(1) do
        local percentage = math.floor(i/parameters:size(1)*100)
        if percentage > lastPercentage then
            print(string.format('%d%% completed', percentage))
            lastPercentage = percentage
        end
        if type == 'fixed_point' then
            local parameter = parameters[i]
            local absPara = math.abs(parameter)
            local sign = parameter/absPara
            if absPara > math.pow(2,(intWidth))-math.pow(2,-fracWidth) then
                parameters[i] = sign * (math.pow(2,(intWidth))-math.pow(2,-fracWidth))
            else
                keyDigits = math.floor(absPara*math.pow(2, fracWidth+1))
                if math.fmod(keyDigits,2) == 1 then
                    parameters[i] = sign * (keyDigits/math.pow(2, fracWidth+1)+math.pow(2, -(fracWidth+1)))
                else
                    parameters[i] = sign * keyDigits/math.pow(2, fracWidth+1)
                end
            end
        elseif type == 'binary' then
            if parameters[i] > 0 then
                parameters[i] = opt.scale
            else
                parameters[i] = - opt.scale
            end
        end
    end
end

quantize(parameters, opt.qt, opt.int, opt.frac)

net.net:evaluate();
for j = 1, #testFiles do
	local testImg = image.load(testFiles[j], nChannel, 'double')
	local pred = net:infer(testImg, classMap)
    -- pred = demirror(pred, testImg:size(2), testImg:size(3))
	local fileName = string.split(testFiles[j],'/')
	fileName = paths.concat(outputDir, fileName[#fileName])
	print(string.format('saving image into %s', fileName))
	image.save(fileName, pred)
end
collectgarbage()



