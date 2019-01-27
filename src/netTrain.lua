require 'image'
--require 'cunn'
dofile 'networks.lua'


local datasetName = 'DRIVE'
local netName = 'unet'

if datasetName == 'Gland' then
	imgFormat = 'png'
	nChannel = 3
	nClass = 2
	classMap = torch.Tensor{0, 1}
elseif datasetName == 'DRIVE' then
    imgFormat = 'png'
    nChannel = 3
    nClass = 2
    classMap = torch.Tensor{0, 1}
end

local net
if netName == 'fcn8s' then
    net = fcn8s(nClass, nChannel)
elseif netName == 'unet' then
    net = unet(nClass, nChannel)
end



local epochNum = 200000
local datasetPath = paths.concat('../dataset', datasetName)
local outputDir = '../output/' .. netName .. '/' .. datasetName .. '/train'

if not paths.dirp(datasetPath) then
    print('dataset not existing')
end


trainPath = paths.concat(datasetPath, 'train/image')
truthPath = paths.concat(datasetPath, 'train/truth')
trainingFiles = {}
for file in paths.files(trainPath) do
    if file:find('.' .. imgFormat) then
        trainFile = paths.concat(trainPath, file)
        truthFile = paths.concat(truthPath, file)
        if paths.filep(truthFile) then
            table.insert(trainingFiles, {trainFile, truthFile})
        end
    end
end

testPath = paths.concat(datasetPath, 'test/image')
testFiles = {}
for file in paths.files(testPath) do
    if file:find('.' .. imgFormat) then
        local testFile = paths.concat(testPath, file)
        table.insert(testFiles, testFile)
    end
end

if (not paths.dirp(outputDir)) then
    paths.mkdir(outputDir)
end


local learningRate = 0.001


checkpointDir = paths.concat(outputDir, 'checkpoint')
if (not paths.dirp(checkpointDir)) then
    paths.mkdir(checkpointDir)
end

print('training ...')
net:train(epochNum, trainingFiles, classMap, checkpointDir)


