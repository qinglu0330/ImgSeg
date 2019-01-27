require 'nn'
require 'nngraph'
require 'image'
require 'optim'
local gpu = 1
if gpu == 1 then
	require 'cunn'
	require 'cudnn'
end

local checkpoint = 20000
function augment(imgPair)
	local imgPair = imgPair or {}
	local theta = math.random(4) * 0.5 * math.pi
	local trainImg = image.rotate(imgPair[1], theta)
	local truthImg = image.rotate(imgPair[2], theta)
	if 	math.random() > 0.5 then
		trainImg = image.hflip(trainImg)
		truthImg = image.hflip(truthImg)
	end
	return {trainImg, truthImg}
end

function extrapolate(img, top, bottom, left, right)
    local img = img or {} 
    local tempImg = img:clone()
    local nRow = img:size(2)
    local nCol = img:size(3)
    -- print(nRow, nCol, top, bottom, left, right)
    if 	top > 0 then
    	tempImg = tempImg[{{}, {top, nRow}, {}}]
    else
    	local topWidth = math.abs(top) + 1
        tempImg = torch.cat(image.vflip(img[{{}, {1, topWidth}, {}}]), tempImg, 2)
    end
    local nRowNew = tempImg:size(2)
    if 	bottom <= nRow then	
    	local bottomWidth = nRow - bottom
    	tempImg = tempImg[{{}, {1, nRowNew - bottomWidth}, {}}]
    else
    	local bottomWidth = bottom - nRow
        tempImg = torch.cat(tempImg, image.vflip(img[{{}, {nRow - bottomWidth + 1, nRow}, {}}]), 2)
    end
    img = tempImg
    if 	left > 0 then
    	tempImg = tempImg[{{}, {}, {left, nCol}}]
    else
    	local leftWidth = math.abs(left) + 1
       	tempImg = torch.cat(image.hflip(tempImg[{{}, {}, {1, leftWidth}}]), tempImg, 3)
    end
    local nColNew = tempImg:size(3)
    if right <= nCol then 	
    	local rightWidth = nCol - right
    	tempImg = tempImg[{{}, {}, {1, nColNew - rightWidth}}]
    else
    	local rightWidth = right - nCol
    	tempImg = torch.cat(tempImg, image.hflip(img[{{}, {}, {nCol - rightWidth +1, nCol}}]), 3)
    end
    return tempImg
end

function unet(nClass, nChannel)
	--print(nClass)
	local nClass = nClass or 2
	local nChannel = nChannel or 3

	-- Group 1
	local input1 = nn.Identity()()
	local conv1_1 = nn.SpatialConvolution(nChannel, 64, 3, 3, 1, 1, 0, 0)(input1)
	local relu1_1 = nn.ReLU(true)(conv1_1)
	local conv1_2 = nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 0, 0)(relu1_1)
	local relu1_2 = nn.ReLU(true)(conv1_2)

	-- Group 2
	local input2 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(relu1_2)
	local conv2_1 = nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 0, 0)(input2)
	local relu2_1 = nn.ReLU(true)(conv2_1)
	local conv2_2 = nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 0, 0)(relu2_1)
	local relu2_2 = nn.ReLU(true)(conv2_2)

	-- Group 3
	local input3 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(relu2_2)
	local conv3_1 = nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 0, 0)(input3)
	local relu3_1 = nn.ReLU(true)(conv3_1)
	local conv3_2 = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 0, 0)(relu3_1)
	local relu3_2 = nn.ReLU(true)(conv3_2)

	-- Group 4
	local input4 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(relu3_2)
	local conv4_1 = nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 0, 0)(input4)
	local relu4_1 = nn.ReLU(true)(conv4_1)
	local conv4_2 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 0, 0)(relu4_1)
	local relu4_2 = nn.ReLU(true)(conv4_2)

	-- Group 5
	local input5 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(relu4_2)
	local conv5_1 = nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 0, 0)(input5)
	local relu5_1 = nn.ReLU(true)(conv5_1)
	local conv5_2 = nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 0, 0)(relu5_1)
	local relu5_2 = nn.ReLU(true)(conv5_2)
	
	
	-- Group 6
	local input6_1 = nn.SpatialZeroPadding(-4, -4, -4, -4)(relu4_2)
	local input6_2 = nn.SpatialFullConvolution(1024, 512, 4, 4, 2, 2, 1, 1, 0, 0)(relu5_2)
	local concat6 = nn.JoinTable(-3)({input6_1, input6_2})
	local conv6_1 = nn.SpatialConvolution(1024, 512, 3, 3, 1, 1, 0, 0)(concat6)
	local relu6_1 = nn.ReLU(true)(conv6_1)
	local conv6_2 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 0, 0)(relu6_1)
	local relu6_2 = nn.ReLU(true)(conv6_2)


	-- Group 7
	local input7_1 = nn.SpatialZeroPadding(-16, -16, -16, -16)(relu3_2)
	local input7_2 = nn.SpatialFullConvolution(512, 256, 4, 4, 2, 2, 1, 1, 0, 0)(relu6_2)
	local concat7 = nn.JoinTable(-3)({input7_1, input7_2})
	local conv7_1 = nn.SpatialConvolution(512, 256, 3, 3, 1, 1, 0, 0)(concat7)
	local relu7_1 = nn.ReLU(true)(conv7_1)
	local conv7_2 = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 0, 0)(relu7_1)
	local relu7_2 = nn.ReLU(true)(conv7_2)

	-- Group 8
	local input8_1 = nn.SpatialZeroPadding(-40, -40, -40, -40)(relu2_2)
	local input8_2 = nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1, 0, 0)(relu7_2)
	local concat8 = nn.JoinTable(-3)({input8_1, input8_2})
	local conv8_1 = nn.SpatialConvolution(256, 128, 3, 3, 1, 1, 0, 0)(concat8)
	local relu8_1 = nn.ReLU(true)(conv8_1)
	local conv8_2 = nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 0, 0)(relu8_1)
	local relu8_2 = nn.ReLU(true)(conv8_2)

	-- Group 9
	local input9_1 = nn.SpatialZeroPadding(-88, -88, -88, -88)(relu1_2)
	local input9_2 = nn.SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1, 0, 0)(relu8_2)
	local concat9 = nn.JoinTable(-3)({input9_1, input9_2})
	local conv9_1 = nn.SpatialConvolution(128, 64, 3, 3, 1, 1, 0, 0)(concat9)
	local relu9_1 = nn.ReLU(true)(conv9_1)
	local conv9_2 = nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 0, 0)(relu9_1)
	local relu9_2 = nn.ReLU(true)(conv9_2)

	-- output
	local output = nn.SpatialConvolution(64, nClass, 1, 1, 1, 1, 0, 0)(relu9_2)
	local net = nn.gModule({input1}, {output})

	-- parameter initialization 
	for k, module in ipairs(net:listModules()) do
	    if (torch.type(module):find('SpatialFullConvolution')) then
	        local stdv=math.sqrt(2/(module.nInputPlane*4))
	        module.weight:normal(0, stdv)
	        module.bias:normal(0, stdv)
	    end
	    if (torch.type(module):find('SpatialConvolution')) then
	        local stdv=math.sqrt(2/(module.nInputPlane*module.kH*module.kW))
	        module.weight:normal(0, stdv)
	        module.bias:normal(0, stdv)
	    end
	end
	if gpu == 1 then
		net = net:cuda()
	end 

	local unet = {net = net}

	function getBatch(trainingFiles, batchSize)
		-- function getBatch(trainingFiles, batchSize, nChannel)
		local trainingFiles = trainingFiles or {}
		local batchSize = batchSize or #trainingFiles
		local batch = {}
		for i = 1, batchSize do
			local N = #trainingFiles
			local selection = math.random(1, N)
			-- getting the input image 
			local sample = image.load(trainingFiles[selection][1], nChannel, 'double')
			local sampleWidth = sample:size(3)
			local sampleHeight = sample:size(2)
			local left = math.random(1, sampleWidth - 388 + 1) 
			local right = left + 387 
			local top = math.random(1, sampleHeight - 388 + 1) 
			local bottom = top + 387
			local theta = math.random() * 2 * math.pi
			local margin = (1.5 * 572 - 388) / 2
			local rotateTrain = extrapolate(sample, top - margin, bottom + margin, left - margin, right + margin)
			local rotatedTrain = image.rotate(rotateTrain, theta)
			local trainImg = image.crop(rotatedTrain, 572 / 4, 572 / 4, 572 / 4 + 572, 572 / 4 + 572)
			-- getting the truth of corresponding to the input image
			local sample = image.load(trainingFiles[selection][2], 1, 'double')
			local rotateTruth = extrapolate(sample, top - margin, bottom + margin, left - margin, right + margin)
			local rotatedTruth = image.rotate(rotateTruth, theta)
			local truthImg = image.crop(rotatedTruth, (1.5 * 572 - 388) / 2, (1.5 * 572 - 388) / 2, (1.5 * 572 - 388) / 2 + 388, (1.5 * 572 - 388) / 2 + 388)
			local ifFlip = math.random()
			if 	ifFlip > 0.5 then
				trainImg = image.hflip(trainImg)
				truthImg = image.hflip(truthImg)
			end
			table.insert(batch, {trainImg, truthImg})
		end
		function batch:size() return batchSize; end
		return batch
	end

	

	function unet:train(epochNum, trainingFiles, classMap, checkpointDir)
		local epochNum = epochNum or {}
		local trainingFiles = trainingFiles or {}
		local classMap = classMap or torch.linspace(1, nClass, nClass)
		local checkpointDir = checkpointDir or './'
		-- local nClass = classMap:size(1) or 3
		local errLog = torch.Tensor(100):fill(0)
		local meanErr
		local params, gradParams = self.net:getParameters()
		local config = {learningRate = 0.0001, momentum = 0.99}
		local criterion = cudnn.SpatialCrossEntropyCriterion()
		function feval(params)
			local batch = getBatch(trainingFiles, 1)
			local img = batch[1][1]
			local truth = batch[1][2]
			self.net:training()
			truth:apply(function(x)
				    for j = 1, classMap:size(1) do
				        if x == classMap[j] then
				            return j
				        end
				    end
				    return 'error'
				end)
			local input = torch.Tensor(1, img:size(1), img:size(2), img:size(3))
			input[1] = img
			local target = truth
			if gpu == 1 then
				input = input:cuda()
				target = target:cuda()
				criterion = criterion:cuda()
			end
			gradParams:zero()
			local pred = self.net:forward(input)
			local err = criterion:forward(pred, target)
			local gradLoss = criterion:backward(pred, target)
			self.net:backward(input, gradLoss)
			return err, gradParams
		end
		for epoch = 1, epochNum do
			if epoch > 1000 then
				config.learningRate = 0.00001
			end
			local _, err = optim.adam(feval, params, config)
			errLog = torch.cat(errLog:sub(2, 100), torch.Tensor(1):fill(err[1]), 1)
			if epoch < 100 then
				meanErr = errLog:sub(-epoch, -1):mean()
				stringToPrint = string.format('Mean mean error of the past %d epochs at Epoch %d is %.6f', epoch, epoch, meanErr)
			else
				meanErr = errLog:mean()
				stringToPrint = string.format('Mean mean error of the past %d epochs at Epoch %d is %.6f', 100, epoch, meanErr)
			end
			print(stringToPrint)
			-- logFile = io.open(checkpointDir .. '/log.txt', 'a')
			-- logFile:write(stringToPrint .. '\n')
			-- logFile:close()
			if (epoch % checkpoint == 0) then
				local netName= string.format('unet.bin');
				local netDir = paths.concat(checkpointDir, 'net', epoch)
				if (not paths.dirp(netDir)) then
				    paths.mkdir(netDir)
				end
				netName = paths.concat(netDir, netName)
				self.net:clearState()
				print(string.format('saving net at Epoch %d', epoch))
				torch.save(netName, self)

				local validationDir = paths.concat(checkpointDir, 'validation', epoch)
				if (not paths.dirp(validationDir)) then
				    paths.mkdir(validationDir)
				end
				for j = 1, #trainingFiles do
					local validationFile = trainingFiles[j][1]
					local validationImg = image.load(validationFile, nChannel, 'double')
					local output = self:infer(validationImg, classMap)
					local fileName = string.split(validationFile,'/')
					fileName = paths.concat(validationDir, fileName[#fileName])
					print(string.format('saving image into %s', fileName))
					image.save(fileName, output)
				end
			end
			collectgarbage()
		end
		return
	end

	function unet:infer(img, classMap)
		local img = img or torch.rand(3, 32, 32)
		local classMap = classMap or torch.linspace(0, 255, 3)
		-- classMap = classMap:cuda()
		local imgChannelNum = img:size(1)
		local imgRowNum = img:size(2)
		local imgColNum = img:size(3)
		local stepSize = math.floor(388/1)
		local tile = torch.Tensor(imgChannelNum, 388, 388);
		local weightTile = torch.Tensor(1, 388, 388):fill(1)
		for i = 1, 388 do
			for j = 1, 388 do
				weightTile[1][i][j] = 1/ math.max((math.abs(i - 194.5) + math.abs(j - 194.5)), 1)
			end
		end
		local fullPred = torch.Tensor(1, imgRowNum, imgColNum):fill(0)
		local weightSum = torch.Tensor(fullPred:size()):fill(0)
		local tileCount = 0
		self.net:evaluate()
		local rowStart, colStart, rowEnd, colEnd
		rowStart = 1
		rowEnd = rowStart + 387
		while rowEnd < imgRowNum + stepSize do
			if 	rowEnd > imgRowNum then
				rowEnd = imgRowNum
				rowStart = imgRowNum - 387
			end
			colStart = 1
			colEnd = colStart + 387
			while colEnd < imgColNum + stepSize do
				tileCount = tileCount + 1
				-- print(string.format("Tile No. %d", tileCount))
				if 	colEnd > imgColNum then
					colEnd = imgColNum
					colStart = imgColNum - 387
				end
				-- tile:copy(img:sub(1,imgChannelNum,rowStart, rowEnd, colStart, colEnd))

				local softMax = nn.SpatialSoftMax()
				-- print("top bottom left right are", rowStart, rowEnd, colStart, colEnd)
				local input = extrapolate(img, rowStart - 92, rowEnd + 92, colStart - 92, colEnd + 92)
				if gpu == 1 then
					softMax = softMax:cuda()
					input = input:cuda()
					classMap = classMap:cuda()

				end
				local output = self.net:forward(input)
				local tileResult = softMax:forward(output)
				local _, idx = tileResult:max(1)
				idx = idx:double()
				local tilePred = idx:apply(function(x) return classMap[x] end)
				tilePred:cmul(weightTile)
				fullPred:sub(1, 1, rowStart, rowEnd, colStart, colEnd):add(tilePred)
				weightSum:sub(1, 1, rowStart, rowEnd, colStart, colEnd):add(weightTile)
				colStart = colStart + stepSize
				colEnd = colStart + 387
			end
			rowStart = rowStart + stepSize
			rowEnd = rowStart + 387
		end
		fullPred:cdiv(weightSum)
		return fullPred
	end

	return unet
end


function fcn8s(nClass, nChannel, tileSize)
	--print(nClass)
	local nClass = nClass or 2
	local nChannel = nChannel or 3
	local tileSize = tileSize or 32 * 12

	local input = nn.Identity()()

	-- Group 1
	local conv1_1 = nn.SpatialConvolution(nChannel, 64, 3, 3, 1, 1, 1, 1)(input)
	local relu1_1 = nn.ReLU(true)(conv1_1)
	local conv1_2 = nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(relu1_1)
	local relu1_2 = nn.ReLU(true)(conv1_2)
	local pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(relu1_2)

	-- Group 2
	local conv2_1 = nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)(pool1)
	local relu2_1 = nn.ReLU(true)(conv2_1)
	local conv2_2 = nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(relu2_1)
	local relu2_2 = nn.ReLU(true)(conv2_2)
	local pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(relu2_2)

	-- Group 3
	local conv3_1 = nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)(pool2)
	local relu3_1 = nn.ReLU(true)(conv3_1)
	local conv3_2 = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(relu3_1)
	local relu3_2 = nn.ReLU(true)(conv3_2)
	local conv3_3 = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(relu3_2)
	local relu3_3 = nn.ReLU(true)(conv3_3)
	local pool3 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(relu3_3)

	-- Group 4
	local conv4_1 = nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)(pool3)
	local relu4_1 = nn.ReLU(true)(conv4_1)
	local conv4_2 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu4_1)
	local relu4_2 = nn.ReLU(true)(conv4_2)
	local conv4_3 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu4_2)
	local relu4_3 = nn.ReLU(true)(conv4_3)
	local pool4 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(relu4_3)

	-- Group 5
	local conv5_1 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(pool4)
	local relu5_1 = nn.ReLU(true)(conv5_1)
	local conv5_2 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu5_1)
	local relu5_2 = nn.ReLU(true)(conv5_2)
	local conv5_3 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu5_2)
	local relu5_3 = nn.ReLU(true)(conv5_3)
	local pool5 = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)(relu5_3)


	-- Group 6
	local conv6 = nn.SpatialConvolution(512, 4096, 7, 7, 1, 1, 3, 3)(pool5)
	local relu6 = nn.ReLU(true)(conv6)
	local drop6 = nn.SpatialDropout()(relu6)

	-- Group 7
	local conv7 = nn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0)(drop6)
	local relu7 = nn.ReLU(true)(conv7)
	local drop7 = nn.SpatialDropout()(relu7)

	-- End Score
	local score7 = nn.SpatialConvolution(4096, nClass, 1, 1, 1, 1, 0, 0)(drop7)
	local score7x2 = nn.SpatialFullConvolution(nClass, nClass, 4, 4, 2, 2, 1, 1, 0, 0)(score7)
	local score4 = nn.SpatialConvolution(512, nClass, 1, 1, 1, 1, 0, 0)(pool4)
	local fuse1 = nn.CAddTable(false)({score7x2, score4})
	local fuse1x2 = nn.SpatialFullConvolution(nClass, nClass, 4, 4, 2, 2, 1, 1, 0, 0)(fuse1)
	local score3 = nn.SpatialConvolution(256, nClass, 1, 1, 1, 1, 0, 0)(pool3)
	local fuse2 = nn.CAddTable(false)({fuse1x2, score3})
	local fuse2x8 = nn.SpatialFullConvolution(nClass, nClass, 16, 16, 8, 8, 4, 4, 0, 0)(fuse2)
	-- local pred = nn.SpatialSoftMax()(fuse2x8)
	local net = nn.gModule({input}, {fuse2x8})

	for k, module in ipairs(net:listModules()) do
	    if (torch.type(module):find('SpatialFullConvolution')) then
	        local stdv=math.sqrt(2/(module.nInputPlane*4))
	        module.weight:normal(0, stdv)
	        module.bias:normal(0, stdv)
	    end
	    if (torch.type(module):find('SpatialConvolution')) then
	        local stdv=math.sqrt(2/(module.nInputPlane*module.kH*module.kW))
	        module.weight:normal(0, stdv)
	        module.bias:normal(0, stdv)
	    end
	end
	if gpu == 1 then
		net = net:cuda()
	end 

	local fcn8s = {net = net}

	function getBatch(trainingFiles, batchSize)
		local trainingFiles = trainingFiles or {}
		local batchSize = batchSize or #trainingFiles
		local batch = {}
		for i = 1, batchSize do
			local N = #trainingFiles
			local selection = math.random(1, N)
			-- getting the input image 
			local sample = image.load(trainingFiles[selection][1], nChannel, 'double')
			-- print(sample:size())
			local sampleWidth = sample:size(3)
			local sampleHeight = sample:size(2)
			local left = math.random(1, sampleWidth - tileSize + 1) 
			local right = left + tileSize - 1 
			local top = math.random(1, sampleHeight - tileSize + 1) 
			local bottom = top + tileSize - 1
			local theta = math.random() * 2 * math.pi
			local margin = 0.5 * tileSize / 2
			local rotateTrain = extrapolate(sample, top - margin, bottom + margin, left - margin, right + margin)
			local rotatedTrain = image.rotate(rotateTrain, theta)
			local trainTile = image.crop(rotatedTrain, margin, margin, margin + tileSize, margin + tileSize)
			local sample = image.load(trainingFiles[selection][2], 1, 'double')
			local rotateTruth = extrapolate(sample, top - margin, bottom + margin, left - margin, right + margin)
			local rotatedTruth = image.rotate(rotateTruth, theta)
			local truthTile = image.crop(rotatedTruth, margin, margin, margin + tileSize, margin + tileSize)
			local ifFlip = math.random()
			if 	ifFlip > 0.5 then
				trainImg = image.hflip(trainTile)
				truthImg = image.hflip(truthTile)
			end
			table.insert(batch, {trainTile, truthTile})
		end
		function batch:size() return batchSize; end
		return batch
	end

	function fcn8s:train(epochNum, trainingFiles, classMap, checkpointDir)
		local epochNum = epochNum or {}
		local trainingFiles = trainingFiles or {}
		local classMap = classMap or torch.linspace(1, nClass, nClass)
		local checkpointDir = checkpointDir or './'
		-- local nClass = classMap:size(1) or 3
		local errLog = torch.Tensor(100):fill(0)
		local meanErr
		local params, gradParams = self.net:getParameters()
		local config = {learningRate = 0.0001, momentum = 0.99}
		local criterion = cudnn.SpatialCrossEntropyCriterion()
		function feval(params)
			local batch = getBatch(trainingFiles, 1)
			local img = batch[1][1]
			local truth = batch[1][2]
			self.net:training()
			truth:apply(function(x)
				    for j = 1, classMap:size(1) do
				        if x == classMap[j] then
				            return j
				        end
				    end
				    return 'error'
				end)
			local input = torch.Tensor(1, img:size(1), img:size(2), img:size(3))
			input[1] = img
			local target = truth
			if gpu == 1 then
				input = input:cuda()
				target = target:cuda()
				criterion = criterion:cuda()
			end
			gradParams:zero()
			local pred = self.net:forward(input)
			local err = criterion:forward(pred, target)
			local gradLoss = criterion:backward(pred, target)
			self.net:backward(input, gradLoss)
			return err, gradParams
		end
		for epoch = 1, epochNum do
			if epoch > 1000 then
				config.learningRate = 0.00001
			end
			local _, err = optim.adam(feval, params, config)
			errLog = torch.cat(errLog:sub(2, 100), torch.Tensor(1):fill(err[1]), 1)
			if epoch < 100 then
				meanErr = errLog:sub(-epoch, -1):mean()
				stringToPrint = string.format('Mean mean error of the past %d epochs at Epoch %d is %.6f', epoch, epoch, meanErr)
			else
				meanErr = errLog:mean()
				stringToPrint = string.format('Mean mean error of the past %d epochs at Epoch %d is %.6f', 100, epoch, meanErr)
			end
			print(stringToPrint)
			-- logFile = io.open(checkpointDir .. '/log.txt', 'a')
			-- logFile:write(stringToPrint .. '\n')
			-- logFile:close()
			if (epoch % checkpoint == 0) then
				local netName = string.format('fcn8s.bin');
				local netDir = paths.concat(checkpointDir, 'net', epoch)
				if (not paths.dirp(netDir)) then
				    paths.mkdir(netDir)
				end
				netName = paths.concat(netDir, netName)
				self.net:clearState()
				print(string.format('saving net at Epoch %d', epoch))
				torch.save(netName, self)

				local validationDir = paths.concat(checkpointDir, 'validation', epoch)
				if (not paths.dirp(validationDir)) then
				    paths.mkdir(validationDir)
				end
				for j = 1, #trainingFiles do
					local validationFile = trainingFiles[j][1]
					local validationImg = image.load(validationFile, nChannel, 'double')
					local output = self:infer(validationImg, classMap)
					local fileName = string.split(validationFile,'/')
					fileName = paths.concat(validationDir, fileName[#fileName])
					print(string.format('saving image into %s', fileName))
					image.save(fileName, output)
				end
			end
			collectgarbage()
		end
		return
	end

	function fcn8s:infer(img, classMap)
		local img = img or torch.rand(3, 32, 32)
		local classMap = classMap or torch.linspace(0, 255, 3)
		-- classMap = classMap:cuda()
		local imgChannelNum = img:size(1)
		local imgRowNum = img:size(2)
		local imgColNum = img:size(3)
		local stepSize = math.floor(tileSize/4)
		local tile = torch.Tensor(imgChannelNum, tileSize, tileSize);
		local weightTile = torch.Tensor(1, tileSize, tileSize):fill(1)
		for i = 1, tileSize do
			for j = 1, tileSize do
				weightTile[1][i][j] = 1/ math.max((math.abs(i - (tileSize + 1)/2) + math.abs(j - (tileSize + 1)/2)), 1)
			end
		end
		local fullPred = torch.Tensor(1, imgRowNum, imgColNum):fill(0)
		local weightSum = torch.Tensor(fullPred:size()):fill(0)
		local tileCount = 0
		self.net:evaluate()
		local rowStart, colStart, rowEnd, colEnd
		rowStart = 1
		rowEnd = rowStart + tileSize - 1
		while rowEnd < imgRowNum + stepSize do
			if 	rowEnd > imgRowNum then
				rowEnd = imgRowNum
				rowStart = imgRowNum - (tileSize - 1)
			end
			colStart = 1
			colEnd = colStart + tileSize - 1
			while colEnd < imgColNum + stepSize do
				tileCount = tileCount + 1
				-- print(string.format("Tile No. %d", tileCount))
				if 	colEnd > imgColNum then
					colEnd = imgColNum
					colStart = imgColNum - (tileSize - 1)
				end
				tile:copy(img:sub(1,imgChannelNum,rowStart, rowEnd, colStart, colEnd))

				local softMax = nn.SpatialSoftMax()
				-- print("top bottom left right are", rowStart, rowEnd, colStart, colEnd)
				local input = img[{{}, {rowStart, rowEnd}, {colStart, colEnd}}]
				if gpu == 1 then
					softMax = softMax:cuda()
					input = input:cuda()
					classMap = classMap:cuda()

				end
				local output = self.net:forward(input)
				local tileResult = softMax:forward(output)
				local _, idx = tileResult:max(1)
				idx = idx:double()
				local tilePred = idx:apply(function(x) return classMap[x] end)
				tilePred:cmul(weightTile)
				fullPred:sub(1, 1, rowStart, rowEnd, colStart, colEnd):add(tilePred)
				weightSum:sub(1, 1, rowStart, rowEnd, colStart, colEnd):add(weightTile)
				colStart = colStart + stepSize
				colEnd = colStart + (tileSize - 1)
			end
			rowStart = rowStart + stepSize
			rowEnd = rowStart + (tileSize - 1)
		end
		fullPred:cdiv(weightSum)
		return fullPred
	end

	return fcn8s
end

	
	
