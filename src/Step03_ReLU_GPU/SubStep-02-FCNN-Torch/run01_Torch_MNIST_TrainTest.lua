require 'nn'
require 'image'
require 'torch'
require 'xlua'
require 'optim'

-- (-1) Parse command line arguments
if #arg<1 then
    print('Usage: $program {L1:L2:...:Ln} {batchSize} {#Epochs} {numReps}')
    os.exit(1)
end

strParam=arg[1]
arrParamStr=string.split(strParam,':')
arrParam={}
for i=1,#arrParamStr do
    arrParam[i]=tonumber(arrParamStr[i])
end

paramBatchSize=128
paramEpochs=10
paramReps=10

if #arg>3 then
    paramBatchSize  = tonumber(arg[2])
    paramEpochs     = tonumber(arg[3])
    paramReps       = tonumber(arg[4])
end

print('Parameter arguments: ')
print(arrParam)
print('Batch Size: '..paramBatchSize..', #Epochs: '..paramEpochs..', #Reps: '..paramReps)
--------------------------------------------------------------------------------------
-- (0) Load Dataset
trainData = torch.load('mnist.t7/train_32x32.t7', 'ascii')
testData  = torch.load('mnist.t7/test_32x32.t7', 'ascii')
trainData.size=function() return (#trainData.labels)[1] end
testData.size =function() return (#testData.labels)[1] end
print(trainData)
print(testData)

trainSize=trainData.size()
testSize=testData.size()

print('#Train: ' .. trainSize .. ', #Test: '.. testSize)

-- trainSize=1000
-- testSize=1000

-- (1) Define FCN Model:
nfeats=1
nrow=32
ncol=32
ninputs=nfeats*nrow*ncol
noutputs = 10
--
modelFCN=nn.Sequential()
modelFCN:add(nn.Reshape(ninputs))
for i=1,#arrParam do
    if i==1 then
        modelFCN:add(nn.Linear(ninputs, arrParam[i]))
        modelFCN:add(nn.Sigmoid())
    else
        modelFCN:add(nn.Linear(arrParam[i-1], arrParam[i]))
        modelFCN:add(nn.Sigmoid())
    end
end
modelFCN:add(nn.Linear(arrParam[#arrParam], noutputs))
modelFCN:add(nn.LogSoftMax())
print(modelFCN)
--
loss = nn.ClassNLLCriterion()

-- (2) Setup train parameters
local optimState = {
   learningRate = 1e-2,
   momentum = 0.9,
   weightDecay = 0.0005
}
batchSize = paramBatchSize
-- (3) Prepare additional parameters for Gradient calculation
local x = torch.Tensor(batchSize,trainData.data:size(2),
         trainData.data:size(3), trainData.data:size(4))
local yt = torch.Tensor(batchSize)

local w,dE_dw = modelFCN:getParameters()

-- (4) Define TRAIN function:
local function train()
    local shuffle = torch.randperm(trainSize)
    for t = 1, trainSize, batchSize do
        --xlua.progress(t, trainSize)
        collectgarbage()
        -- [check batch fits]
        if (t + batchSize - 1) > trainSize then
            break
        end
        -- [create mini batch]
        local idx = 1
        for i = t, t + batchSize - 1 do
            x[idx] = trainData.data[shuffle[i]]
            yt[idx] = trainData.labels[shuffle[i]]
            if yt[idx] == 0 then
                yt[idx] = 1
            end
            idx = idx + 1
        end
        -- create [local function] to evaluate f(X) and df/dX
        local eval_E = function(w)
            -- reset gradients
            dE_dw:zero()
            -- evaluate function for complete mini batch
            local y = modelFCN:forward(x)
            local E = loss:forward(y,yt)
            -- estimate df/dW
            local dE_dy = loss:backward(y,yt)
            modelFCN:backward(x,dE_dy)
            -- return f and df/dX
            return E, dE_dw
        end
        -- optimize on current mini-batch
        optim.sgd(eval_E, w, optimState)
    end
end

function test()
   -- labels
    classes = {'1','2','3','4','5','6','7','8','9','0'}
    confusion = optim.ConfusionMatrix(classes)
    for t = 1,testSize do
        -- disp progress
        -- xlua.progress(t, testSize)
        -- get new sample
        local input = testData.data[t]
        local target = testData.labels[t]
        input=input:double()
        -- test sample
        local pred = modelFCN:forward(input):view(10)
        confusion:add(pred, target)
    end
    print(confusion)
    return confusion.averageValid
end

--
modelName='ModelFCN-Torch-p'..strParam..'-b'..paramBatchSize..'-e'..paramEpochs
foutLog = modelName .. '-Log.txt'
file = io.open (foutLog, 'w')
file:write('model, timeTrain, timeTest, acc\n')
for rr = 1,paramReps do
    local time = sys.clock()
    for i = 1, paramEpochs do
        train()
    end
    timeTrain = sys.clock() - time
    retACC=test()
    timeTest = sys.clock() - time
    tstr = '' .. modelName .. ', ' .. timeTrain .. ', ' .. timeTest .. ', '..retACC
    print(tstr)
    file:write(tstr..'\n')
    -- torch.save('modelFCN-p'..strParam..'.net', modelFCN)
end
io.close(file)
