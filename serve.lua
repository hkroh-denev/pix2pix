-- serve

require 'image'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')

opt = {
    DATA_ROOT = './datasets/facades',           -- path to images (should have subfolders 'train', 'val', etc)
    name = 'facades_generation',
    batchSize = 1,            -- # images in batch
    loadSize = 256,           -- scale images to this size
    fineSize = 256,           --  then crop to this size
    flip=0,                   -- horizontal mirroring data augmentation
    display = 1,              -- display samples while training. 0 = false
    display_id = 200,         -- display window id.
    gpu = 0,                  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    how_many = 'all',         -- how many test images to run (set to all to run on every image found in the data/phase folder)
    which_direction = 'BtoA', -- AtoB or BtoA
    phase = 'val',            -- train, val, test ,etc
    preprocess = 'regular',   -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
    aspect_ratio = 1.0,       -- aspect ratio of result images
    name = '',                -- name of experiment, selects which model to run, should generally should be passed on command line
    input_nc = 3,             -- #  of input image channels
    output_nc = 3,            -- #  of output image channels
    serial_batches = 1,       -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,    -- iter into serial image list
    cudnn = 0,                -- set to 0 to not use cudnn (untested)
    checkpoints_dir = './checkpoints', -- loads models from here
    results_dir='./results/',          -- saves results here
    which_epoch = 'latest',            -- which epoch to test? set to 'latest' to use latest cached model
    nThreads = 2,
    phase = 'val',
    netG_name = 'facades_generation/latest_net_G'
}

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(torch.random(1, 10000))

-- translation direction
local idx_A = nil
local idx_B = nil
local input_nc = opt.input_nc
local output_nc = opt.output_nc
if opt.which_direction=='AtoB' then
  idx_A = {1, input_nc}
  idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
  idx_A = {input_nc+1, input_nc+output_nc}
  idx_B = {1, input_nc}
else
  error(string.format('bad direction %s',opt.which_direction))
end
----------------------------------------------------------------------------

local input = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

print('checkpoints_dir', opt.checkpoints_dir)
local netG = util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt)

print(netG)

path = 'input.jpg'
local im = image.load(path, 3, 'float')
local h = im:size(2)
local w = im:size(3)

im = image.scale(im, opt.loadSize, opt.loadSize)
local perm = torch.LongTensor{3, 2, 1}
im = im:index(1, perm) --:mul(256.0): brg, rgb
im = im:mul(2):add(-1)

assert(im:max()<=1,"A: badly scaled inputs")
assert(im:min()>=-1,"A: badly scaled inputs")

if opt.flip == 1 and torch.uniform() > 0.5 then
  print('input flipped') 
  im = image.hflip(im)
end

input = torch.Tensor(1, im:size(1), im:size(2), im:size(3));
input[1] = im

if opt.gpu > 0 then
  input = input:cuda()
end

output = util.deprocess_batch(netG:forward(input))
output = output:float()

image_dir = './results/facades_generation'
image.save(paths.concat(image_dir,'output.jpg'), image.scale(output[1], 256, 256))
