function SR = DSen2(im10, im20, im60)
% Super-resolution of Sentinel-2 images with the DSen2 network.
% 
% THE INPUT VALUES MUST BE IN THE ORIGINAL FILE'S REFLECTANCE!!
% TYPICALLY [0 - 10000]
% 
% Inputs:
%   im10 - The highest resolution bands in the following order:
%               (B4, B3, B2, B8)
%               Dims: (x,y,4)
%   im20 - The medium resolution bands in the following order:
%               (B5, B6, B7, B8a, B11, B12)
%               Dims: (x/2,y/2,6)
%   im60 - (OPTIONAL) The low resolution bands in the following order: 
%               (B1, B9) - The band B10 is not sharpened.
%               Dims: (x/6,y/6,2)
% 
% Output:
%   SR - The super-resolved bands.
%         - If the inputs are im10 and im20, then im20 is super-resolved.
%               Output dims: (x,y,6)
%         - If the inputs include im60, then im60 is super-resolved.
%               Output dims: (x,y,2)
% 
% Remark:
%   Only use this network with Sentinel-2 data and not simulations,
%   as it is sensible to the exact spectral response of each band.
% 
% Compatibility:
%   In order to run this code you need MATLAB 2018a or newer with the
%   Neural Network Toolbox.
% 
% If you use this code please cite:
%   C. Lanaras, J. Bioucas-Dias, S. Galliani, E. Baltsavias, K. Schindler.
%   Super-resolution of Sentinel-2 images: Learning a globally applicable
%   deep neural network. ISPRS Journal of Photogrammetry and Remote Sensing,
%   Volume 146, Dec. 2018, Pages 305-319.
%   https://doi.org/10.1016/j.isprsjprs.2018.09.018

if nargin==2
    patchSize = 80;
    pad = 8;
    tic
    q_image = patches(patchSize, pad, im10, im20);
    load net20.mat
    pred = predict(net20, q_image/2000)*2000;
else
    patchSize = 192;
    pad = 12;
    tic
    q_image = patches(patchSize, pad, im10, im20, im60);
    load net60.mat
    pred = predict(net60, q_image/2000)*2000;
end

SR = full_im(pred, size(im10), pad);
toc
end


function outtt = patches(patchSize, pad, im10, im20, im60)
% Tiles the input into patches

% Check input sizes
if size(im10,3)~=4 || size(im20,3)~=6
    error(['The inputs must have 4 channels for 10m resolution and '...
        '6 channels for the 20m resolution.'])
end
if size(im10,1)/size(im20,1)~=2 || size(im10,2)/size(im20,2)~=2
    error(['Please check the size of the input images. They must be '...
        'exactly 2x bigger (per dimension).'])
end
if size(im10,1)<(patchSize-2*pad)
    error(['Minimum input size is ' num2str((patchSize-2*pad)) ' pixels'...
        ' in each dimension. Please use padarray to make your input '...
        'larger...']);
end

run_60 = false;
if nargin==5
    run_60 = true;
    % Check input sizes
    if size(im60,3)~=2
        error(['The input 60m resolution must have only 2 bands, '...
            'B1 and B9. Do not input B10.'])
    end
    if size(im10,1)/size(im60,1)~=6 || size(im10,2)/size(im60,2)~=6
        error(['Please check the size of the input 60m band images. '...
            'They must be exactly 6x bigger than the 10m (per dimension).'])
    end
    im60 = padarray(imresize(single(im60),6,'Method','bilinear'),...
        [pad, pad, 0], 'symmetric');
end

im10 = padarray(single(im10), [pad, pad, 0], 'symmetric');
im20 = padarray(imresize(single(im20),2,'Method','bilinear'),...
    [pad, pad, 0], 'symmetric');

s10 = size(im10);

patchesAlongi = floor((s10(1) - 2 * pad) / (patchSize - 2 * pad));
patchesAlongj = floor((s10(2) - 2 * pad) / (patchSize - 2 * pad));

nrPatches = (patchesAlongi + 1) * (patchesAlongj + 1);

range_i = 1:(patchSize-2*pad):(s10(1) - patchSize + 1);
range_j = 1:(patchSize-2*pad):(s10(2) - patchSize + 1);

if mod(s10(1)-2*pad,patchSize-2*pad)~=0
    range_i(end+1) = s10(1)-patchSize + 1;
end
if mod(s10(2)-2*pad,patchSize-2*pad)~=0
    range_j(end+1) = s10(2)-patchSize + 1;
end

if run_60
    outtt = single(zeros(patchSize, patchSize, 12, nrPatches));
else
    outtt = single(zeros(patchSize, patchSize, 10, nrPatches));
end

pCount = 1;
for ii=range_i
    for jj=range_j
        outtt(:,:,1:4,pCount) = im10(ii:ii+patchSize-1,jj:jj+patchSize-1,:);
        outtt(:,:,5:10,pCount) = im20(ii:ii+patchSize-1,jj:jj+patchSize-1,:);
        if run_60
            outtt(:,:,11:12,pCount) = im60(ii:ii+patchSize-1,jj:jj+patchSize-1,:);
        end
        pCount = pCount + 1;
    end
end

outtt = outtt(:,:,:,1:pCount-1);

end

function q_image = full_im(outtt, im_size, pad)
% Recomputes the full image from the patches

sz = size(outtt);

patchSize = sz(1) - 2 * pad;

x_tiles = ceil(im_size(1)/patchSize);
y_tiles = ceil(im_size(2)/patchSize);

q_image = zeros(im_size(1), im_size(2), sz(3));

current_patch = 1;
for y=1:y_tiles
    ypoint = (y - 1) * patchSize;
    if ypoint > im_size(2) - patchSize
        ypoint = im_size(2) - patchSize;
    end
    for x=1:x_tiles
        xpoint = (x - 1) * patchSize;
        if xpoint > im_size(1) - patchSize
            xpoint = im_size(1) - patchSize;
        end
        q_image(ypoint+1:ypoint+patchSize, xpoint+1:xpoint+patchSize, :) = ...
            squeeze(outtt(pad+1:sz(1)-pad, pad+1:sz(2)-pad, :, current_patch));
        current_patch = current_patch + 1;
%         imagesc(q_image(:,:,1))
    end
end
end
