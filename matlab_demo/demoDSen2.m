% ----------------------------------------------------------------------- %
% DSen2: Deep Sentinel-2
% 
% Super-resolution of Sentinel-2 images: Learning a globally applicable
% deep neural network
% 
% C. Lanaras, J. Bioucas-Dias, S. Galliani, E. Baltsavias, K. Schindler.
% ISPRS Journal of Photogrammetry and Remote Sensing
% Volume 146, Dec. 2018, Pages 305-319
% https://doi.org/10.1016/j.isprsjprs.2018.09.018
% 
% ----------------------------------------------------------------------- %
% 
% DEMO
%
% This demo is only available with MATLAB 2018a or newer. You can speed up
% the computation by using the Parallel Computing Toolbox with your GPU.
% 
% For computational efficiency only the DSen2 network is implemented (and
% not the very deep (VDSen2).
% 
% The images used are from the test set.  


% Siberia, same area of Fig. 8 in the paper
disp('Siberia')
load('../data/S2B_MSIL1C_20170725_T43WFQ.mat')
SR20 = DSen2(im10, im20);
% Evaluation against the ground truth on the 20m resolution bands (simulated)
disp('DSen2:')
RMSE(SR20,imGT);
disp('Bicubic:')
RMSE(imresize(im20,2),imGT);
figure(1)
imagesc(SR20(:,:,2))
axis off
colorbar
title 'Super-resolved band B6'
figure(2)
imagesc(abs(SR20(:,:,5)-imGT(:,:,5)),[0 200])
axis off
colorbar
title 'Absolute differences to the GT, band B11'

% South Africa, same area of Fig. 9 in the paper
disp('S. Africa')
load('../data/S2A_MSIL1C_20171028_T34HCH.mat')
SR60 = DSen2(im10, im20, im60);
% Evaluation against the ground truth on the 60m resolution bands (simulated)
disp('DSen2:')
RMSE(SR60,imGT);
disp('Bicubic:')
RMSE(imresize(im60,6),imGT);
figure(3)
imagesc(abs(SR60(:,:,2)-imGT(:,:,2)),[0 200])
axis off
colorbar
title 'Absolute differences to the GT, band B9'


% New York, same area of Fig. 10 (bottom) in the paper
disp('New York')
load('../data/S2B_MSIL1C_20170928_T18TWL.mat')
SR20 = DSen2(im10, im20);
% Evaluation against the ground truth on the 20m resolution bands (simulated)
disp('DSen2:')
RMSE(SR20,imGT);
disp('Bicubic:')
RMSE(imresize(im20,2),imGT);


% Malmö, Sweden, same area of Fig. 10 (top) in the paper
disp('Malmö, no ground truth')
load('../data/S2A_MSIL1C_20170527_T33UUB.mat')
SR20 = DSen2(im10, im20);
SR60 = DSen2(im10, im20, im60);
% No ground truth available, no simulation. Comparison to the low-res input
figure(4)
subplot(1,2,1)
imagesc(im60(:,:,1), [min(min(im60(:,:,1))), max(max(im60(:,:,1)))])
axis off square
title 'Band B1, input 60m'
subplot(1,2,2)
imagesc(SR60(:,:,1),[min(min(im60(:,:,1))), max(max(im60(:,:,1)))])
axis off square
title 'Band B1, 10m super-resolution'

figure(5)
subplot(1,2,1)
imagesc(im20(:,:,2), [min(min(im20(:,:,2))), max(max(im20(:,:,2)))])
axis off square
title 'Band B6, input 20m'
subplot(1,2,2)
imagesc(SR20(:,:,2),[min(min(im20(:,:,2))), max(max(im20(:,:,2)))])
axis off square
title 'Band B6, 10m super-resolution'


% Shark bay, Australia, same area of Fig. 10 (middle) in the paper
disp('Shark Bay, no ground truth')
load('../data/S2B_MSIL1C_20171022_T49JGM.mat')
SR20 = DSen2(im10, im20);
SR60 = DSen2(im10, im20, im60);

% Stretching the image for better visualization
imSR = SR60(:,:,[1 2 1]);
im60s = im60(:,:,[1 2 1]);
for i=1:3
    a = prctile(reshape(imSR(:,:,i),[],1),1);
    b = prctile(reshape(imSR(:,:,i),[],1),99);
    imSR(:,:,i) = (imSR(:,:,i)-a)/(b-a);
    im60s(:,:,i) = (im60s(:,:,i)-a)/(b-a);
end
figure(6)
subplot(1,2,1)
imshow(im60s)
title(['Color composite (B1,B9,B1)' newline '60m input'])
subplot(1,2,2)
imshow(imSR)
title(['Color composite (B1,B9,B1)' newline '10m super-resolution'])


% Stretching the image for better visualization
imSR = SR20(:,:,[6 4 1]);
im20s = im20(:,:,[6 4 1]);
for i=1:3
    a = prctile(reshape(imSR(:,:,i),[],1),1);
    b = prctile(reshape(imSR(:,:,i),[],1),99);
    imSR(:,:,i) = (imSR(:,:,i)-a)/(b-a);
    im20s(:,:,i) = (im20s(:,:,i)-a)/(b-a);
end
figure(7)
subplot(1,2,1)
imshow(im20s)
title(['Color composite (B12,B8a,B5)' newline '20m input'])
subplot(1,2,2)
imshow(imSR)
title(['Color composite (B12,B8a,B5)' newline '10m super-resolution'])
