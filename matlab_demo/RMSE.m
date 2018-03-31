function rms = RMSE(y1,y2)
    s = (double(y1)-double(y2)).^2;
    rms = sqrt(mean(s(:)));
    disp(['RMSE: ' num2str(rms)])
end