function [hog,valid_points] = my_vlhogptws(im, points, patchsize, ncells)

imsz = size(im);

halfsize = (single(patchsize) - mod(single(patchsize),2))/2;
cellsize = (patchsize - mod(patchsize,ncells))/ncells;

npoints = points.Count;

roi = [1 1 patchsize*ones(1,2)]; % [c r w h] [x y w h]

validPointIdx   = zeros(1,npoints,'uint32');
validPointCount = zeros(1,'uint32');

hog = zeros(npoints,36*ncells^2,'single');

for i = 1:npoints
    
    roi(1:2) = round(points.Location(i,:)) - halfsize;
    
    if (all(roi(1:2) >= 1) && roi(2)+roi(4)-1 <= imsz(1) && roi(1)+roi(3)-1 <= imsz(2))    
    
        im_tmp = im(roi(2):roi(2)+roi(4)-1,roi(1):roi(1)+roi(3)-1);
        hogi = vl_hog(im_tmp, cellsize, 'variant', 'dalaltriggs');
        
        validPointCount = validPointCount + 1;
        
        hog(validPointCount,:) = hogi(:);
        
        validPointIdx(validPointCount) = i; % store valid indices
        
    end;
        
end;

hog = hog(1:validPointCount,:);
validPointIdx = validPointIdx(1:validPointCount);
valid_points = cornerPoints(points.Location(validPointIdx,:),'Metric',...
    points.Metric(validPointIdx,:));

end