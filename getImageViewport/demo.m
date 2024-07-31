clc
path = 'H:\VRdataset\OIQA\images\';
%fid=fopen('./name_list.txt', 'r');
%C = textscan(fid, "%s");
%name_list = C{1};
name_list = dir(path);
% 去除 '.' 和 '..'
name_list = {name_list(~[name_list.isdir]).name};
% 提取数字并转化为double类型，然后升序排序
[~, idx] = sort(str2double(regexp(name_list, '\d+', 'match', 'once'))); 
name_list = name_list(idx);

FOV = pi/3; %视口大小设置
startPoint = 0;
startPointPi = deg2rad(startPoint);
lon_interval = [0 72 45 72 0];
%lon_interval = [0 72 18 72 0];
lat = deg2rad(0);

for imgID = 1 : length(name_list)
    %if imgID <= 132
    %    continue;
    %end
    savepath = strcat('./viewports_8/',name_list{imgID},'/');
    if ~exist(savepath, 'dir')
        mkdir(savepath);
    end
    img = imread([path, name_list{imgID}]);
    [hight,width,depth] = size(img);
    viewport_size = floor(FOV/(2*pi)*width);
    for lon = startPointPi : deg2rad(lon_interval(3)) : startPointPi + 2*pi - 0.1
        rad2deg(lon);
        rimg1 = imresize(cut_patch (img(:,:,1),lon ,lat,viewport_size), [512, 512]);
        rimg2 = imresize(cut_patch (img(:,:,2),lon ,lat,viewport_size), [512, 512]);
        rimg3 = imresize(cut_patch (img(:,:,3),lon ,lat,viewport_size), [512, 512]);
        rimg1 = uint8(rimg1);
        rimg2 = uint8(rimg2);
        rimg3 = uint8(rimg3);
        rimg = cat(3,rimg1,rimg2,rimg3);
        if lon > pi
           newLon = lon - 2*pi;
        elseif lon < -pi
           newLon = lon + 2*pi;
        else
           newLon = lon;
        end
        saveName = strcat(savepath, num2str(rad2deg(newLon)),'.',num2str(lat),'.png');
        imwrite(uint8(rimg),saveName);
    end
    disp(imgID);
    disp(name_list{imgID});
end
