clear all
close all
datapath = '/home/abhay/Desktop/Academics/ComputerVision/Assignment5/data/train';
clc;   

mainfolder = dir(datapath);
numberofsubfolders = size(mainfolder,1);
X = [];
for i=3:numberofsubfolders
    str = strcat(datapath,'/',mainfolder(i).name);
    content = dir(str);
    sizeofcurrentfolder = size(content,1);
        for j=3:sizeofcurrentfolder
            currentfile = content(j).name;
            img = imread(currentfile);
            [r c] = size(img);
            temp = reshape(img' , r*c ,1);
            X = [X temp];
        end
end

mat = X;
m = mean(X,2);
imgcount = size(X,2);

A = [];
for i=1 : imgcount
    temp = double(X(:,i)) - m;
    A = [A temp];
end

L= A' * A;

[V,D]=eig(L);


M_high = [];
for i = 1 : 135
    if( D(i,i) > 1 )
        M_high = [M_high V(:,i)];
    end
end

%eigenclass1 = A(:,1:9) * M_high;

eigenfaces = A * M_high;


for l = 1:9
    subplot(3,3,l);
    a = reshape(eigenfaces(:,l),c,r);
    B = imrotate(a,270);
    B = mat2gray(B);
    imshow(B);
end




projectimage = []; 
for i = 1 : 135
    temp = eigenfaces' * A(:,i);
    projectimage = [projectimage temp];
end

[idx,C] = kmeans(projectimage',15);

% Reading test images



testpath = '/home/abhay/Desktop/Academics/ComputerVision/Assignment5/data/test';

mainfolder_test = dir(testpath);
numberofsubfolders_test = size(mainfolder_test,1);
X = [];
index =[];
correct = 0;
ind = 0;

for i=3:numberofsubfolders_test
    str = strcat(testpath,'/',mainfolder_test(i).name);
    content_test = dir(str);
    sizeofcurrentfolder = size(content_test,1);
    count = i;
        for j=3:sizeofcurrentfolder
             currenttestfile = content_test(j).name;
             test_image = imread(currenttestfile);
             %imshow(test_image)
             [r c] = size(test_image);
             temp = reshape(test_image',r*c,1); % creating (MxN)x1 image vector from the 2D image
             temp = double(temp)-m;             % mean subtracted vector
             projtestimg = eigenfaces' * temp;  % projection of test image onto the facespace
            
             euclide_dist = [];
             for i = 1 : 135
                temp = (norm(projtestimg-projectimage(:,i)))^2;
                euclide_dist = [euclide_dist temp];
             end
     
             [min_dist,ind] = min(euclide_dist);
             index = [index, ind];
             
              a = (count-2)*9;
              b = (count-3)*9;
                    if ((ind <= a) & (ind > b))
                        correct = correct+1;
                    end
        end
end

Accuracy = (correct/30)*100









