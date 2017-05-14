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

%mat = X;
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

%%%%%%%%%%%% Projecting images onto facespace %%%%%%%%%%%%%%%%%%%%%%%

projectimage = []; 
for i = 1 : 135
    temp = eigenfaces' * A(:,i);
    projectimage = [projectimage temp];
end

%[idx,C] = kmeans(projectimage',15);


%%%%%%%%%%%%%%%%%%%%%%%% Fisher Discriminant Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



mean_PCA = mean(projectimage,2); % Total mean in eigenspace

numberofclasses = 15;
trainingsamples = 135;

m_fisher = zeros(126,15); 
Sw = zeros(126,126); % Initialization of Within Scatter Matrix
Sb = zeros(126,126); % Initialization of Between Scatter Matrix

for i = 1 : numberofclasses
    m_fisher(:,i) = mean(( projectimage(:,((i-1)*9+1):i*9) ), 2 )';    
    
    S  = zeros(126,126); 
    for j = ( (i-1)*9+1 ) : ( i*9 )
        S = S + (projectimage(:,j)- m_fisher(:,i))*(projectimage(:,j)-m_fisher(:,i))';
    end
    
    Sw = Sw + S;                                % Within Scatter Matrix
    Sb = Sb + (m_fisher(:,i)-mean_PCA) * (m_fisher(:,i)-mean_PCA)'; % Between Scatter Matrix
end



%%%%%%%%%%%%%%%%%%%%%%%% Calculating Fisher discriminant basis's
% We want to maximise the Between Scatter Matrix, while minimising the Within Scatter Matrix.

[F_eigenvec, F_eigenval] = eig(Sb,Sw); % Cost function J = inv(Sw) * Sb
F_eigenvec = fliplr(F_eigenvec);

for i = 1 : numberofclasses-1 
    V_Fisher(:,i) = F_eigenvec(:,i); % Largest (C-1) eigen vectors of matrix J
end




for i = 1 : 135
    projectimage_Fisher(:,i) = V_Fisher' * projectimage(:,i);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%% Testing an given image %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55



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

             	InputImage = imread(currenttestfile);
		temp = InputImage;

		[irow icol] = size(temp);
		test_image = reshape(temp',irow*icol,1);
		test_image = double(test_image)- m; % Centered test image
		ProjectedTestImage = V_Fisher' * eigenfaces' * test_image; % Test image feature vector


		Euc_dist = [];
		for i = 1 : 135
		    q = projectimage_Fisher(:,i);
		    temp = (norm( ProjectedTestImage - q))^2;
		    Euc_dist = [Euc_dist temp];
		end

     
             [min_dist,ind] = min(Euc_dist);
             index = [index, ind];
             
              a = (count-2)*9;
              b = (count-3)*9;
                    if ((ind <= a) & (ind > b))
                        correct = correct+1;
                    end
        end
end


Accuracy = (correct/30)*100






