function main_v2(varargin)
% This function calculates the minimum average precision of 
% document image patches. It calculates deep-features aswell as hand crafted features 
%====================================================================================
			%%%%%% Input parameters %%%%%%%%%%
%====================================================================================
setup
global allowdedRatio
mode = 'train'; %train or test
%dataHomeLoc = ['/ssd_scratch/cvit/jobinkv/']; % data path
%dfolder_locNo = 4;
%datafolder = strsplit(dataHomeLoc,'/');
imdb.modelDir='/home/jobinkv/models/'; % model path
imdb.pretrainedNet ='imagenet-vgg-m.mat';% 'net-epoch-5.mat';% 'imagenet-vgg-m.mat';%'imagenet-vgg-verydeep-19.mat'; %'imagenet-vgg-m.mat';
imdb.shuffle=false; %shuffle image patches across images
%allowdedRatio = 0.9;
%imdb.patchSize=224;
%imdb.stride =100;
imdb.feature = 'dcnn'; % {'dcnn', 'rcnn', 'hog', 'gabor'}
imdb.expDir = ['/home/jobinkv/classification_rvlmodels/exp2/',imdb.feature]; % 9--> 1:8k,10-->1:end
opts.useGpu = true;
imdb.pcaLen = 2000;
imdb.pca=true;
%imdb.dataset = 'div'; % div and iam
%-- slic paprameters-------
%imdb.imageScale = 4;% resize to 1/imageScale;
%imdb.regionSize = 10;
%imdb.regularizer = 0.01;
%imdb.patchSize = 28;
tsne = true;
%imdb = vl_argparse(imdb, varargin) ;
%-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=
%====================================================================================
			%%%%%% Internal setups  %%%%%%%%%%
%====================================================================================
%dfoldr = datafolder{dfolder_locNo};
%if (exist(['/tmp/',dfoldr])~=7)
%        disp(['copying the dataset',dfoldr,' to the /tmp folder ...'])
%        disp('please wait ...')
%        command = ['cp -r ',dataHomeLoc,' /tmp/'];
%        [status,cmdout] = system(command,'-echo');
%        disp('copyed!')
%end
%global allowdedRatio
imdb.dataLoc =['/ssd_scratch/cvit/jobinkv/'];
imdb = get_database(imdb);
imdb.resultPath = fullfile(imdb.expDir, sprintf('results.mat')) ;
imdb.featurePath = fullfile(imdb.expDir, sprintf('features.mat')) ;
imdb.encoderPath = fullfile(imdb.expDir, sprintf('encoder.mat')) ;
fprintf('Your output will appear at %s\n',imdb.expDir);
fprintf('The data path location is %s\n',imdb.dataLoc);
imdb.encoder=[];
imdb.net = [];
dbstop if error
opts.model=[imdb.modelDir,imdb.pretrainedNet];
switch imdb.pretrainedNet
case {'imagenet-vgg-m.mat'}
opts.layer.rcnn = 19;
opts.layer.dcnn = 13;
case {'imagenet-vgg-verydeep-19.mat'}
opts.layer.dcnn = 39;
end
if exist(imdb.resultPath)
    load(imdb.resultPath)
	disp(['The current setup is already computed please check folder ',imdb.expDir])
	fprintf('mAP train: %.1f, test: %.1f\n', ...
  	mean(info.train.ap)*100, ...
  	mean(info.test.ap)*100);
%	keyboard
	return;
end
%====================================================================================
			%%%%%% Load the network and encoder %%%%%%%%%%
%====================================================================================
if (exist('net')~=1)
	switch imdb.feature
  		case {'rcnn'}
    			imdb.net = load(opts.model) ;
    			imdb.net.layers = imdb.net.layers(1:opts.layer.rcnn) ;
    			if opts.useGpu
      				imdb.net = vl_simplenn_move(imdb.net, 'gpu') ;
      				imdb.net.useGpu = true ;
    			else
      				imdb.net = vl_simplenn_move(imdb.net, 'cpu') ;
      				imdb.net.useGpu = false ;
    			end
			%imdb.encoder = 'not required';
			if exist(imdb.encoderPath)
	            		load(imdb.encoderPath,'encoder');
        	    		imdb.encoder = encoder;
				clear eigvector;
			end
  		case {'dcnn'}
    			imdb.net = load(opts.model) ;
    			imdb.net.layers = imdb.net.layers(1:opts.layer.dcnn) ;
    			if opts.useGpu
      				imdb.net = vl_simplenn_move(imdb.net, 'gpu') ;
      				imdb.net.useGpu = true ;
    			else
      				imdb.net = vl_simplenn_move(imdb.net, 'cpu') ;
      				imdb.net.useGpu = false ;
    			end
			if exist(imdb.encoderPath)
	            		load(imdb.encoderPath,'encoder');
        	    		imdb.encoder = encoder;
			end
        	case 'gabor'% this functionality is not ready
            		net = gaborFilterBank(5,8,39,39);
       		 case 'dsift'% this functionality is not ready
	    	if (exist(imdb.encoderPath, 'file') == 2)
	    		load(imdb.encoderPath, 'encoder');
            		imdb.encoder = encoder;
		end
    end
end
if (exist(imdb.expDir)~=7)
    mkdir(imdb.expDir)
end
% check the encoder is present and create it if necessary
%====================================================================================
			%%%%%% Creator encoder %%%%%%%%%%
%====================================================================================
imdb.batchSize = 1000;
switch mode
case 'train'
cnt=1;
if isempty(imdb.encoder)
    disp('no encoder found!');
    for i =1:imdb.batchSize:numel(imdb.Sets{1}.imagePath)
            tic
		imdb.pca=false;
            code{cnt} = extractFeature(imdb,i,'mode','train');
	    cnt=cnt+1;
            toc
    end
    %code = cat(2, code{:}) ;
	switch imdb.feature
		case 'dcnn'
    encoder = encoder_train_from_segments(code,'type',imdb.feature,'model',imdb.net);
    save(imdb.encoderPath, 'encoder');
    disp(['encoder created and saved to ',imdb.encoderPath]);
    imdb.encoder = encoder;
    clear code;
		case 'rcnn'
    	k=1;
    	for i=1:numel(code)
        	for j=1:numel(code{i}.feature)
            	encode{k}=code{i}.feature{j};
            	k=k+1;
        	end
	end
	clear code
	encode=cat(2,encode{:});
	[r,c]=size(encode);
	m=mean(encode')';
	d=encode-repmat(m,1,c);
	co=1 / (c-1)*d*d';
	[eigvector,eigvl]=eig(co);
	encoder.eigvector = eigvector;
	encoder.means = m;
	imdb.encoder = encoder;	
    	save(imdb.encoderPath, 'encoder');
    	disp(['encoder created and saved to ',imdb.encoderPath]);
	clear encode eigvector eigvl;
	imdb.pca = true;
	end
end
%====================================================================================
			%%%%%% train and test %%%%%%%%%%
%====================================================================================
cnt=1;
for i = 1:imdb.batchSize:numel(imdb.Sets{1}.imagePath)
        tic
        out{cnt} = extractFeature(imdb,i,'mode','train');
	cnt=cnt+1;
        toc
end
for i = 1:imdb.batchSize:numel(imdb.Sets{3}.imagePath)
        tic
        out{cnt} = extractFeature(imdb,i,'mode','val');
	cnt=cnt+1;
        toc
end
for i = 1:imdb.batchSize:numel(imdb.Sets{2}.imagePath)
        tic
        out{cnt} = extractFeature(imdb,i,'mode','test');
	cnt=cnt+1;
        toc
end
disp(['No of training images = ',cnt*imdb.batchSize])
Noimg= cnt*imdb.batchSize
case 'test'
disp('entered test mode')
cnt=1;
for i = 1:imdb.batchSize:numel(imdb.Sets{2}.imagePath)
        tic
        out{cnt} = extractFeature(imdb,i,'mode','test');
	cnt=cnt+1;
%	if cnt>10
%	break 
%	end
        toc
end
disp(['No of testing images = ',cnt*imdb.batchSize])
end
% create the feature matrics
if exist(imdb.resultPath)
    load(imdb.resultPath)
else
%    numberOfimg = numel(out);
 %   numberPatch = numel(out{1}.feature);
  %  encode=cell(size(numberPatch*numberOfimg)) ;
   % labelClass=cell(size(numberPatch*numberOfimg)) ;
   % sets = cell(size(numberPatch*numberOfimg)) ;
    k=1;
    for i=1:numel(out)
        for j=1:size(out{i}.pca,2)
            encode{k}=out{i}.pca(:,j);
            k=k+1;
        end
        labelClass{i}=out{i}.labels;
	lbForTsne{i} = out{i}.labeID;
        sets{i}=out{i}.set;
    end
	clear out;
    encode=cat(2,encode{:});
    imdbs.segments.label=cat(2,labelClass{:});
   % tsneLabels = cat(2,lbForTsne{:});
    %tsneData.feature = encode';
    %tsneData.Labels = cat(2,lbForTsne{:});
    %save('tsneData.mat','tsneData')
   % keyboard
   % tsneFeature = double(encode');
    
   % mappedX = tsne(tsneFeature, [], 2, 100,64);
   % save('tsneLabels.mat','tsneLabels')
   % save('mappedX.mat','mappedX')
    train_test_split = cat(2,sets{:});
    if (imdb.shuffle)
	disp('suffling hapenings')
    shuffles_set = train_test_split(randperm(length(train_test_split)));
    imdbs.segments.set=shuffles_set;
    else
	disp('suffling is not  hapenings')
    imdbs.segments.set = cat(2,sets{:});
    end
	switch mode
		case 'train'
    			info = traintest_modified(imdbs, encode);
		case 'test'
    			info = test_modified(imdbs, encode);
	end
    save(imdb.resultPath,'info');
    disp(['results and svm model saved to ',imdb.resultPath]);
end
% dataloc = fullfile(imdb.dataLoc,'image');
disp('yahooooo')

%--------------------------------------------------
function [out] = extractFeature(imdb,ii,varargin)
opts.mode='train';
opts = vl_argparse(opts, varargin) ;
no_labels = length(imdb.classes.name);
id=10;
switch opts.mode
	case 'train'
		id=1;
	case 'test'
		id=2;
	case 'val'
		id=3;
end
bachGround_patch =[];% cell(1,imdb.batchSize) ;
labels=ones(no_labels,imdb.batchSize)*-1;
labeID =[];
j=1;
fileID = fopen('Check1.txt','a');
for i=ii:ii+imdb.batchSize-1
	bachGround_patch{j} = imread(imdb.Sets{id}.imagePath{i});
	label_id = imdb.Sets{id}.label(i); % +1 for matlab indices
    	temps = ones(no_labels,1)*-1;
	temps(label_id,1) =1;
	labels(:,j)=temps;
 	labeID(1,j) =label_id;
	j=j+1;
	fprintf(fileID,'%15s %d %d\n',imdb.Sets{id}.imagePath{i}, label_id,i);
end
fclose(fileID);
%fileID = fopen(fullfile(imdb.maskDir,[imdb.images.name{ii},'.txt']));
%C = textscan(fileID,'%d %d %d %d %s');
%numberOfObjects = numel(C{1});
%bachGround_patch = cell(1,numberOfObjects);
%labels=ones(no_labels,numberOfObjects)*-1;
%labeID =[];

 %       if numberOfObjects>=1
  %              for j=1:numberOfObjects
   %                 xmin = C{1,1}(j);
    %                ymin  = C{1,2}(j);
     %               xmax = C{1,3}(j);
      %              ymax = C{1,4}(j);
       %             labl = C{1,5}(j);
        %            bachGround_patch{j} = imcrop(image,[xmin, ymin, xmax-xmin,ymax-ymin]);
         %           label_id = mapLabel(labl);
	%	    temps = ones(no_labels,1)*-1;;
	%	    temps(label_id,1) =1;
	%	    labels(:,j)=temps;
 	%	    labeID(1,j) =label_id;
	%	    setss(1,j) = imdb.images.set(ii);
        %	end
	%end
%fclose(fileID);
%image =imresize(image1,1/imdb.imageScale);

% slic
%segments = vl_slic(single(image), imdb.regionSize, imdb.regularizer, 'verbose') ;
%imdb.no_patches = max(max(segments));
%if (size(image1,3)==3)
 %   image = rgb2gray(image1);
%elseif (size(image1,3)==1)
 %   image = image1;
%else
 %   disp('unknown format image')
%end
%mask1 = imread(fullfile(imdb.maskDir,[imdb.images.vocid{ii},'.png']));
%mask = imgDownSample(mask1(:,:,3),imdb.imageScale);
%height = size(image,1);
%width  = size(image,2);
%bachGround_patch = cell(1,imdb.no_patches) ;
%cnt=0;
%labels=ones(4,imdb.no_patches)*-1;
%labels =[];
%labeID =[];
%for i=1:imdb.no_patches
%    [row,col,v] = find(segments==i);
%    mean_row = floor(mean(row));
%    mean_col = floor(mean(col));
%    y_min = mean_row - imdb.patchSize/2;
%    x_min = mean_col - imdb.patchSize/2;
%    if (x_min>0 && y_min>0 && y_min+imdb.patchSize<height && x_min+imdb.patchSize<width)
%	   tval = zeros(1,size(row,1));
%	    for j=1:size(row,1)
%	    	tval(1,j) = mask(row(j,1),col(j,1),1);
%	    end
%	switch imdb.dataset
%		case {'div'}	
%	    		label_id = mapLabel(mode(tval));
%		case {'iam'}
%	    		label_id = mode(tval);
%	end
%
 %   cnt=cnt+1;
  %  temps = [1; 1; 1; 1]*-1;
%    temps(label_id,1) =1;
%    labels(:,cnt)=temps;
 %   labeID(1,cnt) =label_id;
%    setss(1,cnt) = imdb.images.set(ii);
%    bachGround_patch{cnt} = imcrop(image,[x_min y_min  imdb.patchSize imdb.patchSize]);
%     labeel1= mode(tval);
    %data.images{cnt} = imresize(tmpimg,[224 224]);
   % data.labels(cnt) = mapLabel(mode(tval)); %gt(mean_row*imageScale,mean_col*imageScale,3);
   % data.slicId{cnt} = i;
%    end 
%end
        

%code = get_rcnn_features_modified(net, bachGround_patch);
switch imdb.feature
    case 'rcnn'
        out.feature=get_rcnn_features_modified(imdb.net, bachGround_patch);
    case 'dcnn'
        out.feature = get_dcnn_features_modified(imdb.net,bachGround_patch,'encoder',imdb.encoder);
    case 'dsift'
        out.feature = get_dcnn_features_modified([],bachGround_patch,'useSIFT', true,'encoder', imdb.encoder,'numSpatialSubdivisions',1,'maxNumLocalDescriptorsReturned', 500);
    case 'gabor'
        out.feature = get_gabor_feature(imdb.net,bachGround_patch);
end
if imdb.pca
featu = cat(2,out.feature{:});
featu = featu-repmat(imdb.encoder.means,1,imdb.batchSize);
out.pca =  imdb.encoder.eigvector(:,1:imdb.pcaLen)'*featu;
end

clear bachGround_patch;
out.labels = labels;
out.set = ones(1,imdb.batchSize)*id;
out.labeID = labeID;
%--------------------------------------------

function out = mapLabel(id)
switch id{1}
	case 'figure'
		out = 1; 
	case 'table' 
		out = 2; 
	case 'section' 
		out = 3;
	case 'caption' 
		out = 4; 
	case 'list' 
		out = 5; 
	case 'text' 
		out = 6; 
	otherwise
		out = 7;
end
%---------------------------------------------
function id = findlabelId(label_ofpatch)
%-------------------------------------
back_ground = sum(sum(label_ofpatch(:,:,1)));
graphics = sum(sum(label_ofpatch(:,:,2)));
text = sum(sum(label_ofpatch(:,:,3)));
total_val = back_ground+graphics+text;
bg_percentage = back_ground/total_val;
gr_percentage = graphics/total_val;
tx_percentage = text/total_val;
if(bg_percentage>=allowdedRatio)
    id =1;
elseif(gr_percentage>=allowdedRatio)
    id=2;
elseif(tx_percentage>=allowdedRatio)
    id=3;
else
    id=4;
end

%---------------------------------------------
function id = findlabelId_centre(label_ofpatch)
width = floor(size(label_ofpatch,1)/2);
height = floor(size(label_ofpatch,2)/2);
temp = label_ofpatch(width,height,:);
if(temp(:,:,1)>=(temp(:,:,2)+temp(:,:,3)))
    id =1;
elseif(temp(:,:,2)>=(temp(:,:,1)+temp(:,:,3)))
    id=2;
else
    id=3;
end

%-------------------------------------------------
function imdb = get_database(imdb,varargin)
opts.seed = 1 ;
opts = vl_argparse(opts, varargin) ;
imdb.imageDir = fullfile(imdb.dataLoc, 'images') ;
%imdb.maskDir = fullfile(imdb.dataLoc, 'dsse-200') ;
imdb.classes.name={...
'letter'
'memo'
'email'
'file folder'
'form'
'handwritten'
'invoice'
'advertisement'
'budget'
'news article'
'presentation' 
'scientific publication'
'questionnaire' 
'resume'
'scientific report'
'specification'};
numClass = length(imdb.classes.name);
%imageFiles = dir(fullfile(imdb.imageDir, '*.jpg'));
%imdb.images.name = {imageFiles.name};
%numImages = length(imdb.images.name);
% finding the number of patches
%image = imread(fullfile(imdb.imageDir,imdb.images.name{1}));
%height = size(image,1);
%width  = size(image,2);
%no_xmoves = floor( (width-imdb.patchSize)/imdb.stride + 1);
%no_ymoves = floor((height-imdb.patchSize)/imdb.stride + 1);
%imdb.no_patches = floor(no_ymoves*no_xmoves);
%imdb.images.label = ones(numClass, numImages*imdb.no_patches)*-1;
%imdb.images.vocid = cellfun(@(S) S(1:end-4), imdb.images.name, 'UniformOutput', false);
%imdb.images.set = zeros(1, numImages);
%imdb.images.id = 1:numImages;
% Loop over images and record the imag sets
imageSets = {'train','test','val'};
for s = 1:length(imageSets),
    %imageSetPath = fullfile(imdb.dataLoc, 'labels', sprintf('%s.txt',imageSets{s}));
    imageSetPath = fullfile('./labels_small/', sprintf('%s.txt',imageSets{s}));
    fileID = fopen(imageSetPath);
    C = textscan(fileID,'%s %d');
    numberOfImages = numel(C{1});
    if numberOfImages>=1
         for j=1:numberOfImages
		tempLoc = C{1,1}(j);
                imdb.Sets{s}.imagePath{j} = [imdb.imageDir,'/',tempLoc{1}];
                imdb.Sets{s}.label(j) = C{2}(j)+1;
          end
    end
    fclose(fileID);

%keyboard	
    %gtids1 = textread(imageSetPath,'%s');
    %gtids = cellfun(@(S) S(1:end-4), gtids1, 'UniformOutput', false);
    %[membership, loc] = ismember(gtids, imdb.images.vocid);
    %assert(all(membership));
    %imdb.images.set(loc) = s;
end
% Remove images not part of train, val, test sets
%valid = ismember(imdb.images.set, 1:length(imageSets));
%imdb.images.name = imdb.images.name(imdb.images.id(valid));
%imdb.images.id = 1:numel(imdb.images.name);
%imdb.images.label = imdb.images.label(:, valid);
%imdb.images.set = imdb.images.set(valid);
%imdb.images.vocid = imdb.images.vocid(valid);
%------------------------------------------------------

function code = get_gabor_feature(net, im)
%keyboard
code = cell(numel(im));
for l=1:numel(im)
    code{l} = gaborFeatures(im{l},net,25,25) ;
end

%-----------============================================
% -------------------------------------------------------------------------
function encoder = encoder_train_from_segments(code, varargin)
% -------------------------------------------------------------------------
opts.type = 'rcnn' ;
opts.model = '' ;
opts.layer = 0 ;
opts.useGpu = false ;
opts.regionBorder = 0.05 ;
opts.numPcaDimensions = +inf ;
opts.numSamplesPerWord = 1000 ;
opts.whitening = false ;
opts.whiteningRegul = 0 ;
opts.renormalize = false ;
opts.numWords = 64 ;
opts.numSpatialSubdivisions = 1 ;
opts.encoderType = 'fv';
opts = vl_argparse(opts, varargin) ;

%initialize ?!
encoder.projection = 1 ;
encoder.projectionCenter = 0 ;

encoder.encoderType = opts.encoderType;

encoder.type = opts.type ;
encoder.regionBorder = opts.regionBorder ;
switch opts.type
    case {'dcnn', 'dsift'}
        encoder.numWords = opts.numWords ;
        encoder.renormalize = opts.renormalize ;
        encoder.numSpatialSubdivisions = opts.numSpatialSubdivisions ;
end

switch opts.type
    case {'rcnn', 'dcnn'}
        encoder.net = opts.model ;
        encoder.net.layers = encoder.net.layers(1:opts.layer) ;
        if opts.useGpu
            encoder.net = vl_simplenn_move(encoder.net, 'gpu') ;
            encoder.net.useGpu = true ;
        else
            encoder.net = vl_simplenn_move(encoder.net, 'cpu') ;
            encoder.net.useGpu = false ;
        end
end

switch opts.type
    case 'rcnn'
        return ;
end

% Step 0: sample descriptors
fprintf('%s: getting local descriptors to train GMM\n', mfilename) ;
%code = encoder_extract_for_segments(encoder, imdb, segmentIds) ;
descrs = cell(1, numel(code)*numel(code{1}.feature)) ;
numImages = numel(code)*numel(code{1}.feature);
numDescrsPerImage = floor(opts.numWords * opts.numSamplesPerWord / numImages);
job=1;
for i=1:numel(code)
	for j=1:numel(code{i}.feature)
		descrs{job} = vl_colsubset(code{i}.feature{j}, numDescrsPerImage) ;
		job=job+1;
	end
end
descrs = cat(2, descrs{:}) ;
fprintf('%s: obtained %d local descriptors to train GMM\n', ...
    mfilename, size(descrs,2)) ;


% Step 1 (optional): learn PCA projection
if opts.numPcaDimensions < inf || opts.whitening
    fprintf('%s: learning PCA rotation/projection\n', mfilename) ;
    encoder.projectionCenter = mean(descrs,2) ;
    x = bsxfun(@minus, descrs, encoder.projectionCenter) ;
    X = x*x' / size(x,2) ;
    [V,D] = eig(X) ;
    d = diag(D) ;
    [d,perm] = sort(d,'descend') ;
    d = d + opts.whiteningRegul * max(d) ;
    m = min(opts.numPcaDimensions, size(descrs,1)) ;
    V = V(:,perm) ;
    if opts.whitening
        encoder.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
    else
        encoder.projection = V(:,1:m)' ;
    end
    clear X V D d ;
else
    encoder.projection = 1 ;
    encoder.projectionCenter = 0 ;
end
descrs = encoder.projection * bsxfun(@minus, descrs, encoder.projectionCenter) ;
if encoder.renormalize
    descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
end

encoder.encoderType = opts.encoderType;

% Step 2: train Encoder

switch (opts.encoderType)
    case {'bovw', 'vlad', 'llc'}
        encoder.words = vl_kmeans(descrs, opts.numWords, 'verbose', 'algorithm', 'ann') ;
        encoder.kdtree = vl_kdtreebuild(encoder.words, 'numTrees', 2) ;
        
    case {'fv'}
        
        v = var(descrs')' ;
        [encoder.means, encoder.covariances, encoder.priors] = ...
            vl_gmm(descrs, opts.numWords, 'verbose', ...
            'Initialization', 'kmeans', ...
            'CovarianceBound', double(max(v)*0.0001), ...
            'NumRepetitions', 1);
        
end
%-------------------------------------------------
function out = imgDownSample(x,imageScale)
Y = downsample(x,imageScale,0);
out = (downsample(Y',imageScale,0))';
%------------------------------------------------------------------------
%===============================================
