function mainCbd()
%function mainCbd(featur)
%
%
% This function calculates the mean average precision of 
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
imdb.feature = 'drcnn'; % {'dcnn', 'rcnn','drcnn', 'hog', 'gabor'}
imdb.expDir = ['/home/jobinkv/toboco/exp01/',imdb.feature]; % exp02--> 16 gmm, exp03---> 32 gmm
opts.useGpu = true;
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
imdb.layer.rcnn = 19;
imdb.layer.dcnn = 13;
case {'imagenet-vgg-verydeep-19.mat'}
imdb.layer.rcnn = 39;
imdb.layer.dcnn = 35;
end
if (exist(imdb.resultPath) && strcmp(mode,'train'))
    load(imdb.resultPath)
	disp(['The current setup is already computed please check folder ',imdb.expDir])
	fprintf('mAP train: %.1f, test: %.1f\n', ...
  	mean(info.train.ap)*100, ...
  	mean(info.test.ap)*100);
	return;
end
%====================================================================================
			%%%%%% Load the network and encoder %%%%%%%%%%
%====================================================================================
if (exist('net')~=1)
	switch imdb.feature
  		case {'rcnn'}
    			imdb.net = load(opts.model) ;
    			if opts.useGpu
      				imdb.net = vl_simplenn_move(imdb.net, 'gpu') ;
      				imdb.net.useGpu = true ;
    			else
      				imdb.net = vl_simplenn_move(imdb.net, 'cpu') ;
      				imdb.net.useGpu = false ;
    			end
    		%	imdb.net.layers = imdb.net.layers(1:opts.layer.rcnn) ;
			imdb.encoder = 'not required';
  		case {'dcnn','drcnn'}
    			imdb.net = load(opts.model) ;
    		%	imdb.net.layers = imdb.net.layers(1:opts.layer.dcnn) ;
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
        	case 'gabor' % this functionality is not ready
            		net = gaborFilterBank(5,8,39,39);
       		 case 'dsift' % this functionality is not ready
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
imdb.batchSize = 1741;
switch mode
case 'train'
cnt=1;
if isempty(imdb.encoder)
    disp('no encoder found!');
    for i =1:imdb.batchSize:2000 %numel(imdb.Sets{1}.imagePath)
            tic
            code{cnt} = extractFeatureForEncode(imdb,i,'mode','train');
	    cnt=cnt+1;
            toc
    end
    %code = cat(2, code{:}) ;
    encoder = encoder_train_from_segments(code,'type',imdb.feature,'model',imdb.net);
    save(imdb.encoderPath, 'encoder');
    disp(['encoder created and saved to ',imdb.encoderPath]);
    imdb.encoder = encoder;
    clear code;
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
case 'test'
disp('entered test mode')
end
    k=1;
    for i=1:numel(out)
        for j=1:numel(out{i}.feature)
            encode{k}=out{i}.feature{j};
            k=k+1;
        end
        labelClass{i}=out{i}.labels;
	lbForTsne{i} = out{i}.labeID;
        sets{i}=out{i}.set;
    end
	clear out;
%tains = [randsample(1:231,100):randsample(231:600,100):randsample(231:600,100)]
    imdbs.segments.set = cat(2,sets{:});
    encode=cat(2,encode{:});
    imdbs.segments.label=cat(2,labelClass{:});
    tsneLabels = cat(2,lbForTsne{:});
	% creating training set
	for i=1:10
		selected = randsample(find(tsneLabels==i),100);
		for j=1:100
			imdbs.segments.set(selected(j))=2;
		end
	end
    %tsneData.feature = encode';
    %tsneData.Labels = cat(2,lbForTsne{:});
    %save('tsneData.mat','tsneData')
   % keyboard
   % tsneFeature = double(encode');
    
   % mappedX = tsne(tsneFeature, [], 2, 100,64);
   % save('tsneLabels.mat','tsneLabels')
   % save('mappedX.mat','mappedX')
    if (imdb.shuffle)
	disp('suffling hapenings')
    train_test_split = cat(2,sets{:});
    shuffles_set = train_test_split(randperm(length(train_test_split)));
    imdbs.segments.set=shuffles_set;
    else
	disp('suffling is not  hapenings')
    %imdbs.segments.set = cat(2,sets{:});
    end
	switch mode
		case 'train'
    			info = traintest_modified(imdbs, encode);
    			save(imdb.resultPath,'info');
    			disp(['results and svm model saved to ',imdb.resultPath]);
		case 'test'
			if exist(imdb.resultPath)
			    load(imdb.resultPath)
			else
				disp('train the model first')
			end
    			classes = getTested(encode,info);
			temp=cat(1,classes{:});
			[val,imdb.Sets{2}.pred]=max(temp);
			testres = acuracyCal(imdb.Sets{2}.label,imdb.Sets{2}.pred);
			writeOuts(imdb);
	end
% dataloc = fullfile(imdb.dataLoc,'image');
disp('yahooooo')
%------------------------------------------------
function writeOuts(imdb)
fileID = fopen([imdb.feature,'.txt'],'w');
for i=1:size(imdb.Sets{2}.pred,2)
	temps=split(imdb.Sets{2}.imagePath{i},'kv/images');
fprintf(fileID,'%6s\t%s\t%s\n',temps{2},getLabel(imdb.Sets{2}.label(i)+1),getLabel(imdb.Sets{2}.pred(i)));
end
fclose(fileID);
!scp *cnn.txt jobinkv@10.2.16.195:/var/www/cgi-bin/dv3/
%------------------------------------------------
function scores = getTested(psi,info)
w = info.w ;
b = info.b;
scores=[];
for c=1:numel(info.b) %numel(info.classes)
  pred = w{c}'*psi + b{c} ;
  scores{c} = pred ;
end
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
bachGround_patch = cell(1,imdb.batchSize) ;
labels=ones(no_labels,imdb.batchSize)*-1;
labeID =[];
j=1;
for i=ii:ii+imdb.batchSize-1
	bachGround_patch{j} = imread(imdb.Sets{id}.imagePath{i});
	label_id = imdb.Sets{id}.label(i); % +1 for matlab indices
    	temps = ones(no_labels,1)*-1;
	temps(label_id,1) =1;
	labels(:,j)=temps;
 	labeID(1,j) =label_id;
	j=j+1;
end
switch imdb.feature
    case 'rcnn'
    	imdb.net.layers = imdb.net.layers(1:imdb.layer.rcnn) ;
        out.feature=get_rcnn_features_modified(imdb.net, bachGround_patch);
    case 'dcnn'
    	imdb.net.layers = imdb.net.layers(1:imdb.layer.dcnn) ;
        out.feature = get_dcnn_features_modified(imdb.net,bachGround_patch,'encoder',imdb.encoder);
    case 'drcnn'
    	imdb.net.layers = imdb.net.layers(1:imdb.layer.rcnn) ;
        out.featureR=get_rcnn_features_modified(imdb.net, bachGround_patch);
    	imdb.net.layers = imdb.net.layers(1:imdb.layer.dcnn) ;
        out.featureD = get_dcnn_features_modified(imdb.net,bachGround_patch,'encoder',imdb.encoder);
	for i=1:numel(out.featureD)
		out.feature{i}=[out.featureR{i};out.featureD{i}];
	end
	out.featureR=[];
	out.featureD=[];
    case 'dsift'
        out.feature = get_dcnn_features_modified([],bachGround_patch,'useSIFT', true,'encoder', imdb.encoder,'numSpatialSubdivisions',1,'maxNumLocalDescriptorsReturned', 500);
    case 'gabor'
        out.feature = get_gabor_feature(imdb.net,bachGround_patch);
end
clear bachGround_patch;
out.labels = labels;
out.set = ones(1,imdb.batchSize)*id;
out.labeID = labeID;
%--------------------------------------------
function [out] = extractFeatureForEncode(imdb,ii,varargin)
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
bachGround_patch = cell(1,imdb.batchSize) ;
labels=ones(no_labels,imdb.batchSize)*-1;
labeID =[];
j=1;
for i=ii:ii+imdb.batchSize-1
	bachGround_patch{j} = imread(imdb.Sets{id}.imagePath{i});
	label_id = imdb.Sets{id}.label(i)+1; % +1 for matlab indices
    	temps = ones(no_labels,1)*-1;
	temps(label_id,1) =1;
	labels(:,j)=temps;
 	labeID(1,j) =label_id;
	j=j+1;
end
switch imdb.feature
    case 'rcnn'
    	imdb.net.layers = imdb.net.layers(1:imdb.layer.rcnn) ;
        out.feature=get_rcnn_features_modified(imdb.net, bachGround_patch);
    case 'dcnn'
    	imdb.net.layers = imdb.net.layers(1:imdb.layer.dcnn) ;
        out.feature = get_dcnn_features_modified(imdb.net,bachGround_patch,'encoder',imdb.encoder);
    case 'drcnn'
    	imdb.net.layers = imdb.net.layers(1:imdb.layer.rcnn) ;
        out.feature=get_rcnn_features_modified(imdb.net, bachGround_patch);
    	imdb.net.layers = imdb.net.layers(1:imdb.layer.dcnn) ;
        out.feature = get_dcnn_features_modified(imdb.net,bachGround_patch,'encoder',imdb.encoder);
    case 'dsift'
        out.feature = get_dcnn_features_modified([],bachGround_patch,'useSIFT', true,'encoder', imdb.encoder,'numSpatialSubdivisions',1,'maxNumLocalDescriptorsReturned', 500);
    case 'gabor'
        out.feature = get_gabor_feature(imdb.net,bachGround_patch);
end
clear bachGround_patch;
out.labels = labels;
out.set = ones(1,imdb.batchSize)*id;
out.labeID = labeID;
%-------------------------------------------
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
%!./copyTrainTestSet.sh
opts = vl_argparse(opts, varargin) ;
imdb.imageDir = fullfile(imdb.dataLoc, 'Tobacco3482') ;
trainFileId = fopen([imdb.imageDir,'/list.txt']);
trainRead = textscan(trainFileId,'%s %s %s','Delimiter', '/');
imdb.classes.name = sort(unique(trainRead{2}));
fclose(trainFileId);
numClass = length(imdb.classes.name);
%0 letter
%1 form
%2 email
%3 handwritten
%4 advertisement
%5 scientific report
%6 scientific publication
%7 specification
%8 file folder
%9 news article
%10 budget
%11 invoice
%12 presentation
%13 questionnaire
%14 resume
%15 memo

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
imageSets = {'list'};
for s = 1:length(imageSets),
    imageSetPath = fullfile(imdb.dataLoc, 'Tobacco3482', sprintf('%s.txt',imageSets{s}));
    %imageSetPath = fullfile(sprintf('%s.txt',imageSets{s}));
    fileID = fopen(imageSetPath);
    %C = textscan(fileID,'%s %s','Delimiter', ' ');
    C = textscan(fileID,'%s');
    numberOfImages = numel(C{1});
    if numberOfImages>=1
         for j=1:numberOfImages
		tempLoc = C{1,1}(j);
                imdb.Sets{s}.imagePath{j} = [imdb.dataLoc,'Tobacco3482',tempLoc{1}];
		imdb.images{j}=tempLoc{1};
		tempLabel = strsplit(imdb.images{j},'/');
		IndexC = strfind(imdb.classes.name, tempLabel{2});
		%Index = find(not(cellfun('isempty', IndexC)));
                imdb.Sets{s}.label(j) = find(not(cellfun('isempty', IndexC)));
          end
    end
    fclose(fileID);

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
opts.numSamplesPerWord = 2000 ;
opts.whitening = false ;
opts.whiteningRegul = 0 ;
opts.renormalize = false ;
opts.numWords = 16 ;
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
    case {'dcnn', 'dsift','drcnn'}
        encoder.numWords = opts.numWords ;
        encoder.renormalize = opts.renormalize ;
        encoder.numSpatialSubdivisions = opts.numSpatialSubdivisions ;
end

switch opts.type
    case {'rcnn', 'dcnn','drcnn'}
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
size(descrs)
numImages = numel(code)*numel(code{1}.feature);
numDescrsPerImage = floor(opts.numWords * opts.numSamplesPerWord / numImages)
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
	size(descrs)
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
function label=getLabel(id)
switch id
	case 1
		label='letter'
	case 2
		label='form'
	case 3
		label='email'
	case 4
		label='handwritten'
	case 5
		label='advertisement'
	case 6
		label='scientific report'
	case 7
		label='scientific publication'
	case 8
		label='specification'
	case 9
		label='file folder'
	case 10
		label='news article'
	case 11
		label='budget'
	case 12
		label='invoice'
	case 13
		label='presentation'
	case 14
		label='questionnaire'
	case 15
		label='resume'
	case 16
		label='memo'
end
%0 letter
%1 form
%2 email
%3 handwritten
%4 advertisement
%5 scientific report
%6 scientific publication
%7 specification
%8 file folder
%9 news article
%10 budget
%11 invoice
%12 presentation
%13 questionnaire
%14 resume
%15 memo
%===============================================
