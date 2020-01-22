%% this script is used to simulate a linear ODE dY/dt=AY and A is a sparse matrix
close all;
clear all;
comp='/Users/yuewu/';%the computer user location
workdir=[comp 'Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/nc_model/nnt/simulation/simulat.linear.small.mutinnt/'];%%the working folder
cd(workdir);
%---------------------------------------------------------------------
nthetaset=10000;%%number of random theta sets to be generated. ==training set size
timerang=[0 10.0];%% time range
stepsize=0.1;%% time stepsize
randseed=1;
ndim=8;%%the dimension(length) of Y and theta matrix
maxrandrag=2;%the max value in connection. the minimal value for other connections is 1 by default; the diagonal is always connected though. (so connection >= 2)
scalefactor=1.01;%make sure no near zero eigen value and there is enough oscialation(imag part) signal
%---------------------------------------------------------------------
% Create time grid
timeseq=timerang(1):stepsize:timerang(2);
ntime=length(timeseq);
%---------------------------------------------------------------------
rng(randseed);%%random seed for reproducibility
nrandconnect=ceil(rand(1,ndim)*maxrandrag);%random number from 1 to maxrandrag
preconnectmat=zeros(ndim,ndim);
for maxrandragi=1:ndim
  % preconnectmat(maxrandragi,ceil(rand(1,nrandconnect(maxrandragi))*ndim))=1;
  preconnectmat(maxrandragi,randsample(ndim,nrandconnect(maxrandragi)))=1;%%sample without replacement
end
preconnectmat=preconnectmat+diag(ones(1,ndim));
exisind=find(preconnectmat>0);
ntheta=length(exisind);
%---------------------------------------------------------------------
%%initialization of storing array
inputstore=zeros(nthetaset*ntime,ntheta*2+ndim*2+1);%the long&original input vector
inputstore2=zeros(nthetaset*ntime,ndim*4+1);%%use eigen value instead of original parameter matrix
inputstore_stepwise=zeros(nthetaset*(ntime-1),ntheta*2+ndim*2+1);%%the input store for step wise type of nnt
outputstore_stepwise=zeros(nthetaset*(ntime-1),ndim*2);%%the output store for step wise type of nnt
outputstore=zeros(nthetaset*ntime,ndim*2);
outputstorepre=zeros(nthetaset*ntime,ndim*2);
parastore=zeros(nthetaset,ndim*2);
samplevec=[];
samplevec_stepwise=[];
eigenrealvec=[];
eigenimagvec=[];
stepwiseind={1:(ntime-1) 2:ntime};
%---------------------------------------------------------------------
%%generate sample of H-matrices and their ODE solutions
for isample = 1:nthetaset
  % isample
  % index
  samprag=((isample-1):isample)*ntime;
  sampind=(samprag(1)+1):samprag(2);
  %% for step wise training, the first time point will be omitted
  samprag_stepwize=((isample-1):isample)*(ntime-1);
  sampind_stepwize=(samprag_stepwize(1)+1):samprag_stepwize(2);
  %% Set initial random elements of H
  hre=rand(1,ntheta).*2-1;%[-1 1]
  him=rand(1,ntheta).*2-1;
  hvec=hre+him*i;
  hmat=zeros(ndim,ndim);
  hmat(exisind)=hvec;
  %% regenerate of H matrix
  D=eig(hmat);
  % deld=max(abs(real(D)))-min(abs(real(D)));
  deld=abs(max(real(D)));
  dadd= -scalefactor*deld;
  hnew=hmat+diag(repmat(dadd,1,ndim));
  [umat,dmat]=eig(hnew);%% could reuse U D' V theoritically
  %% extract the real parameters (theta) to be stored
  hvecnew=hnew(exisind);
  hrenew=real(hvecnew)';%1*L
  himnew=imag(hvecnew)';
  vmat=inv(umat);
  dvec=diag(dmat);
  %% set initial condition of ODE
  yinireal=rand(ndim,1).*2-1;
  % yinireal=repmat(0.5,ndim,1); % test constant initial condition
  yiniimag=rand(ndim,1).*2-1;
  % yiniimag=repmat(0.5,ndim,1); % test constant initial condition
  yinimat=yinireal+yiniimag*i;
  %% solve ODE by explicit solution
  Avec=vmat*yinimat;
  Bmat=Avec.'.*umat;
  Emat=exp(dvec*timeseq);%%exp(NXM)
  ytmat=Bmat*Emat;
  %%rescaling y through time
  real_y=real(ytmat.');
  imag_y=imag(ytmat.');
  omega_y_real=sum(abs(real_y).^2,1);
  omega_y_imag=sum(abs(imag_y).^2,1);
  % bsxfun(@rdivide,sum(abs(real_y)^2,1)
  outputstorepre(sampind,:)=[real_y imag_y];
  real_y=real_y./sqrt(omega_y_real);
  imag_y=imag_y./sqrt(omega_y_imag);
  %%shifting
    % minrealy=min(real(ytmat),[],2);
    % minimagy=min(imag(ytmat),[],2);
    % yinireal=yinireal-minrealy*scalefactor.*(minrealy<0);
    % yiniimag=yiniimag-minimagy*scalefactor.*(minimagy<0);
  %%second around of simulation
    % yinimat=yinireal+yiniimag*i;
    % Avec=vmat*yinimat;
    % Bmat=Avec'.*umat;
    % Emat=exp(dvec*timeseq);
    % ytmat=Bmat*Emat;
  inputstore(sampind,:)=[repmat([hrenew himnew yinireal' yiniimag'],ntime,1) timeseq'];
  % inputstore(sampind,:)=[repmat([hrenew himnew],ntime,1) timeseq']; %% the one used for input
  inputstore2(sampind,:)=[repmat([real(dvec)' imag(dvec)' yinireal' yiniimag'],ntime,1) timeseq']; %% just for storage
  % inputstore2(sampind,:)=[repmat([real(dvec)' imag(dvec)' ],ntime,1) timeseq'];
  inputstore_stepwise(sampind_stepwize,:)=[repmat([hrenew himnew],(ntime-1),1) real_y(stepwiseind{1},:) imag_y(stepwiseind{1},:) (timeseq(stepwiseind{2})-timeseq(stepwiseind{1}))'];
  outputstore(sampind,:)=[real_y imag_y]; %% the one used for output
  % outputstorelog(sampind,:)=[log(real_y) log(imag_y)];
  outputstore_stepwise(sampind_stepwize,:)=[real_y(stepwiseind{2},:) imag_y(stepwiseind{2},:)];
  parastore(isample,:)=[omega_y_real omega_y_imag];
  samplevec=[samplevec repmat(isample,1,ntime)];
  samplevec_stepwise=[samplevec_stepwise repmat(isample,1,(ntime-1))];
  %%collect eigen value for plotting
  eigenrealvec=[eigenrealvec real(dvec)'];
  eigenimagvec=[eigenimagvec imag(dvec)'];
end
%%real eigen value histogram
fig=figure(), hold on
  histogram(eigenrealvec);
% saveas(fig,strcat(workdir,'realeigen.fig'));
close(fig);
%%2pi/imag_eigen_value histogram really hard to plot
fig=figure(), hold on
  histogram(abs(2*pi./eigenimagvec));
  set(gca,'XScale','log');
  set(gca,'YScale','log');
% saveas(fig,strcat(workdir,'period.fig'));
close(fig);
fig=figure(), hold on
  histogram(abs(eigenimagvec/(2*pi)));
  % set(gca,'XScale','log');
% saveas(fig,strcat(workdir,'frequency.fig'));
close(fig);
% save('./data/sparselinearode_new.small.stepwiseadd.mat','inputstore','inputstore2','inputstore_stepwise','outputstore','outputstore_stepwise','samplevec','samplevec_stepwise','parastore','nthetaset','ntime','ntheta','ndim','outputstorepre','-v7.3');%'outputstorelog',

%% test code
whos inputstore
plot(real(ytmat.'));
plot(imag(ytmat.'));
