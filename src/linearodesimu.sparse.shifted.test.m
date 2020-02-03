%%% the script to test correctness of simulation by both:
%%%% 1. anther math derived formula
%%%% 2. LHS RHS of the original derivative form
% Heinz-Bernd Schuttler
% 191211-2156hbs:
% Added LHS-RHS test for ODE solution
%
% 191211-1819hbs:
% 1) Reformatted from
%      "linearodesimu.sparse.shifted.m" (Y.Wu V.191210).
%    to make code legible
%
% 2) Added ODE solution option using only N x N matrix algebra (HBS)
%    as alternative to using also N x M matrix algebra (YW)
%    Added option to compare YW ODE solution to HBS ODE solution,
%    by calculating relative error, ERel between the two sols.
%---------------------------------------------------------------------
%% this script is used to simulate a linear ODE dY/dt=HY and H is a sparse matrix
close all;
clear all;
%---------------------------------------------------------------------
% Input: Set all input parameters, here only and only here
nthetaset=5;     %%number of random theta sets to be generated. ==training set size
timerang=[0 20.0];  %% time range
stepsize=0.01;   %% time stepsize
randseed=1;%79512497;
ndim=8;         % dimension of H-matrix and ODE soln vector Y
maxrandrag=2;     % max value in connection. the minimal value for other connections is 1 by default; the diagonal is always connected though. (so connection >= 2)
scalefactor=1.01; % make sure no near zero eigen value and; there is enough oscillation(imag part) signal
%--------------------Test related parameters--------------------------
%%%Test 1
jTestLR=1;% 1=Do LHS-RHS test of ODE soln.,  0=Do not
%%%Test 2
jSolvODE=3;% ODE soln algebra: 1=YW, 2=HBS, 3=YW&HBS (compare); The two algebra are from the same original equation set
jSpeedHBS=3;% Diff. speed versions of HBS algebra: set to 1, 2 or 3
%---------------------------------------------------------------------
% % Set working dir for chosen user:
% comp='/Users/yuewu/';%the computer user location
% workdir=[comp 'Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/nc_model/nnt/simulation/simulat.linear.small.mutinnt/'];%%the working folder
% cd(workdir);
%---------------------------------------------------------------------
% Create time grid
timeseq=timerang(1):stepsize:timerang(2);
ntime=length(timeseq);
%---------------------------------------------------------------------
rng(randseed);%%random seed for reproducibility
nrandconnect=ceil(rand(1,ndim)*maxrandrag);%random number from 1 to maxrandrag
preconnectmat=zeros(ndim,ndim);
for maxrandragi=1:ndim
  %X:
  % preconnectmat(maxrandragi,...
  % ceil(rand(1,nrandconnect(maxrandragi))*ndim))=1;
  %:X
  preconnectmat(maxrandragi,randsample(ndim,nrandconnect(maxrandragi)))=1;%%sample without replacement
end
preconnectmat=preconnectmat+diag(ones(1,ndim));
exisind=find(preconnectmat>0);
ntheta=length(exisind);
%DB
%DB Sz_exisind=size(exisind)
%DB Lg_exisind=length(exisind)
%DB Ne_exisind=numel(exisind)
%FB
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
  %-------------------------------------------------------------------
  %X:
  % isample
  % index
  %:X
  samprag=((isample-1):isample)*ntime;
  sampind=(samprag(1)+1):samprag(2);
  %% for step wise training, the first time point will be omitted
  samprag_stepwize=((isample-1):isample)*(ntime-1);
  sampind_stepwize=(samprag_stepwize(1)+1):samprag_stepwize(2);
  %-------------------------------------------------------------------
  %% Set initial random MEs of H
  hre=rand(1,ntheta).*2-1;%[-1 1]
  him=rand(1,ntheta).*2-1;
  hvec=hre+him*i;
  hmat=zeros(ndim,ndim);
  hmat(exisind)=hvec;
  %-------------------------------------------------------------------
  %% Find evals of H, store in vector D, use to apply eval shift
  D=eig(hmat);
  % deld=max(abs(real(D)))-min(abs(real(D)));
  deld=abs(max(real(D)));
  dadd= -scalefactor*deld;
  hnew=hmat+diag(repmat(dadd,1,ndim)); % hnew is shifted H-matrix
  %-------------------------------------------------------------------
  % Re-diagonalize shifted H-matrix
  [umat,dmat]=eig(hnew);%% could reuse U D' V theoretically
  %-------------------------------------------------------------------
  %% Extract the real ME parameters to be stored in theta
  hvecnew=hnew(exisind);
  hrenew=real(hvecnew)';%1*L
  himnew=imag(hvecnew)';
  %-------------------------------------------------------------------
  % Invert evec-matrix U
  vmat=inv(umat);
  dvec=diag(dmat);
  %-------------------------------------------------------------------
  %% Set random ICs of ODE soln Y
  yinireal=rand(ndim,1).*2-1;
  % yinireal=repmat(0.5,ndim,1); % test constant initial condition
  yiniimag=rand(ndim,1).*2-1;
  % yiniimag=repmat(0.5,ndim,1); % test constant initial condition
  yinimat=yinireal+yiniimag*i;
  %-------------------------------------------------------------------
  % Solve ODE system for all time grid pts, t_1,..., t_M
  % ODE soln algebra choices: 1=YW, 2=HBS, 3=YW&HBS (compare)

  % Solve by YW algebra
  if jSolvODE==1 | jSolvODE==3
    Avec=vmat*yinimat;
    Bmat=Avec.'.*umat;
    Emat=exp(dvec*timeseq);%%exp(N x M)
    ytmatYW=Bmat*Emat;
    ytmat=ytmatYW;
  end
  % Solve by HBS algebra
  if jSolvODE==2 | jSolvODE==3
    Avec=vmat*yinimat;
    ytmatHBS=zeros(ndim,ntime);
    if jSpeedHBS==1
      for itime=1:ntime
        EmatD=diag(exp(dvec*timeseq(itime)));
        ytmatHBS(:,itime)=umat*(EmatD*Avec);
      end % for itime=1:ntime
    end
    if jSpeedHBS==2
      for itime=1:ntime
        EtD=exp(dvec*timeseq(itime));
        ytmatHBS(:,itime)=umat*(EtD.*Avec);
      end % for itime=1:ntime
    end
    if jSpeedHBS==3
      for itime=1:ntime
        EtD=exp(dvec*timeseq(itime));
        ytmatHBS(:,itime)=(EtD.*Avec);
      end % for itime=1:ntime
      ytmatHBS=umat*ytmatHBS;
    end
    if jSolvODE==2
      ytmat=ytmatHBS;
    end
  end
  % Compare YW and HBS algebra ODE solns
  if jSolvODE==3
    EAbs=norm(ytmatYW-ytmatHBS,'fro');
    ERel=2.0*EAbs/(norm(ytmatYW,'fro')+norm(ytmatHBS,'fro'));
    fprintf(['isample=%5d --> ODE Solution Error  YW-HBS: '   ...
             'EAbs=%14.6e, ERel=%14.6e\n'             ]       ...
           ,isample,EAbs,ERel);
  end
  %-------------------------------------------------------------------
  % Perform LHS-RHS test of ODE solution
  % using 2nd order finite-diff approximation
  % for num. t-derivative estimation
  if jTestLR==1
    o2dtime=0.5/(timeseq(2)-timeseq(1));
    % LHS:
    dytmat=o2dtime*(ytmat(:,3:ntime)-ytmat(:,1:(ntime-2)));
    % RHS:
    hytmat=hnew*ytmat(:,2:(ntime-1));
    FAbs=norm(dytmat-hytmat,'fro');
    FRel=2.0*FAbs/(norm(dytmat,'fro')+norm(hytmat,'fro'));
    if jSolvODE==3
      fprintf('                                     LHS-RHS: ');
    else
      fprintf('isample=%5d --> ODE Solution Error LHS-RHS: ',isample);
    end
    fprintf(                               ...
             'FAbs=%14.6e, FRel=%14.6e\n'  ...
           ,FAbs,FRel);
  end
  %-------------------------------------------------------------------
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

%cHBS: fig=figure(), hold on
fig=figure(); hold on
%c:HBS
  histogram(eigenrealvec);
% saveas(fig,strcat(workdir,'realeigen.fig'));
close(fig);
%%2pi/imag_eigen_value histogram really hard to plot

%cHBS: fig=figure(), hold on
fig=figure(); hold on
%c:HBS
  histogram(abs(2*pi./eigenimagvec));
  set(gca,'XScale','log');
  set(gca,'YScale','log');
% saveas(fig,strcat(workdir,'period.fig'));
close(fig);

%cHBS: fig=figure(), hold on
fig=figure(); hold on
%c:HBS

  histogram(abs(eigenimagvec/(2*pi)));
  % set(gca,'XScale','log');
% saveas(fig,strcat(workdir,'frequency.fig'));
close(fig);

% save('./data/sparselinearode_new.small.stepwiseadd.mat' ...
%     ,'inputstore','inputstore2','inputstore_stepwise' ...
%     ,'outputstore','outputstore_stepwise','samplevec' ...
%     ,'samplevec_stepwise','parastore','nthetaset' ...
%     ,'ntime','ntheta','ndim','outputstorepre','-v7.3'); ...
%     %'outputstorelog',

%% test code
%cHBS:
%cHBS  whos inputstore
%c:HBS
plot(real(ytmat.'));
plot(imag(ytmat.'));
