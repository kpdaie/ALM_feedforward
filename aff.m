clear
reps = 20;
figure(825);figure_initialize
set(gcf,'position',[15 3 4.5 3]);
for nwi = 1:3
    try
        keep bump dim ampFactor nwi sigma otherBump AMP bumpTime fwhm FR dimPCA reps noise cdcorr corrtd;
    end
    amps = [1.06 1.1 0 .99];
    fbs   = [.9  0    1 .8];
    inpAmps  = [1/3 .01 2 1/10];
    noise    = [.4 .4 .4];
    dt = 0.01;
    t = 0:dt:3.4;
    N = 150;
    m = 60;
    num = 2;
    fun = @(x) (x);
    tau = 0.05;
    figure(nwi);
    figure_initialize
    set(gcf,'position',[15 3 4 1.4]);
    amp = amps(nwi);
    fb = fbs(nwi);;
    inpAmp = inpAmps(nwi);
    T = triu(ones(m),1) - triu(ones(m),num);% feedforward connectivity
    T = T*amp;
    if nwi == 3;
        T(1,1) = fb;%self connection of input (elongates reponse of input neurons as in the data).
    end
    if nwi == 2
        N = 150;
        m = 150;
        tau = 0.02;
        T = triu(ones(m),1) - triu(ones(m),num);% feedforward connectivity
        T = T*amp;
        T(1,:) = (rand(1,m)>.9)/10*amp;% random connections from input direction (reduces dimensionality of dynamics)       
        inds = [10 20 60 70];
        vec = zeros(1,N);vec(inds) = fb;
        vec2 = zeros(1,N);vec2(inds+1) = fb;
        T = T + diag(vec) + diag(vec2);
        T(1,1) = fb;
    end
    if nwi == 1
%         T(1,:) = (rand(1,m)>.9)/10*amp;% random connections from input direction (reduces dimensionality of dynamics)       
        T(1,2:5:end) = 1/20*amp;% random connections from input direction (reduces dimensionality of dynamics)       
        inds = [10 20 ];
        vec = zeros(1,m);vec(inds) = 0;
        vec2 = zeros(1,m);vec2(inds+1) = 0;
        T = T + diag(vec) + diag(vec2);
        T(1,1) = fb;
    end
    % d = blkdiag(d,d);
    v1 = randn(N,m);% Schur dimensions
    v1(:,1) = rand(N,1);% make the mode at the end of the chain all same sign for robustness to photoinhibition
    v1(:,2) = linspace(-1,1,N);
    v1 = (Gram_Schmidt_Process(v1));
    v1(:,[1 2 end]) = v1(:,[2 end 1]);
    v2 = randn(m);v2(:,1) = ones(m,1);
    v2 = fliplr(Gram_Schmidt_Process(v2));
    % v = blkdiag(v1,v2);
    U = v1;
%     U = U(:,[1:25,125:150]);
    w = U*T*U';
    w = w/(num-1);%adjust amplitude to tune ramping strength
    r = zeros(1,N);
    % inp = r;inp([1 ((m)+1)]) = [1 -1];
    inp = [U(:,1)'+randn(1,N)/20;randn(1,N)/18]*inpAmp;
    % inp(2,:) = inp(1,:);%Inputs
    
    for k = 1:2;
        for i = 1:length(t)-1
            r(i+1,:) = r(i,:) + dt/tau*(-r(i,:) + fun(r(i,:))*w + ...
                2*((k==2)-.5)*inp(k,:)*(t(i)>.2 & t(i)<.3) + ...
                2*((k==2)-.5)*inp(k,:)*(t(i)>1.3 & t(i)<1.4));
        end
        R(:,:,k) = r;
    end
    clf
    subplot(231)
    d = diff(R,[],3);
    v = -mean(d(end-100:end,:))';v=v/norm(v);
    % v = -d(end,:)';
    plot(t,R(:,:,1)*v(:,1),'b');hold on
    plot(t,R(:,:,2)*v(:,1),'r');hold on
    subplot(2,3,2);
    imagesc(corr(d(1:end,:)'),[0 1])
    cdcorr(:,:,nwi) = corr(d(1:end,:)');
    vs = -mean(d((t>.1 & t<.2),:))';v=v/norm(v);
    
    figure(nwi)
    subplot(223);
    trueFactor = (nanvar(R(:,:,1)*U(:,end)))/(nanvar(R(:,:,1)*U(:,1)))
    if nwi == 1;
        figure(11)
        for i = 1:25;subplot(5,5,i);plot(squeeze(R(:,round(rand*(N-1)+1),:)));end
        figure(nwi);
    end
    FR(:,:,:,nwi) = R;
    for rep = 1:reps;
        figure(nwi)
        % simulate network with noise to determine Input direction
        for iter = 1:100;
            for k = 1:2;
                a = randn(1,N);
                a=a/norm(a)*norm(inp(k,:))*1;
                a = inp(k,:) + a;
                for i = 1:length(t)-1
                    r(i+1,:) = r(i,:) + dt/tau*(-r(i,:) + fun(r(i,:))*w + ...
                        2*((k==2)-.5)*a*(t(i)>.1 & t(i)<.2) + randn(1,N)*noise(nwi));
                end
                Rn(:,:,k,iter) = r;
            end
        end
        % u = Gram_Schmidt_Process([U(:,1) v]);
        % V = U(:,end);
        % u = U(:,1);
        % u = abs(u);
        clear cc bb tt
        bins = 20:17:length(t);
        d = diff(R,[],3);
        v = -mean(d(end-100:end,:))';v=v/norm(v);
        train = v'*squeeze(mean(Rn(end-50:end,:,1,1:2:end)));
        test = v'*squeeze(mean(Rn(end-50:end,:,1,2:2:end)));
        ccc = nan((-1+length(bins))*2,length(bins)-1);
        for i = 1:length(bins) - 1
            Xtrain = squeeze(mean(Rn(bins(i):bins(i+1),:,1,1:2:end)));
            Xtest = squeeze(mean(Rn(bins(i):bins(i+1),:,1,2:2:end)));
            a = svd(Xtrain');tol = a(end)+1;
            beta = pinv(Xtrain',tol)*train';
            beta2 = pinv(Xtest',tol)*test';
            bb(:,i) = beta/norm(beta);
            bb2(:,i) = beta2/norm(beta2);
            bet(:,i)= beta;
            cc = corr(bb,bb);
            cc1 = corr(bb,bb2);
            cc2 = corr(bb2,bb);
%             for j = 1:length(bins)-1
%                 Xtest= squeeze(mean(Rn(bins(j):bins(j+1),:,1,2:2:end),1));
%                 Xtrain= squeeze(mean(Rn(bins(j):bins(j+1),:,1,1:2:end),1));
%                 cc1(j,i) = corr((beta'*Xtest)',test');
%                 cc2(j,i) = corr((beta2'*Xtrain)',train');
%                 cc(j,i) = corr((beta'*Xtrain)',train');
%             end
            tt(i) = mean(t(bins(i):bins(i+1)));
            strt = floor(mean(bins(i):bins(i+1)));
            %     ccc(end-strt-length(bins)+1:end-strt,i) = cc(:,i);
            %     ampBeta(i) = sum(beta'*Xtest);
        end
        
        nn = size(cc,2);
        ccc = nan(nn*2,nn);
        for i = 1:nn;
            ccc((1:nn)+(nn-i),i) = cc1(:,i);
        end
        corrtd(:,:,nwi,rep) = cc1;
        figure(nwi)
        d = -diff(mean(Rn(:,:,:,2:2:end),4),[],3);
        subplot(235);
        amp = nanvar(d*bb);
        plot(amp(2:end-1),'o-')
        AMP(:,nwi,rep) = amp;
        trueFactor = (nanvar(R(:,:,1)*U(:,end)))/(nanvar(R(:,:,1)*U(:,1)));
        ampFactor(rep,nwi) = mean(amp(end-3:end-1))/mean(amp(2:7));
%         ttt = text(10,max(ylim)/2,num2str(round(ampFactor(rep,nwi))));
        dtt=mean(diff(tt));
        tt = 0:dtt:dtt*(size(ccc,1)-1);
        tt = tt - tt(floor(length(tt)/2));
        subplot(236);cla
        yy = nanmean(ccc(:,1:end),2);
        ind = find(isnan(yy)==0);
        plot(tt(ind),yy(ind))
%         k = ezfit('a*exp(-(x-g).^2/b^2)');
%         ttt = text(0,.5,num2str(round(k.m(2)*10)/10));
        bumpTime(:,rep,nwi) = tt(ind);
        bump(:,rep,nwi) = yy(ind);
%         sigma(rep,nwi) = k.m(2);
        fwhm = [];
%         fwhm(rep,nwi) = tt(max(find(yy>max(yy)/2)))-tt(min(find(yy>max(yy)/2)-1));
%         xlim([-1.5 1.5])
%         ylim([0 1])
        subplot(234);cla
        imagesc(cc1'/2+cc2'/2,[0 1]);
        %     title(lab);
        colormap(jet)
        
        subplot(2,3,3);cla
        ad = (bb+bb2)/2;
        ss = svd(ad).^2;
        mn = (bb-bb2)/2;
        sn = svd(mn).^2;
        DD = ss-sn;
        plot(cumsum(DD)/sum(DD),'o-');hold on;
        figure(nwi)
        dim(:,rep,nwi) = cumsum(DD)/sum(DD);
        ttt = text(10,.7,num2str(sum(dim(:,rep,nwi)<.99)));
        otherBump(:,rep,nwi) = cc(8,:);
        
        rr = squeeze(Rn(:,:,1,:))-squeeze(Rn(:,:,2,:));
        r1 = mean(rr(:,:,1:2:end),3);
        r2 = mean(rr(:,:,2:2:end),3);
        ad = (r1+r2);
        ss = svd(ad).^2;
        mn = (r1-r2)/2;
        sn = svd(mn).^2;
        DD = ss-sn;
        plot(cumsum(DD)/sum(DD),'ko-');hold on;
        xlim([0 20])
        figure(nwi)
        dimPCA(:,rep,nwi) = cumsum(DD)/sum(DD);
    end
    figure_finalize
end
%%
marg = [.3];

figure(5);clf
figure_initialize
% set(gcf,'position',[3 3 3.75 1.5]);
set(gcf,'position',[17 3 1.2 .8]);
KDsubplot(1,1,[1 1],.2);
% KDsubplot(3,1,1,.3);
fb = [255 0 0];
ff = [0 0 255];
col = [(fb+ff)/2;ff;fb]/255;
ttt = 0:dtt:dtt*(size(bump,1)-1);
ttt = ttt - ttt(floor(length(ttt)/2));

for i = 3:-1:1;
    ttt = mean(bumpTime(:,:,i),3);
    confidence_bounds_percentile(ttt,squeeze(bump(:,:,i)),95,col(i,:),col(i,:),.2);hold on;
%     plot(ttt,mean(bump(:,:,i),2),'color',col(i,:));hold on;
end
set(gca,'ytick',[0 .5]);
% ylim([0 1.1]);
% xlim([-1.5 1.5])
figure_finalize
figure(6);clf
figure_initialize
set(gcf,'position',[3 3 1.7 1.5]);
marg = [.3];
KDsubplot(1,1,[1 1],marg);
dm = squeeze(mean(dim,2));
dm = squeeze(sum(dm<.95))+1;
for i = 1:3;
    confidence_bounds_percentile(1:size(dim,1),squeeze(dim(:,:,i))*100,95,col(i,:),col(i,:),.2);
    hold on;
    plot([1 1]*dm(i),[0 95],':','color',col(i,:));
%     plot(1:size(dim,1),squeeze(mean(dim(:,:,i),2)),'o-','markerfacecolor','w','color',col(i,:),'markersize',4);
end
set(gca,'ytick',[0 50 95],'xtick',sort([dm 20]))
ylim([0 100])
plot(xlim,[95 95],'k:');

figure_finalize

figure(7);clf
figure_initialize
set(gcf,'position',[3 3 3.75 1.5]);
KDsubplot(1,2.4,[1 1],marg);
% KDsubplot(1,2,[1 1],marg);
for i = [3 1];
    x = 0:size(AMP,1)-1;
    y = mean(AMP(:,i,:),3);
    scl = y(2);
    confidence_bounds_percentile(x(1:end),squeeze(AMP(2:end,i,:))/scl,75,col(i,:),col(i,:),.2);
    hold on;
%     plot(x(2:end-1),y(2:end-1)/scl,'o-','markerfacecolor','w','color',col(i,:),'markersize',5);
end
ylim([-4 30])
set(gca,'ytick',[0 20],'xtick',[1 17])
xlim([1 17])
figure_finalize
% set(gca,'yscale','log')
%%
figure(8);clf
figure_initialize
set(gcf,'position',[3 3 4.5 4.5]);
marg = [.3];
dm = squeeze(mean(dimPCA,2));
dm = squeeze(sum(dm<.95))+1;
corrTD = mean(corrtd,4);
for i = 1:3;
    KDsubplot(3,3,[1,i],.3)
    confidence_bounds_percentile(1:size(dimPCA,1),squeeze(dimPCA(:,:,i))*100,95,'k','k',.2);
    hold on;
    xlim([0 15])
    ylim([50 100])
    plot([1 1]*dm(i),[0 95],'k:');
    plot(xlim,[95 95],'k:');
    set(gca,'ytick',[0 50 95],'xtick',sort(unique([1 dm(i) 10])))
%     plot(1:size(dim,1),squeeze(mean(dim(:,:,i),2)),'o-','markerfacecolor','w','color',col(i,:),'markersize',4);
    KDsubplot(3,3,[2,i],.3);
    imagesc(cdcorr(:,:,i),[0 1]);
    set(gca,'xtick',[141 341],'xticklabel',{'-2','0'})
    set(gca,'ytick',[141 341],'yticklabel',{'-2','0'});hold on;
    plot([141 141],ylim,'w');
    plot(xlim,[141 141],'w');
    colormap(jet);
    
    dtt = dt*mean(diff(bins));
    del = 1.2/dtt;
    KDsubplot(3,3,[3,i],.3);
    imagesc(corrTD(:,:,i),[0 1]);
    set(gca,'xtick',[del size(corrTD,2)],'xticklabel',{'-2','0'})
    set(gca,'ytick',[del size(corrTD,2)],'yticklabel',{'-2','0'});hold on;
    plot([1 1]*del,ylim,'w');
    plot(xlim,[1 1]*del,'w');
    colormap(jet);
end
figure_finalize
%%
figure(11);
% dat = load('C:\Users\daiek\Dropbox (HHMI)\ALM_feedforward_network\model_data\hff_realistic_psths.mat');
% R = dat.R;
% t = dat.t;
clf
figure_initialize
set(gcf,'position',[18 1 2.5 1.3]);
marg = [.2 .15];
% inds = [77 37 88 39 67 107 56 76 126 36 51 35 ];
% inds = [63 47 40 67, 84 58 72
R = FR(:,:,:,1);

inds(1:4) = [64 15 45 84];
inds(5:8) = [20 10 98 16];
inds(9:12) = [18 78 51 57];
for i = 1:12;
    KDsubplot(3,4,[i],marg);
    plot(t-t(end),squeeze(R(:,inds(i),:))*100);
    hold on;
    plot([0 0]-2,ylim,'k:');
    plot([0 0]-3.2,ylim,'k:');
    plot([0 0],ylim,'k:');
    yt = get(gca,'ytick');
    yt = yt([1 end]);
    set(gca,'ytick',yt)
    axis tight
    if i <9 
        set(gca,'xtick',[]);
    else
        set(gca,'xtick',[-2 0]);
    end
    colororder(gca,[0 0 1;1 0 0])
%     set(gca,'visible','off')
end
figure_finalize