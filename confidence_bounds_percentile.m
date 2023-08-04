function confidence_bounds_percentile(x,y,err,linecolor,facecolor,transparency);

if isempty(err);
    up = prctile(y',97.5);
    dn = prctile(y',2.5);
    y = nanmean(y,2);
else
    up = prctile(y',100-err);
    dn = prctile(y',err);
    y = nanmean(y,2);
end

ind = find(isnan(y)==0);
x=reshape(x,1,[]);
y=reshape(y,1,[]);
err=reshape(err,1,[]);
y = y(ind);x = x(ind); up = up(ind);dn=dn(ind);
X=[x fliplr(x)];
bnd=[dn fliplr(up)];
p=patch(X,bnd,facecolor);
set(p,'facealpha',transparency,'edgecolor','none');
hold on;
if ~isempty(linecolor)
plot(x,y,'-','color',linecolor)
end
hold off