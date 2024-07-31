function img_patch=cut_patch(a,longitude,latitude,blocksize)

%given viewpoint and fov range, return the viewport image
[sizeiny,sizeinx]=size(a);
sizeoutx=blocksize;
sizeouty=blocksize;

anglex=(180*blocksize/sizeiny)*pi/180;
angley=(180*blocksize/sizeiny)*pi/180;    %angley=pi-angley',angley'=2*atan(sizeinx/sizeiny*tan(anglex/2))
faceSizex = 2*tan(anglex/2);
faceSizey = 2*tan(angley/2);
img_patch=zeros(sizeouty,sizeoutx);


lat=latitude; 
lon=longitude;
coords = getcoords3(lon,lat,anglex,angley);
for ii=1:sizeoutx
    for jj=1:sizeouty
        c = 1.0 * ii / sizeoutx;
        d = 1.0 * jj / sizeouty;

        x = (1-c)*(1-d)*coords(1,1)+c*(1-d)*coords(1,2)+(1-c)*d*coords(1,3)+c*d*coords(1,4);
        y = (1-c)*(1-d)*coords(2,1)+c*(1-d)*coords(2,2)+(1-c)*d*coords(2,3)+c*d*coords(2,4);
        z = (1-c)*(1-d)*coords(3,1)+c*(1-d)*coords(3,2)+(1-c)*d*coords(3,3)+c*d*coords(3,4);

        r = sqrt(x^2+y^2+z^2);

        theta=asin(z/r);
        if(x<0&&y<=0)
            phi=atan(y/x)-pi;
        elseif(x<0&&y>0)
            phi=atan(y/x)+pi;
        else
            phi=atan(y/x);
        end
        theta=(pi/2-theta)*sizeiny/pi;
        phi=(phi+pi)*sizeinx/2/pi;
        thetaf=floor(theta);
        phif=floor(phi);
        p=theta-thetaf;
        q=phi-phif;
        if thetaf==0
            thetaf=1;
            p=0;
        end
        if thetaf>=sizeiny
            thetaf=sizeiny;
            img_patch(jj,ii)=(1-q)*a(thetaf,mod(phif-1,sizeinx)+1)+q*a(thetaf,mod(phif,sizeinx)+1);
        else
            img_patch(jj,ii)=(1-p)*(1-q)*a(thetaf,mod(phif-1,sizeinx)+1)+(1-p)*q*a(thetaf,mod(phif,sizeinx)+1)+p*(1-q)*a(thetaf+1,mod(phif-1,sizeinx)+1)+p*q*a(thetaf+1,mod(phif,sizeinx)+1);
        end
    end
end



    
    
