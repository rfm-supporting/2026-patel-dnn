function [ imf, alpha, m ] = dakinbex( im, plotit )

% DAKINBEX  Dakin-Bex lightness model
%
% [ imf, alpha, m ] = dakinbex( im )

% 31-Mar-2021 -- created (RFM)

% set default arguments
if nargin < 1
    im = imread( 'images/einstein.tif' );
    im = double( im ) / 256;
end
if nargin < 2
    plotit = true;
end

% make filter bank
persistent FILTERBANK
if isempty( FILTERBANK ) || any( FILTERBANK.dim ~= size(im) )
    
    % record image size
    FILTERBANK.dim = size(im);
    
    % make frequency matrix
    f = meshrt( FILTERBANK.dim );
    
    % create bandpass filters in frequency domain
    FILTERBANK.nfilter = floor(log2( min(FILTERBANK.dim) )) - 1;
    FILTERBANK.sigma = sqrt(log(2)) / (2*sqrt(2));  % makes filter one octave wide (FWHM)
    FILTERBANK.filter = [];
    for i = 1:FILTERBANK.nfilter
        FILTERBANK.filter(i).fpeak = 2 ^ (i-1);
        FILTERBANK.filter(i).ft = ifftshift( gb( f, FILTERBANK.filter(i).fpeak, FILTERBANK.sigma ) );
    end
    
end

% band-pass filter image
im_ft = fft2( im );
imbp_ft = cell( [ 1 FILTERBANK.nfilter ] );
for i = 1:FILTERBANK.nfilter
    imbp_ft{i} = im_ft .* FILTERBANK.filter(i).ft;
end

% try several values of alpha
alphalist = linspace( -1, 1, 21 );
mlist = NaN( size( alphalist ) );
for k = 1:numel( alphalist )
    
    % construct fourier transform of alpha-filtered image
    imalpha_ft = dbfilter( imbp_ft, alphalist(k), FILTERBANK );
    
    % find slope of log energy vs log frequency
    [ mlist(k), logpeakf, loge ] = dbslope( imalpha_ft, FILTERBANK );
    
    % show results
    if plotit
        subplot(2,2,1); plot( logpeakf, loge, 'ro-' ); xlabel 'log frequency'; ylabel 'log energy'
        subplot(2,2,2); imagesc( ifft2( imalpha_ft ) ); axis image; colormap gray;
        drawnow;
    end
    
end

% find best value of alpha
f = @( m ) interp1( mlist, alphalist, m );
mtarget = -0.04;
alpha = f( mtarget );

% find best filtered image
imf_ft = dbfilter( imbp_ft, alpha, FILTERBANK );
imf = ifft2( imf_ft );

% find slope of best filtered image
m = dbslope( imf_ft, FILTERBANK );
if abs( (m-mtarget)/mtarget ) > 0.20
    warning( 'unable to create slope near target value of %.4f; slope is %.4f', mtarget, m );
end

% show results
if plotit
    
    subplot(2,2,2); imagesc( im ); axis image; colormap gray;

    subplot(2,2,3);
    plot( alphalist, mlist, 'go-' );
    hold on;
    ylim = get( gca, 'YLim' );
    plot( [ alpha alpha ], ylim, 'k-' );
    hold off;
    xlabel alpha
    ylabel slope
    
    subplot(2,2,4);
    imagesc( imf ); axis image; colormap gray;
    drawnow;
    
    fprintf( 'alpha = %.2f, m = %.4f\n', alpha, m );
    
    if nargout == 0
        clear imf
    end
    
end

end


function imalpha_ft = dbfilter( imbp_ft, alpha, filterbank )

% find weighted sum of band-pass images
imalpha_ft = zeros( size( imbp_ft{1} ) );
for i = 1:filterbank.nfilter
    w = 1 / filterbank.filter(i).fpeak ^ alpha;
    imalpha_ft = imalpha_ft + w * imbp_ft{i};
end

end


function [ m, logpeakf, loge ] = dbslope( im_ft, filterbank )

% find log peak frequencies
logpeakf = log( [ filterbank.filter(:).fpeak ] );

% find energy in frequency bands of image
loge = NaN( [ 1 filterbank.nfilter ] );
for i = 1:filterbank.nfilter
    im_bp_ft = im_ft .* filterbank.filter(i).ft;
    loge(i) = log( mean( abs( im_bp_ft ).^2, 'all' ) );
end

% find slope of energy vs log peak frequency
b = regress( loge', [ logpeakf' ones(size(logpeakf')) ] );
m = b(1);

end


function g = gb( f, fpeak, sigma )

% GB  Gabor filter

g1 = log( f / fpeak ).^2;
% g2 = 2 * log( sigma / fpeak )^2;   % *** shouldn't this just be 2 * sigma^2 ?
g2 = 2 * sigma^2;
g = exp( -g1/g2 );

end

function [ r, t ] = meshrt( dim, radflag )

% MESHRT  Make rt coordinate matrices
% 
% [ r, t ] = meshrt( dim, radflag )

% 03-Apr-2021 -- created (RFM)

% set default argument
if nargin < 2, radflag = true; end

% create xy coordinate matrices
midn = floor(dim/2)+1;
xvec =    (1:dim(2)) - midn(2);
yvec = -( (1:dim(1)) - midn(1) );
[x,y] = meshgrid( xvec, yvec );

% create radius matrix
r = sqrt( x.^2 + y.^2 );

% create angle matrix
if radflag
    t = atan2( y, x );
else
    t = atan2d( y, x );
end

end

function imp = pad( im, dim, bg )

% PAD  Pad an image to a required size
%
% imp = pad( im, dim, bg )

% 07-Apr-2017 -- created (RFM)

% actual size
dima = size( im );

% image already the right size?
if all( size(im)==dim )
    imp = im;
    return
end

% image too large?
if any(dima>dim)
    error('input argument ''im'' is larger than %.0f x %.0f',dim);
end

% find offset
ddim = dim - dima;
offset = ceil( ddim/2 )+1;

% create padded image
imp = repmat(bg,dim);
imp(offset(1):offset(1)+dima(1)-1,offset(2):offset(2)+dima(2)-1) = im;

end
