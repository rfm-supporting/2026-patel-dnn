% apply_models.m  Apply classical models to illusion stimuli

clear; clc;

addpath('models');

srcdir = '../illusions_matching';
dstdirbase = 'model_out';
modellist = { 'odog' 'dakinbex' 'retinex99' };

for i = 1:numel(modellist)

    model = modellist{i};
    fprintf('%s\n',model);

    dstdir = [ dstdirbase '_' model ];
    if exist(dstdir, 'dir')
        rmdir(dstdir, 's');
    end
    mkdir(dstdir);

    d = dir(fullfile(srcdir, '*_lum.hdr'));
    for j = 1:numel(d)

        if contains(d(j).name, 'mask')
            continue
        end

        fprintf('    %s\n',d(j).name);

        fname = fullfile(d(j).folder, d(j).name);
        im = hdrread(fname);
        im = double(im(:,:,1));

        switch model
            case 'odog'
                im_large = imresize(im,[ 1024 1024 ],'nearest');
                im_out = odog(im_large);
                im_out = shrink(im_out,2);
            case 'dakinbex'
                im_out = dakinbex(im, 0);
            case 'retinex99'
                im_out = retinex_mccann99(log10(im), 4);
        end

        [~,n] = fileparts(fname);
        n = strrep(n,'_lum','_refhat');
        fname_out = fullfile(dstdir, [ n '.mat' ]);
        save(fname_out,'im_out');

    end

end

function im2 = shrink(im, k)
dim = size(im);
im = reshape(im,k,1,[]);
im = sum(im,1);
im = reshape(im,dim(1)/k,k,[]);
im = sum(im,2);
im2 = reshape(im,dim/k);
end
