% model_matches.m  Find reflectances for classical models with illusion stimuli

clear; clc;

scenedir = 'illusion_scenes';
modellist = { 'irnet' };
lumscale = 24.528;  % scale factor that maps luminances in rendered .hdr
                    % images to luminances displayed in experiment with human observers
                    % - note that this factor is only applied to rescale
                    %   the axes; it was not applied to the stimuli used
                    %   as model inputs

fid = fopen('model_matches.txt','w');
fprintf(fid,'model,run,stimname,reflum,matchlum\n');

for model = modellist

    if strcmp(model,'irnet')
        runs = 10;
    else
        runs = 1;
    end

    for run = 1:runs

        if strcmp(model,'irnet')
            modeldir = sprintf('model_out_irnet/model_run-%d',run);
        else
            modeldir = [ 'model_out_' model{1} ];
        end

        flist = dir(fullfile('..',scenedir,'*0p05*_lum.hdr'));
        for i = 1:numel(flist)

            [~,basename] = fileparts(flist(i).name);
            cast = ismember('X',basename);

            scenename0p05 = fullfile(flist(i).folder, flist(i).name);
            maskname = strrep(scenename0p05,'_0p05','-mask');
            mask = hdrread(maskname);
            mask_ref = mask(:,:,1) > 1.1*mask(:,:,2);
            mask_match = mask(:,:,2) > 1.1*mask(:,:,1);

            rlist = [ 0.05 0.1:0.1:0.8 ];
            lum_ref = NaN(size(rlist));
            lum_match = NaN(size(rlist));
            out_ref = NaN(size(rlist));
            out_match = NaN(size(rlist));

            for j = 1:numel(rlist)

                tag = sprintf('%.2f',rlist(j));
                tag = strrep(tag,'.','p');
                scenename = strrep(scenename0p05,'0p05',tag);

                stim = hdrread(scenename);
                stim = lumscale * stim(:,:,1);
                lum_ref(j) = median(stim(mask_ref));
                lum_match(j) = median(stim(mask_match));

                [~,outbase] = fileparts(scenename);
                outname = fullfile(modeldir, outbase);
                if strcmp(model,'irnet')
                    outname = strcat(outname,'_output.tif');
                    modelout = imread(outname);
                else
                    outname = strrep(outname,'_lum','_refhat.mat');
                    modelout = load(outname);
                    modelout = modelout.im_out;
                end
                out_ref(j) = median(modelout(mask_ref));
                out_match(j) = median(modelout(mask_match));

            end

            x = lum_match;
            y = out_match - out_ref;
            pp = spline(x, y);

            if strcmp(model,'irnet')
                xx = linspace(min(x),max(x),100);
                yy = ppval(pp, xx);
                [~,min_i] = min(abs(yy));
                lum_matchpoint = xx(min_i);
            else
                f = @(x) abs(ppval(pp, x));
                lum_matchpoint = fminbnd(f, 0.5 * min(x), 1.5 * max(x));
            end
            
            lum_refpoint = median(lum_ref);

            [~,n] = fileparts(scenename0p05);
            n = split(n,'_');
            n = n{2};
            datastr = sprintf('%s,%d,%s,%.6f,%.6f\n',model{1},run,n,lum_refpoint,lum_matchpoint);
            fprintf(fid,'%s',datastr);
            fprintf('%s',datastr);

        end

    end

end

fclose(fid);
