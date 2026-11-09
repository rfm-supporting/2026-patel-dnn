import os, random, imageio, copy
import numpy as np
import cv2 as cv

def stim2mask(fname):
    "convert stimulus filename to mask filename"
    dirname, name = os.path.split(fname)
    substr = name.split('_')
    substr[1] += '-mask'
    substr[3] = 'ref.hdr'
    return os.path.join(dirname, '_'.join(substr))

def grey2unit(g):
    "map range [0,255] to [-1,1]"
    return -1.0 + 2.0*(g/255.0)

def loadstim(directory, cal, nrepeat):
    '''load and process stimuli; returns a list of dictionaries, with one
    dictionary for each stimulus'''
    
    # make list of stimulus image filenames
    stimfiles = [os.path.join(directory, file) for file in os.listdir(directory) if ('lum' in file) and ('mask' not in file)]
    
    # find maximum rendered luminance in stimulus images
    maxlum_render = max([imageio.imread(stimfile).max() for stimfile in stimfiles])
    
    # find maximum luminance displayable on monitor
    maxlum_monitor = cal.grey2lum(np.array(255))
    
    # prepare stimuli
    stimlist = []
    for stimfile in stimfiles:
        
        # find stimulus name
        _, name = os.path.split(stimfile)
        name = name.split('_')[1]
        
        # load and calibrate stimulus image
        stimlum = imageio.imread(stimfile)
        stimlum = stimlum[::-1,:,0]
        stimlum = (maxlum_monitor/maxlum_render) * stimlum
        stimgrey = cal.lum2grey(stimlum)
        stimunit = grey2unit(stimgrey)
        
        # load mask image, and find high and low lightness regions
        maskfile = stim2mask(stimfile)
        maskref = imageio.imread(maskfile)
        maskref = maskref[::-1,:]
        masklow =  np.logical_and(maskref[:,:,0]==1, maskref[:,:,2]<0.95)
        maskhigh = np.logical_and(maskref[:,:,1]==1, maskref[:,:,2]<0.95)
        
        # make stimulus prompts
        prompt = np.tile(0.5+0.5*stimunit[...,np.newaxis], (1,1,3))
        red =  (0.8, 0.5, 0.5)
        green = (0.5, 0.8, 0.5)
        prompt[maskhigh,:] = green
        prompt[masklow,:] = red
        
        # make dictionary with stimulus information
        stim = {'name': name,
            'stimlum': stimlum.copy(), 'stimgrey': stimgrey.copy(), 'stimunit': stimunit.copy(),
            'maskref': masklow.copy(), 'maskmatch': maskhigh.copy(), 'prompt': prompt.copy()}
        stimlist.append(stim)
    
    def hasrepeat(x):
        "test whether any two consecutive stimuli have the same name"
        for i in range(len(x)-1):
            if x[i]['name'] == x[i+1]['name']:
                return True
        return False
    
    # repeat stimuli
    stimlist_repeat = []
    for i in range(nrepeat):
        stimlist_repeat.extend(copy.deepcopy(stimlist))

    # create random stimulus order without repeats on consecutive trials
    random.seed()
    while True:
        random.shuffle(stimlist_repeat)
        if not hasrepeat(stimlist_repeat): break
    
    return stimlist_repeat
