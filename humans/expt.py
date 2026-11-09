import sys, os, argparse, datetime, random
import numpy as np
from psychopy import visual, sound, event, core

# add tools directory to path
tooldir = os.path.join(os.getcwd(), 'tools')
if tooldir not in sys.path: sys.path.append(tooldir)
import calibrate, tools

## parse arguments
## - uncomment this section if you want to call the experiment from the
#    command line and provide subject initials as an argument; otherwise
#    set the 'initials' variable below to the observer's initials
#parser = argparse.ArgumentParser()
#parser.add_argument('--initials', type=str, default='test')
#args = parser.parse_args()
#initials = args.initials.lower()
initials = 'test'

# load calibration data
cal = calibrate.CalLum('tools/CalLum.pkl')

# load and process stimuli
stimlist = tools.loadstim(directory='illusions_isoluminant', cal=cal, nrepeat=5)

# open window and create a few useful objects
win = visual.Window(units='pix', fullscr=True)
im = visual.ImageStim(win)
timer = core.Clock()
beep = sound.Sound(value='C', octave=4, secs=0.1, volume=0.3)
mouse = event.Mouse(visible=False, newPos=[0,0])

def showstim(stim, matchgrey):
    "assign match greylevel and show stimulus"
    stim['stimunit'][stim['maskmatch']] = tools.grey2unit(matchgrey)
    im.image = stim['stimunit']
    im.size = stim['stimunit'].shape[0:2]
    im.draw()
    win.flip()

def showprompt(stim, dur=0.5):
    "show prompts at reference and match locations"
    im.image = stim['prompt']
    im.size = stim['prompt'].shape[0:2]
    im.draw()
    win.flip()
    core.wait(dur)

def mousegrey(rand=False):
    "get greylevel (0-255) from mouse position (y-coordinate)"
    limit = 500
    if rand: mouse.setPos((0, random.choice((-2*0.5*limit, 2*0.5*limit))))  # --- factor of two may not be necessary outside macOS
    grey = round(255 * (mouse.getPos()[1]+limit)/(2*limit))
    return max(min(grey,255),0)

# initialize data file
datafile = f'data/match_{initials}.csv'
def dataline(s):
    with open(datafile, 'a') as f: f.write(s)
dataline(f'# {datafile}\n')
dataline(f'# start: {datetime.datetime.now().isoformat(" ","seconds")}\n')
dataline('trial,name,refgrey,reflum,matchgrey,matchlum,rt\n')

# run trials
for trialnum, stim in enumerate(stimlist):
    
    # start trial
    matchgrey = mousegrey(rand=True)  # choose random initial match greylevel
    showprompt(stim, dur=1)
    timer.reset()
    
    # get observer's match setting
    while True:
        
        # get match setting and show stimulus
        matchgrey = mousegrey()
        showstim(stim, matchgrey)
        
        # show prompt
        if mouse.getPressed()[0]: showprompt(stim)
        
        # check for keypress responses (space = accept match, q = quit)
        keys = event.getKeys(keyList=['space','q'])
        if keys:
            rt = timer.getTime()
            break
    
    if keys[0]=='q': break
    
    # find greylevel and luminance of reference and match patches
    refgrey = stim['stimgrey'][stim['maskref']].mean()
    reflum = stim['stimlum'][stim['maskref']].mean()
    matchlum = cal.grey2lum(np.array(matchgrey))

    # save trial information
    dataline(f'{trialnum+1},{stim["name"]},{refgrey:.2f},{reflum:.2f},{matchgrey:.2f},{matchlum:.2f},{rt:.3f}\n')

# shut down
win.close()
dataline(f'# end: {datetime.datetime.now().isoformat(" ","seconds")}\n\n')
