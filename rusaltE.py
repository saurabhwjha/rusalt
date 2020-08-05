#!/usr/bin/env python
import sys

def load_modules():
    # Define a function to load all of the modules so that they don't' import 
    # unless we need them
    global iraf
    from pyraf import iraf
    iraf.pysalt()
    iraf.saltspec()
    iraf.saltred()
    iraf.set(clobber='YES')
    
    global sys
    import sys

    global os
    import os

    global shutil
    import shutil

    global glob
    from glob import glob
    
    global pyfits
    import pyfits

    global np
    import numpy as np
    
    global lacosmicx
    import lacosmicx
    
    global interp
    from scipy import interp
    
    global signal
    from scipy import signal
    
    global ndimage
    from scipy import ndimage
    
    global interpolate
    from scipy import interpolate
    
    global WCS
    from astropy.wcs import WCS
    
    global optimize
    from scipy import optimize
    
    global ds9
#    import pyds9 as ds9
    import ds9
    
    global GaussianProcess
    from sklearn.gaussian_process import GaussianProcess
    
    global pandas
    import pandas
    
    iraf.onedspec()
    iraf.twodspec()
    iraf.longslit()
    iraf.apextract()
    iraf.imutil()
    iraf.rvsao(motd='no')

# System specific path to pysalt
# pysaltpath = '/iraf/extern/pysalt'
pysaltpath = '/usr/local/astro64/iraf/extern/pysalt'

# Define the stages
allstages = ['sorting',
             'identify2d', 'rectify', 'slitnormalize', 'background', 'lax', 'fixpix',
             'extract', 'split1d','stdsensfunc', 'fluxcal','trim', 'speccombine',
             'mktelluric', 'telluric']


def tofits(filename, data, hdr=None, clobber=False):
    """simple pyfits wrapper to make saving fits files easier."""
    from pyfits import PrimaryHDU, HDUList
    hdu = PrimaryHDU(data)
    if hdr is not None:
        hdu.header = hdr
    hdulist = HDUList([hdu])
    hdulist.writeto(filename, clobber=clobber, output_verify='ignore')


def ds9display(filename):
    targs = ds9.ds9_targets()
    if targs is None:
        # Open a new ds9 window
        d = ds9.ds9(start=True)
    else:
        # Default grab the first ds9 instance
        d = ds9.ds9(targs[0])
    d.set('file ' + filename)
    d.set('zoom to fit')
    d.set('zscale')
    d.set("zscale contrast 0.1")

def run(files=None, dostages='all', stdsfolder='./', flatfolder=None, brightstar=False, fastred=False):
    # Load the modules if they aren't already.
    if not 'iraf' in sys.modules:
        load_modules()
    # Make sure the stages parameters makes sense
    try:
        print(dostages)
        if dostages == 'all':
            n0 = 0
            n = len(allstages)
        elif '-' in dostages:
            n0 = allstages.index(dostages.split('-')[0])
            n = allstages.index(dostages.split('-')[1])
        elif '+' in dostages:
            n0 = allstages.index(dostages[:-1])
            n = len(allstages)
        else:
            n0 = allstages.index(dostages)
            n = allstages.index(dostages)

    except:
        print "Please choose a valid stage."

    stages = allstages[n0:n + 1]

    if ',' in dostages:
        stages = dostages.split(',')

    print('Doing the following stages:')
    print(stages)

    for stage in stages:
        if stage == 'flatten':
            flatten(fs=files, masterflatdir=flatfolder)
        elif stage == 'sorting':
            sorting(fs=files,fastred=fastred)
        elif stage == 'lax':
            lax(fs=files,bright=brightstar)
        elif stage == 'fluxcal' or stage == 'telluric':
            globals()[stage](fs=files,stdsfolder=stdsfolder)
        else:
            globals()[stage](fs=files)


def get_chipgaps(hdu, deriv2=False):
        # Get the x coordinages of all of the chip gap pixels
        # recall that pyfits opens images with coordinates y, x
        # get the BPM from 51-950 which are the nominally good pixels
        # (for binning = 4 in the y direction)
        # (the default wavelength solutions are from 50.5 - 950.5)
        # [swj CHANGED this to use rows 250-750 to avoid potential bad rows]
        # Note this throws away one extra pixel on either side but it seems to
        # be necessary.

        # check binning in y and x directions
        ccdsum = int(hdu[0].header['CCDSUM'].split()[1])
        ccdxsum = int(hdu[0].header['CCDSUM'].split()[0])

        if ccdxsum != 2:
            sys.exit("Abort! Only bin by 2 in wavelength direction is supported. Data are bin by " + str(ccdxsum))
        
        if not deriv2:
            #ypix = slice(200 / ccdsum + 1, 3800 / ccdsum)  [swj CHANGE]
            ypix = slice(1000 / ccdsum + 1, 3000 / ccdsum)
            d = hdu[1].data[ypix].copy()
            bpm = hdu[2].data[ypix].copy()
            w = np.where(np.logical_or(bpm > 0, d == 0))[1]
            # Note we also grow the chip gap by 1 pixel on each side
            # Chip 1
            chipgap1 = (np.min(w[w > 700]) - 1, np.max(w[w < 1300]) + 1)
            # Chip 2
            chipgap2 = (np.min(w[w > 1750]) - 1, np.max(w[w < 2350]) + 1)
            # edge of chip 3
            chipgap3 = (np.min(w[w > 2900]) - 1, hdu[2].data.shape[1] + 1)
        else:
            # in the quicklook (nighttime or "fast") reductions the chipgaps are not set to zero
            # rather, they are linearly interpolated over. we can find them as the 2nd derivative == 0
            print("chipgaps: trying 2nd derivative method to find them, fast reduction data")
            ypix = slice(1980 / ccdsum + 1, 2020 / ccdsum)
            d = hdu[1].data[ypix].copy()
            bpm = hdu[2].data[ypix].copy()
            w = np.where(np.logical_or(bpm[:,1:-1] > 0, np.diff(d,n=2,axis=1) == 0.0))[1]
            chipgap1 = (np.min(w[w > 1003]) - 3, np.max(w[w < 1117]) + 3)
            chipgap2 = (np.min(w[w > 2083]) - 3, np.max(w[w < 2197]) + 3)
            chipgap3 = (np.min(w[w > 3133]) - 3, hdu[2].data.shape[1] + 1)
        
        # test if chipgaps look okay, if not, set them manually 
        if (chipgap1[0] >= chipgap1[1]) or (chipgap2[0] >= chipgap2[1]) or (chipgap3[0] >= chipgap3[1]):
            print("chipgaps: didn't work... try --fastred option? manually setting chipgaps")
            print("chipgaps: this should be conservative for PG0300 or PG0900 data")
            print("chipgaps: but watch out if you are using PG1800, PG2300, or PG3000")
            chipgap1 = (1000, 1120)
            chipgap2 = (2080, 2200)
            chipgap3 = (3130, hdu[2].data.shape[1] + 1)

        return (chipgap1, chipgap2, chipgap3)


def sorting(fs=None,fastred=False):
    # Run the pysalt pipeline on the reduced data.
    if fs is None:
        fs = sorted(glob('mbxgpP*.fits'))
    if len(fs) == 0:
        fs = sorted(glob('mbxpP*.fits'))
    if len(fs) == 0:
        print "WARNING: No product files to run PySALT pre-processing."
        return

    # Copy the raw files into a raw directory
    if not os.path.exists('product'):
        os.mkdir('product')
    if not os.path.exists('work'):
        os.mkdir('work')
    if not os.path.exists('work/srt'):
        os.mkdir('work/srt')
    for f in fs:
        shutil.copy2(f, 'product/')
    scifs, scigas = get_ims(fs, 'sci')
    arcfs, arcgas = get_ims(fs, 'arc')

    ims = np.append(scifs, arcfs)
    gas = np.append(scigas, arcgas)

    for i, f in enumerate(ims):
        ga = gas[i]
        if f in scifs:
            typestr = 'sci'
        else:
            typestr = 'arc'
        # by our naming convention, imnum should be the last 4 characters
        # before the '.fits'
        imnum = f[-9:-5]
        outname = 'srt/' + typestr
        outname += '%05.2fmos%04i.fits' % (float(ga), int(imnum))
        shutil.move(f, 'work/'+outname)
        iraf.cd('work') 
        h = pyfits.open(outname, 'update')
        maskim = h[1].data.copy()
        maskim[:, :] = 0.0
        maskim[abs(h[1].data) < 1e-5] = 1
        imhdu = pyfits.ImageHDU(maskim)

        h.append(imhdu)
        h[1].header['BPMEXT'] = 2
        h[2].header['EXTNAME'] = 'BPM'
        h[2].header['CD2_2'] = 1
        h.flush()

        if fastred:
            # the quicklook pipeline interpolates over the chip gaps
            # here we try to set the chip gaps to zero instead
            print(outname)
            chipgaps = get_chipgaps(h, deriv2=True)
            print (" -- srt chipgaps --")
            print (chipgaps)
            # Chip 1
            h[2].data[:, chipgaps[0][0]:chipgaps[0][1]] = 1
            # Chip 2
            h[2].data[:, chipgaps[1][0]:chipgaps[1][1]] = 1
            # edge of chip 3
            h[2].data[:, chipgaps[2][0]:chipgaps[2][1]] = 1
            # Cover the other blank regions
            h[2].data[[h[1].data == 0]] = 1
            # Set all of the data to zero in the BPM
            h[1].data[h[2].data == 1] = 0.0
            h.flush()

        h.close()
        iraf.cd('..')

def get_ims(fs, imtype):
    imtypekeys = {'sci': 'OBJECT', 'arc': 'ARC', 'flat': 'FLAT'}
    ims = []
    grangles = []
    for f in fs:
        if pyfits.getval(f, 'OBSTYPE') == imtypekeys[imtype]:
            ims.append(f)
            grangles.append(pyfits.getval(f, 'GR-ANGLE'))
    return np.array(ims), np.array(grangles)


def get_scis_and_arcs(fs):
    scifs, scigas = get_ims(fs, 'sci')
    arcfs, arcgas = get_ims(fs, 'arc')

    ims = np.append(scifs, arcfs)
    gas = np.append(scigas, arcgas)
    return ims, gas


def identify2d(fs=None):
    iraf.cd('work')
    if fs is None:
        fs = sorted(glob('srt/arc*mos*.fits'))
    if len(fs) == 0:
        print "WARNING: No mosaiced (2D) specidentify."
        # Change directories to fail gracefully
        iraf.cd('..')
        return
    arcfs, arcgas = get_ims(fs, 'arc')
    if not os.path.exists('id2'):
        os.mkdir('id2')

    lampfiles = {'Th Ar': 'ThAr.salt', 'Xe': 'Xe.salt', 'Ne': 'NeAr.salt',
                 'Cu Ar': 'CuAr.salt', 'Ar': 'Argon_hires.salt',
                 'Hg Ar': 'HgAr.salt'}
    for i, f in enumerate(arcfs):
        ga = arcgas[i]

        # find lamp and corresponding linelist
        lamp = pyfits.getval(f, 'LAMPID')
        lampfn = lampfiles[lamp]
        if pyfits.getval(f,'GRATING') == 'PG0300' and lamp == 'Ar':
            lampfn = 'Argon_lores.swj'

        ccdsum = int(pyfits.getval(f, 'CCDSUM').split()[1])

        # linelistpath is a global variable defined in beginning, path to
        # where the line lists are.
        lamplines = pysaltpath + '/data/linelists/' + lampfn
        print(lamplines)

        # img num should be right before the .fits
        imgnum = f[-9:-5]
        # run pysalt specidentify
        idfile = 'id2/arc%05.2fid2%04i' % (float(ga), int(imgnum)) + '.db'
        iraf.unlearn(iraf.specidentify)
        iraf.flpr()
        iraf.specidentify(images=f, linelist=lamplines, outfile=idfile,
                          guesstype='rss', inter=True, # automethod='FitXcor',
                          rstep= -1720 / ccdsum,
                          rstart=2000 / ccdsum, startext=1, clobber='yes',
                          #startext=1, clobber='yes',
                          verbose='no', mode='hl', logfile='salt.log',
                          mdiff=2, function='legendre')
    iraf.cd('..')


def rectify(ids=None, fs=None):
    iraf.cd('work')
    if ids is None:
        ids = np.array(sorted(glob('id2/arc*id2*.db')))
    if fs is None:
        fs = sorted(glob('srt/*mos*.fits'))
    if len(ids) == 0:
        print "WARNING: No wavelength solutions for rectification."
        iraf.cd('..')
        return
    if len(fs) == 0:
        print "WARNING: No images for rectification."
        iraf.cd('..')
        return

    # Get the grating angles of the solution files
    idgas = []
    for i, thisid in enumerate(ids):
        f = open(thisid)
        idlines = np.array(f.readlines(), dtype=str)
        f.close()
        idgaline = idlines[np.char.startswith(idlines, '#graang')][0]
        idgas.append(float(idgaline.split('=')[1]))

    ims, gas = get_scis_and_arcs(fs)
    print('_____idgas_____')
    print (np.array(idgas))
    print('_____ga_____')
    print (gas)


    if not os.path.exists('rec'):
        os.mkdir('rec')
    for i, f in enumerate(ims):
        fname = f.split('/')[1]
        typestr = fname[:3]
        ga, imgnum = gas[i], fname[-9:-5]

        outfile = 'rec/' + typestr + '%05.2frec' % (ga) + imgnum + '.fits'
        iraf.unlearn(iraf.specrectify)
        iraf.flpr()
        # idfiles = ids[np.array(idgas) == ga]
        # if len(idfiles)==0:
        #     print "WARNING: No wavelength solution for GR-ANGLE=" + "%f" % (ga)
        #     break
        # idfile = idfiles[0]
        # print(fname,idfile)
        idfiles = ids[np.abs(np.array(idgas) - ga) < 0.019]
        if len(idfiles)==0:
            print "WARNING: No wavelength solution for GR-ANGLE=" + "%f" % (ga)
            break
        idfile = idfiles[0]
        iraf.specrectify(images=f, outimages=outfile, solfile=idfile,
                         outpref='', function='legendre', order=3,
                         inttype='interp', conserve='yes', clobber='yes',
                         verbose='yes')

        pyfits.setval(outfile, 'WAVESOLN', extname='SCI', value=idfile, comment='RUSALT pipeline identify2d')

        # Update the BPM to mask any blank regions
        h = pyfits.open(outfile, 'update')
        # Cover the chip gaps. The background task etc do better if the chip
        # gaps are straight
        # To deal with this we just throw away the min and max of each side of
        # the curved chip gap
        chipgaps = get_chipgaps(h)
        print (" -- chipgaps --")
        print (chipgaps)

        # Chip 1
        h[2].data[:, chipgaps[0][0]:chipgaps[0][1]] = 1
        # Chip 2
        h[2].data[:, chipgaps[1][0]:chipgaps[1][1]] = 1
        # edge of chip 3
        h[2].data[:, chipgaps[2][0]:chipgaps[2][1]] = 1
        # Cover the other blank regions
        h[2].data[[h[1].data == 0]] = 1

        # Set all of the data to zero in the BPM
        h[1].data[h[2].data == 1] = 0.0
        h.flush()
        h.close()
    iraf.cd('..')


def slitnormalize(fs=None):
    iraf.cd('work')
    if fs is None:
        fs = sorted(glob('rec/*rec*.fits'))
    if len(fs) == 0:
        print "WARNING: No rectified files for slitnormalize."
        # Change directories to fail gracefully
        iraf.cd('..')
        return
    if not os.path.exists('nrm'):
        os.mkdir('nrm')

    for f in fs:
        outname = f.replace('rec', 'nrm')
        iraf.unlearn(iraf.specslitnormalize)
        iraf.specslitnormalize(images=f, outimages=outname, outpref='',
                               order=5, clobber=True, mode='h')

    iraf.cd('..')


def background(fs=None):
    iraf.cd('work')
    # Get rectified science images
    if fs is None:
        fs = sorted(glob('nrm/sci*nrm*.fits'))
    if len(fs) == 0:
        print "WARNING: No rectified images for background-subtraction."
        iraf.cd('..')
        return

    if not os.path.exists('bkg'):
        os.mkdir('bkg')

    for f in fs:
        print("Subtracting background for %s" % f)
        # Make sure dispaxis is set correctly
        pyfits.setval(f, 'DISPAXIS', value=1)

        # the outfile name is very similar, just change folder prefix and
        # 3-char stage substring
        outfile = f.replace('nrm','bkg')
        # We are going to use fit1d instead of the background task
        # Go look at the code for the background task: it is literally a wrapper for 1D
        # but it removes the BPM option. Annoying.
        iraf.unlearn(iraf.fit1d)
        iraf.fit1d(input=f + '[SCI]', output='tmpbkg.fits', bpm=f + '[BPM]',
                   type='difference', sample='52:949', axis=2,
                   interactive='no', naverage='1', function='legendre',
                   order=5, low_reject=1.0, high_reject=1.0, niterate=5,
                   grow=0.0, mode='hl')

        # Copy the background subtracted frame into the rectified image
        # structure.
        # Save the sky spectrum as extension 3
        hdutmp = pyfits.open('tmpbkg.fits')
        hdu = pyfits.open(f)
        skydata = hdu[1].data - hdutmp[0].data
        hdu[1].data[:, :] = hdutmp[0].data[:, :]

        hdu.append(pyfits.ImageHDU(skydata))
        hdu[3].header['EXTNAME'] = 'SKY'
        hdu[3].data[hdu['BPM'] == 1] = 0.0

        # Add back in the median sky level for things like apall and lacosmicx
        hdu[1].data[:, :] += np.median(skydata)
        hdu[1].data[hdu['BPM'] == 1] = 0.0
        hdutmp.close()
        hdu.writeto(outfile, clobber=True)  # saving the updated file
        # (data changed)
        os.remove('tmpbkg.fits')
    iraf.cd('..')


def isstdstar(f):
    # get the list of standard stars
    stdslist = sorted(glob(pysaltpath + '/data/standards/spectroscopic/*'))
    objname = pyfits.getval(f, 'OBJECT').lower().replace('-','_')
    for std in stdslist:
        if objname in std:
            return True

    # Otherwise not in the list so return false
    return False


def lax(fs=None,bright=False):
    iraf.cd('work')
    if fs is None:
        fs = sorted(glob('bkg/*bkg*.fits'))
    if len(fs) == 0:
        print "WARNING: No background-subtracted files for Lacosmicx."
        iraf.cd('..')
        return

    if not os.path.exists('lax'):
        os.mkdir('lax')
    for f in fs:
        outname = f.replace('bkg','lax')
        hdu = pyfits.open(f)

        # Add a CRM extension
        hdu.append(pyfits.ImageHDU(data=hdu['BPM'].data.copy(),
                                   header=hdu['BPM'].header.copy(),
                                   name='CRM'))
        # Set all of the pixels in the CRM mask to zero
        hdu['CRM'].data[:, :] = 0

        # less aggressive lacosmic on bright objects or standard star observations
        if bright or isstdstar(f):
            print("bright star: less aggressive lacosmicx parameters used")
            objl = 3.0 
            sigc = 10.0
        else:
            objl = 1.0
            sigc = 4.0

        chipgaps = get_chipgaps(hdu)

        chipedges = [[0, chipgaps[0][0]], [chipgaps[0][1] + 1, 
                             chipgaps[1][0]], [chipgaps[1][1] + 1, chipgaps[2][0]]]

        # Run each chip separately
        for chip in range(3):
            # Use previously subtracted sky level = 0 as we have already added
            # a constant sky value in the background task
            # Gain = 1, readnoise should be small so it shouldn't matter much.
            # Default value seems to work.
            chipinds = slice(chipedges[chip][0], chipedges[chip][1])
            crmask, _cleanarr = lacosmicx.lacosmicx(hdu[1].data[:, chipinds].copy(),
                                  inmask=np.asarray(hdu[2].data[:, chipinds].copy(), dtype = np.uint8), sigclip=sigc,
                                 objlim=objl, sigfrac=0.1, gain=1.0, pssl=0.0)


            # Update the image
            hdu['CRM'].data[:, chipinds][:, :] = crmask[:,:]
            # Flag the cosmic ray pixels with a large negative number
            hdu['SCI'].data[:, chipinds][crmask == 1] = -1000000

        # Save the file
        hdu.writeto(outname, clobber=True)
        hdu.close()

    iraf.cd('..')


def fixpix(fs=None):
    iraf.cd('work')
    if fs is None:
        fs = sorted(glob('nrm/sci*nrm*.fits'))
    if len(fs) == 0:
        print "WARNING: No rectified images to fix."
        iraf.cd('..')
        return
    if not os.path.exists('fix'):
        os.mkdir('fix')
    for f in fs:
        outname = f.replace('nrm', 'fix')
        # Copy the file to the fix directory
        shutil.copy(f, outname)
        # Set all of the BPM pixels = 0
        h = pyfits.open(outname, mode='update')
        h['SCI'].data[h['BPM'].data == 1] = 0
        # Grab the CRM extension from the lax file
        laxhdu = pyfits.open(f.replace('nrm', 'lax'))
        h.append(pyfits.ImageHDU(data=laxhdu['CRM'].data.copy(),
                                 header=laxhdu['CRM'].header.copy(),
                                 name='CRM'))
        h.flush()
        h.close()
        laxhdu.close()

        # Run iraf's fixpix on the cosmic rays, not ideal,
        # but better than nothing because apall doesn't take a bad pixel mask
        iraf.unlearn(iraf.fixpix)
        iraf.flpr()
        iraf.fixpix(outname + '[SCI]', outname + '[CRM]', mode='hl')
    iraf.cd('..')


def extract(fs=None):
    iraf.cd('work')
    if fs is None:
        fs = sorted(glob('fix/*fix*.fits'))
    if len(fs) == 0:
        print "WARNING: No fixpixed images available for extraction."
        iraf.cd('..')
        return

    if not os.path.exists('x1d'):
        os.mkdir('x1d')

    print "Note: No continuum? Make nsum small (~-5) with 'line' centered on an emission line."
    for f in fs:
        # Get the output filename without the ".fits"
        outbase = f.replace('fix', 'x1d')[:-5]
        # Get the readnoise, right now assume default value of 5 but we could
        # get this from the header
        readnoise = 5
        # If interactive open the rectified, background subtracted image in ds9
        ds9display(f.replace('fix', 'bkg'))
        # set dispaxis = 1 just in case
        pyfits.setval(f, 'DISPAXIS', extname='SCI', value=1)
        iraf.unlearn(iraf.apall)
        iraf.flpr()
        iraf.apall(input=f + '[SCI]', output=outbase, interactive='yes',
                   review='no', line='INDEF', nsum=-1000, lower=-5, upper=5,
                   b_function='legendre', b_order=5,
                   b_sample='-200:-50,50:200', b_naverage=-10, b_niterate=5,
                   b_low_reject=3.0, b_high_reject=3.0, nfind=1, t_nsum=15,
                   t_step=15, t_nlost=200, t_function='legendre', t_order=5,
                   t_niterate=5, t_low_reject=3.0, t_high_reject=3.0,
                   background='fit', weights='variance', pfit='fit1d',
                   clean='no', readnoise=readnoise, gain=1.0, lsigma=4.0,
                   usigma=4.0, mode='hl')

        # Copy the CCDSUM keyword into the 1d extraction
        pyfits.setval(outbase + '.fits', 'CCDSUM',
                      value=pyfits.getval(f, 'CCDSUM'))

        # Extract the corresponding arc
        arcdbfn = (pyfits.getval(f, 'WAVESOLN', extname='SCI')).split('/')[1]
        arcnames = glob('nrm/' + arcdbfn[0:8] + 'nrm' + arcdbfn[11:15] + '.fits')
        if len(arcnames) == 0:
            print("WARNING: no associated arc file?? for " + f)
            break
        arcname = arcnames[0]
        print("Extracing corresponding arc: " + arcname)
        # set dispaxis = 1 just in case
        pyfits.setval(arcname, 'DISPAXIS', extname='SCI', value=1)
        iraf.unlearn(iraf.apsum)
        iraf.flpr()
        iraf.apsum(input=arcname + '[SCI]', output='auxext_arc',
                   references=f[:-5] + '[SCI]', interactive='no', find='no',
                   edit='no', trace='no', fittrace='no', extras='no',
                   review='no', background='no', mode='hl')
        # copy the arc into the 5 column of the data cube
        arcfs = sorted(glob('auxext_arc*.fits'))
        for af in arcfs:
            archdu = pyfits.open(af)
            scihdu = pyfits.open(outbase + '.fits', mode='update')
            d = scihdu[0].data.copy()
            scihdu[0].data = np.zeros((5, d.shape[1], d.shape[2]))
            scihdu[0].data[:-1, :, :] = d[:, :, :]
            scihdu[0].data[-1::, :] = archdu[0].data.copy()
            scihdu.flush()
            scihdu.close()
            archdu.close()
            os.remove(af)
        # Add the airmass, exptime, and other keywords back into the
        # extracted spectrum header
        kws = ['AIRMASS','EXPTIME',
               'PROPID','PROPOSER','OBSERVER','OBSERVAT','SITELAT','SITELONG',
               'INSTRUME','DETSWV','RA','PM-RA','DEC','PM-DEC','EQUINOX',
               'EPOCH','DATE-OBS','TIME-OBS','UTC-OBS','TIMESYS','LST-OBS',
               'JD','MOONANG','OBSMODE','DETMODE','SITEELEV','BLOCKID','PA',
               'TELHA','TELRA','TELDEC','TELPA','TELAZ','TELALT','DECPANGL',
               'TELTEM','PAYLTEM','MASKID','MASKTYP','GR-ANGLE','GRATING',
               'FILTER'] 
        for kw in kws:
            pyfits.setval(outbase + '.fits', kw, value=pyfits.getval(f,kw))

    iraf.cd('..')


def split1d(fs=None):
    iraf.cd('work')
    if fs is None:
        fs = sorted(glob('x1d/sci*x1d????.fits'))
    if len(fs) == 0:
        print "WARNING: No extracted spectra to split."
        iraf.cd('..')
        return

    for f in fs:
        hdu = pyfits.open(f.replace('x1d', 'fix'))
        chipgaps = get_chipgaps(hdu)
        # Throw away the first pixel as it almost always bad
        chipedges = [[1, chipgaps[0][0]], [chipgaps[0][1] + 1, chipgaps[1][0]],
                     [chipgaps[1][1] + 1, chipgaps[2][0]]]

        w = WCS(f)
        # Copy each of the chips out seperately. Note that iraf is 1 indexed
        # unlike python so we add 1
        for i in range(3):
            # get the wavelengths that correspond to each chip
            lam, _apnum, _bandnum = w.all_pix2world(chipedges[i], 0, 0, 0)
            iraf.scopy(f, f[:-5] + 'c%i' % (i + 1), w1=lam[0], w2=lam[1],
                       format='multispec', rebin='no',clobber='yes')
        hdu.close()
    iraf.cd('..')


def spectoascii(fname, asciiname, ap=0):
    hdu = pyfits.open(fname)
    w = WCS(fname)
    print ('-----w-----')
    print(w)
    # get the wavelengths of the pixels
    npix = hdu[0].data.shape[2]
    print('-----npix-----')
    print (npix)
    lam = w.all_pix2world(np.linspace(0, npix - 1, npix), 0, 0, 0)[0]
    spec = hdu[0].data[0, ap]
    specerr = hdu[0].data[3, ap]
    np.savetxt(asciiname, np.array([lam, spec, specerr]).transpose())
    hdu.close()


def stdsensfunc(fs=None):
    iraf.cd('work')
    if fs is None:
        fs = sorted(glob('x1d/sci*x1d*c?.fits'))
    if len(fs) == 0:
        print "WARNING: No extracted spectra to create sensfuncs from."
        iraf.cd('..')
        return

    if not os.path.exists('std'):
        os.mkdir('std')
    for f in fs:
        # Put the file in the std directory, but last 3 letters of sens
        outfile = 'std/' + f.split('/')[1]
        outfile = outfile.replace('x1d', 'sens').replace('sci', 'std')
        outfile = outfile.replace('.fits', '.dat')
        # if the object name is in the list of standard stars from pysalt
        if isstdstar(f):
            # We use pysalt here because standard requires a
            # dispersion correction which was already taken care of above
            # Write out an ascii file that pysalt.specsens can read
            asciispec = 'std/std.ascii.dat'
            spectoascii(f, asciispec)
            # run specsens
            stdfile = pysaltpath + '/data/standards/spectroscopic/m%s.dat' % pyfits.getval(f, 'OBJECT').lower().replace('-','_')
            extfile = pysaltpath + '/data/site/suth_extinct.dat'
            iraf.unlearn(iraf.specsens)
            iraf.specsens(asciispec, outfile, stdfile, extfile,
                          airmass=pyfits.getval(f, 'AIRMASS'), fitter='gaussian',
                          exptime=pyfits.getval(f, 'EXPTIME'), function='poly',
                          order=3, niter=3, clobber=True, mode='h', thresh=10)
            # delete the ascii file
            S= np.genfromtxt(outfile,skip_header=40,skip_footer=40)
            np.savetxt(outfile,S)
            os.remove(asciispec)
            
    iraf.cd('..')


def fluxcal(stdsfolder='./', fs=None):
    iraf.cd('work')
    if fs is None:
        fs = sorted(glob('x1d/sci*x1d*c*.fits'))
    if len(fs) == 0:
        print "WARNING: No science chip spectra to flux calibrate."
        iraf.cd('..')
        return

    if not os.path.exists('flx'):
        os.mkdir('flx')
    extfile = pysaltpath + '/data/site/suth_extinct.dat'
    stdfiles = sorted(glob(stdsfolder + '/std/*sens*c?.dat'))
    for f in fs:
        outfile = f.replace('x1d', 'flx')
        chip = outfile[-6]
        hdu = pyfits.open(f)
        ga = float(f.split('/')[1][3:8])
        # Get the standard sensfunc with the same grating angle
        stdfile = None
        for stdf in stdfiles:
            stdfga = float(stdf.split('/')[-1][3:8])
            if np.abs(ga-stdfga) < 0.019:
                # Get the right chip number
                if chip == stdf[-5]:
                    stdfile = stdf
                    break
        if stdfile is None:
            print('No standard star with grating-angle close enough to %f' % ga)
            continue
        # for each extracted aperture
        print("Flux calibrating " + f + " with " + stdfile)
        for i in range(hdu[0].data.shape[1]):
            # create an ascii file that pysalt can read
            asciiname = 'flx/sciflx.dat'
            outtmpname = 'flx/scical.dat'
            spectoascii(f, asciiname, i)
            # Run pysalt.speccal
            iraf.unlearn(iraf.speccal)
            iraf.flpr()
            iraf.speccal(asciiname, outtmpname, stdfile, extfile,
                         airmass=pyfits.getval(f, 'AIRMASS'),
                         exptime=pyfits.getval(f, 'EXPTIME'),
                         clobber=True, mode='h')
            # read in the flux calibrated ascii file and copy its
            # contents into a fits file
            flxcal = np.genfromtxt(outtmpname).transpose()
            hdu[0].data[0, i] = flxcal[1]
            hdu[0].data[2, i] = flxcal[2]
            # delete the ascii file
            os.remove(asciiname)
            os.remove(outtmpname)
        hdu.writeto(outfile, clobber=True)
    iraf.cd('..')


def combine_spec_chi2(p, lam, specs, specerrs):
    # specs should be an array with shape (nspec, nlam)
    nspec = specs.shape[0]
    # scale each spectrum by the given value

    scaledspec = (specs.transpose() * p).transpose()
    scaled_spec_err = (specerrs.transpose() * p).transpose()

    chi = 0.0
    # loop over each pair of spectra
    for i in range(nspec):
        for j in range(i + 1, nspec):
            # Calculate the chi^2 for places that overlap
            # (i.e. spec > 0 in both)
            w = np.logical_and(scaledspec[i] != 0.0, scaledspec[j] != 0)
            if w.sum() > 0:
                residuals = scaledspec[i][w] - scaledspec[j][w]
                errs2 = scaled_spec_err[i][w] ** 2.0
                errs2 += scaled_spec_err[j][w] ** 2.0
                chi += (residuals ** 2.0 / errs2).sum()
    return chi
#change to pixels
def trim(fs=None):
	iraf.cd('work')
	if fs is None:
		fs=sorted(glob('flx/sci*c?.fits'))
	if not os.path.exists('trm'):
		os.mkdir('trm')
	for i,f in enumerate(fs):
		outfile=f.replace('flx','trm')
		w=WCS(f)
		hdu= pyfits.open(f)
		npix=hdu[0].data.shape[2]
		W1=int(w.wcs_pix2world(2,0,0,0)[0])
		W2=int(w.wcs_pix2world(npix-2,0,0,0)[0])
		iraf.sarith(input1=f,op='copy',output=outfile,w1=W1,w2=W2
		,clobber='yes')
	iraf.cd('..')
def scopy():
    if not os.path.exists('flx/test/'):
        os.mkdir(Path+'flx/test/')
    fs= sorted(glob('flx/sci*c?.fits'))
    for i,f in enumerate(fs):
        iraf.scopy(f+'[*,1,1]','flx/test/'+'sci'+str(i)+'.fits')

def wspectext():
    if not os.path.exists('flx/test3/'):
        os.mkdir('flx/test3/')
    fs= sorted(glob('flx/test/'+'/sci*.fits'))
    iraf.cd('flx/test/')
    for i,f in enumerate(fs):
        iraf.wspectext(f,('sci'+str(i)+'.dat'))
    iraf.cd('..')
    iraf.cd('..')
def diagnostic():
	import matplotlib.pyplot as plt

	a= 'sci0.dat'
	b= 'sci1.dat'
	c= 'sci2.dat'
	d= 'sci3.dat'
	e= 'sci4.dat'
	f= 'sci5.dat'
	g= 'sci6.dat'
	h= 'sci7.dat'
	i= 'sci8.dat'
	j= 'sci9.dat'
	k= 'sci10.dat'
	l= 'sci11.dat'

	A= 'flx/test3/'+a
	B= 'flx/test3/'+b
	C= 'flx/test3/'+c
	D= 'flx/test3/'+d
	E= 'flx/test3/'+e
	F= 'flx/test3/'+f
	G= 'flx/test3/'+g
	H= 'flx/test3/'+h
	I= 'flx/test3/'+i
	J= 'flx/test3/'+j
	K= 'flx/test3/'+k
	L= 'flx/test3/'+l

	plt.plot(*np.loadtxt(A,unpack=True), linewidth=2.0, label=a)
	plt.plot(*np.loadtxt(B,unpack=True), linewidth=2.0, label=b)
	plt.plot(*np.loadtxt(C,unpack=True), linewidth=2.0, label=c)
	plt.plot(*np.loadtxt(D,unpack=True), linewidth=2.0, label=d)
	plt.plot(*np.loadtxt(E,unpack=True), linewidth=2.0, label=e)
	plt.plot(*np.loadtxt(F,unpack=True), linewidth=2.0, label=f)
	plt.plot(*np.loadtxt(G,unpack=True), linewidth=2.0, label=g)
	plt.plot(*np.loadtxt(H,unpack=True), linewidth=2.0, label=h)
	plt.plot(*np.loadtxt(I,unpack=True), linewidth=2.0, label=i)
	plt.plot(*np.loadtxt(J,unpack=True), linewidth=2.0, label=j)
	plt.plot(*np.loadtxt(K,unpack=True), linewidth=2.0, label=k)
	plt.plot(*np.loadtxt(L,unpack=True), linewidth=2.0, label=l)

	plt.legend()
	plt.pause(0.001)
	plt.show()
	return
def speccombine(fs=None):
    iraf.cd('work')
    if fs is None:
        fs = sorted(glob('trm/sci*c?.fits'))
    if len(fs)==0:
        print("No flux calibrated images to combine.")
        iraf.cd('..')
        return
    #diagnostic()
    nsteps = 8001
    lamgrid = np.linspace(2000.0, 10000.0, nsteps)

    nfs = len(fs)
    # for each aperture
    # get all of the science images
    specs = np.zeros((nfs, nsteps))
    specerrs = np.zeros((nfs, nsteps))
    ap = 0
    for i, f in enumerate(fs):
        hdu = pyfits.open(f)
    #	print ('---hdu.data---')
    #	print (hdu[0].data)
    	w=WCS(f)
    # 	print ('-----w-----')
    #	print(w)
        # get the wavelengths of the pixels
        npix = hdu[0].data.shape[2]
    #	print('-----npix-----')
    #	print(npix)
        lam = w.all_pix2world(np.linspace(0, npix - 1, npix), 0, 0, 0)[0]
    #	print('-----lam-----')
    #	print(lam)
        # interpolate each spectrum onto a comman wavelength scale

        specs[i] = interp(lamgrid, lam, hdu[0].data[0][ap],
                          left=0.0, right=0.0)
        # Also calculate the errors. Right now we assume that the variances
        # interpolate linearly. This is not stricly correct but it should be
        # close. Also we don't include terms in the variance for the
        # uncertainty in the wavelength solution.
        specerrs[i] = interp(lamgrid, lam, hdu[0].data[3][ap] ** 2.0) ** 0.5
    #print ('-----specs-----')
    #print (specs)
    # minimize the chi^2 given free parameters are multiplicative factors
    # We could use linear or quadratic, but for now assume constant
    p0 = np.ones(nfs)

    results = optimize.minimize(combine_spec_chi2, p0,
                                args=(lamgrid, specs, specerrs),
                                method='Nelder-Mead',
                                options={'maxfev': 1e5, 'maxiter': 1e5})

    # write the best fit parameters into the headers of the files
    # Dump the list of spectra into a string that iraf can handle
    iraf_filelist = str(fs).replace('[', '').replace(']', '').replace("'", '')

    # write the best fit results into a file
    lines = []
    for p in results['x']:
        lines.append('%f\n' % (1.0 / p))
    f = open('flx/scales.dat', 'w')
    f.writelines(lines)
    f.close()
    # run scombine after multiplying the spectra by the best fit parameters
    combfile = 'sci_com.fits'
    if os.path.exists(combfile):
        os.remove(combfile)
    iraf.scombine(iraf_filelist, combfile, scale='@flx/scales.dat',
                  reject='avsigclip', lthreshold=-1e-17)

    # Remove the other apertures [TBD]
    # remove the sky and arc bands from the combined spectra. (or add back?? TBD)

    # remove some header keywords that don't make sense in the combined file
    delkws = ['GR-ANGLE','FILTER','BANDID2','BANDID3','BANDID4']
    for kw in delkws:
        pyfits.delval(combfile,kw)

    # combine JD (average), AIRMASS (average), EXPTIME (sum)
    #   we assume there is a c1.fits file for each image
    c1fs = [f for f in fs if 'c1.fits' in f]
    avgjd = np.mean([pyfits.getval(f,'JD') for f in c1fs])
    pyfits.setval(combfile,'JD',value=avgjd,comment='average of multiple exposures')
    print "average JD = " + str(avgjd)
    sumet = np.sum([pyfits.getval(f,'EXPTIME') for f in c1fs])
    pyfits.setval(combfile,'EXPTIME',value=sumet,comment='sum of multiple exposures')
    print "total EXPTIME = " + str(sumet)
    avgam = np.mean([pyfits.getval(f,'AIRMASS') for f in c1fs])
    pyfits.setval(combfile,'AIRMASS',value=avgam,comment='average of multiple exposures')
    print "avg AIRMASS = " + str(avgam)

    # update this to used avg jd midpoint of all exposures? 
    print "barycentric velocity correction (km/s) = ", 
    iraf.bcvcorr(spectra=combfile,keytime='UTC-OBS',keywhen='mid',
                 obslong="339:11:16.8",obslat="-32:22:46.2",obsalt='1798',obsname='saao', 
                 savebcv='yes',savejd='yes',printmode=2)
    pyfits.setval(combfile,'UTMID',comment='added by RVSAO task BCVCORR')
    pyfits.setval(combfile,'GJDN',comment='added by RVSAO task BCVCORR')
    pyfits.setval(combfile,'HJDN',comment='added by RVSAO task BCVCORR')
    pyfits.setval(combfile,'BCV',comment='added by RVSAO task BCVCORR (km/s)')
    pyfits.setval(combfile,'HCV',comment='added by RVSAO task BCVCORR (km/s)')
    iraf.dopcor(input=combfile,output='',redshift=-iraf.bcvcorr.bcv,isvelocity='yes',
                add='no',dispersion='yes',flux='no',verbose='yes')
    pyfits.setval(combfile,'DOPCOR01',comment='barycentric velocity correction applied')

    iraf.cd('..')


# Define the telluric bands wavelength regions
# These numbers were taken directly from Tom Matheson's Cal code from Jeff
# Silverman
#telluricWaves = {'B': (6855, 6935), 'A': (7590, 7685)}
#telluricWaves = [(2000., 3190.), (3216., 3420.), (5500., 6050.), (6250., 6360.),
#                 (6450., 6530.), (6840., 7410.), (7560., 8410.), (8800., 9900.)]
telluricWaves = [(6250., 6360.), (6450., 6530.), (6855., 7400.), (7580., 7720.)]


def fitshdr_to_wave(hdr):
    crval = float(hdr['CRVAL1'])
    cdelt = float(hdr['CDELT1'])
    nlam = float(hdr['NAXIS1'])
    lam = np.arange(crval, crval + cdelt * nlam - 1e-4, cdelt)
    return lam


def mktelluric(fs=None):
    iraf.cd('work')
    if fs is None:
        fs = sorted(glob('sci_com.fits'))
    if len(fs) == 0:
        print "WARNING: No flux-calibrated spectra to make the a telluric correction."
        iraf.cd('..')
        return
        
    if not os.path.exists('tel'):
        os.mkdir('tel')
        
    # for each file
    f = fs[0]
    # if it is a standard star combined file
    if isstdstar(f):
        # read in the spectrum and calculate the wavelengths of the pixels
        hdu = pyfits.open(f)
        spec = hdu[0].data.copy()
        hdr = hdu[0].header.copy()
        hdu.close()
        waves = fitshdr_to_wave(hdr)
        
        template_spectrum = signal.savgol_filter(spec, 21, 3)
        noise = np.abs(spec - template_spectrum)
        noise = ndimage.filters.gaussian_filter1d(noise, 100.0)
        not_telluric = np.ones(spec.shape, dtype=np.bool)
        # For each telluric region
        for wavereg in telluricWaves:
            in_telluric_region = np.logical_and(waves >= wavereg[0],
                                                waves <= wavereg[1])
            not_telluric = np.logical_and(not_telluric,
                                             np.logical_not(in_telluric_region))
        
        # Smooth the spectrum so that the spline doesn't go as crazy
        # Use the Savitzky-Golay filter to presevere the edges of the
        # absorption features (both atomospheric and intrinsic to the star)
        sgspec = signal.savgol_filter(spec, 31, 3)
        #Get the number of data points to set the smoothing criteria for the 
        # spline
        m = not_telluric.sum()
        intpr = interpolate.splrep(waves[not_telluric], sgspec[not_telluric], 
                                   w = 1/noise[not_telluric], k=2,  s=10 * m)

         # Replace the telluric with the smoothed function
        smoothedspec = interpolate.splev(waves, intpr)
        smoothedspec[not_telluric] = spec[not_telluric]
        # Divide the original and the telluric corrected spectra to
        # get the correction factor
        correction = spec / smoothedspec

        # Save the correction
        dout = np.ones((2, len(waves)))
        dout[0] = waves
        dout[1] = correction
        np.savetxt('tel/telcor.dat', dout.transpose())
            
    iraf.cd('..')


def telluric(stdsfolder='./', fs=None):
    iraf.cd('work')
    if fs is None:
        fs = glob('sci_com.fits')
    if len(fs) == 0:
        print "WARNING: No flux-calibrated spectra to telluric-correct."
        iraf.cd('..')
        return

    f = fs[0]
    outfile = 'final.fits'
    # Get the standard to use for telluric correction
    stdfile = glob(stdsfolder + '/tel/telcor.dat')[0]
    
    hdu = pyfits.open(f)
    spec = hdu[0].data.copy()
    hdr = hdu[0].header.copy()
    hdu.close()
    waves = fitshdr_to_wave(hdr)
    
    telwave, telspec = np.genfromtxt(stdfile).transpose()
    # Cross-correlate the standard star and the sci spectra
    # to find wavelength shift of standard star.
    p = fitxcor(waves, spec, telwave, telspec)
    if abs(p[0] - 1.0) > 0.02 or abs(p[1]) > 10.0:
        print "Cross-correlation scale/shift too large; won't do it:"
        print p
        print "   reset to [1.0, 0.0]"
        p = [1.0, 0.0]
    # shift and stretch standard star spectrum to match science
    # spectrum.
    telcorr = interp(waves, p[0] * telwave + p[1], telspec)
    # In principle, we should scale by the proper airmasses, but SALT
    # always observes at ~same airmass
    
    # Divide science spectrum by transformed standard star sub-spectrum
    correct_spec = spec / telcorr
    # Copy telluric-corrected data to new file.
    tofits(outfile, correct_spec, hdr=hdr, clobber=True)
    print "Telluric correction applied; output is " + outfile
    iraf.cd('..')

def ncor(x, y):
    """Calculate the normalized correlation of two arrays"""
    d = np.correlate(x, x) * np.correlate(y, y)
    if d <= 0:
        return 0
    return np.correlate(x, y) / d ** 0.5

def xcorfun(p, warr, farr, telwarr, telfarr):
    # Telluric wavelengths and flux
    # Observed wavelengths and flux
    # resample the telluric spectrum at the same wavelengths as the observed
    # spectrum
    #Make the artifical spectrum to cross correlate
    asfarr = interp( warr, p[0]*telwarr  + p[1], telfarr, left=1.0,right=1.0)
    return abs(1.0 / ncor(farr, asfarr))


def fitxcor(warr, farr, telwarr, telfarr):
    """Maximize the normalized cross correlation coefficient for the telluric
    correction
    """
    res = optimize.minimize(xcorfun, [1.0,0.0], method='Nelder-Mead',
                   args=(warr, farr, telwarr, telfarr))

    return res['x']


if __name__ == '__main__':
    # Parse the input arguments.
    import argparse
    parser = argparse.ArgumentParser(description='Reduce long-slit RSS SALT data. Available stages are: %s.' % allstages)
    parser.add_argument('--files', default=None, metavar='files', help='Files to work on.')
    parser.add_argument('--stages', default='all', metavar='stages', 
        help='Stages to run. Can be "all", a comma separated list, or a range delineated by a "-". Default is "all".')
    parser.add_argument('--stdfolder', metavar='stdfolder', default='./',
               help='Path to the standard star file folder to use for flux calibration and telluric correction.')
    parser.add_argument('--flatfolder', metavar='flatfolder', default=None,
               help='[Obsolete] Path to the file folder with previous flat fields to use if we did not obtain new flats.')
    parser.add_argument("--bright", action="store_true", help="Less aggressive cosmic ray removal for bright objects.")
    parser.add_argument("--fastred", action="store_true", help='Data is from the "fast" quicklook, nightime pipeline.')
    args = parser.parse_args()
    load_modules()
    run(files=args.files, dostages=args.stages, stdsfolder=args.stdfolder, flatfolder=args.flatfolder, 
        brightstar=args.bright, fastred=args.fastred)
    sys.exit("Thanks for using this pipeline!")
