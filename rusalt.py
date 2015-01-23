#!/usr/bin/env python

from pyraf import iraf
iraf.pysalt()
iraf.saltspec()
iraf.saltred()

import sys
import os
import shutil
from glob import glob
import pyfits
import numpy as np
import lacosmicx
from scipy import interp
from astropy.wcs import WCS
from scipy import optimize
import ds9


iraf.onedspec()
iraf.twodspec()
iraf.longslit()
iraf.apextract()
iraf.imutil()

# System specific path to pysalt
pysaltpath = '/usr/local/astro64/iraf/extern/pysalt'

# Define the stages
allstages = ['pysalt', 'makeflats', 'flatten', 'mosaic', 'combine2d',
             'identify2d', 'rectify', 'background', 'lax', 'extract',
             'stdsensfunc', 'fluxscale', 'speccombine', 'mktelluric',
             'telluric']


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


def run(dostages='all', stdstar=False, interactive=False, files=None):
    # Make sure the stages parameters makes sense
    try:
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
        locals()[stage](files)


def pysalt(fs=None):
    # Run the pysalt pipeline on the raw data.
    if fs is None:
        fs = glob('P*.fits')
    if len(fs) == 0:
        print "WARNING: No raw files to run PySALT pre-processing."
        return

    # Copy the raw files into a raw directory
    if not os.path.exists('raw'):
        os.mkdir('raw')
    if not os.path.exists('work'):
        os.mkdir('work')
    for f in fs:
        shutil.copy(f, 'raw/')
        shutil.move(f, 'work/')
    os.chdir('work')

    # Run each of the pysalt pipeline steps deleting temporary files as we go
    # saltprepare
    iraf.unlearn(iraf.saltprepare)
    # Currently, there is not a bad pixel mask provided by SALT
    # so we don't create one here.
    iraf.saltprepare(images='P*.fits', clobber=True, mode='h')

    for f in glob('P*.fits'):
        os.remove(f)
    # saltgain
    iraf.unlearn(iraf.saltgain)
    # Multiply by the gain so that everything is in electrons.
    iraf.saltgain(images='pP*.fits',
                  gaindb=pysaltpath + '/data/rss/RSSamps.dat',
                  mult=True, usedb=True, mode='h')

    for f in glob('pP*.fits'):
        os.remove(f)

    # write a keyword in the header keyword gain = 1 in each amplifier
    fs = glob('gpP*.fits')
    for f in fs:
        for i in range(1, 7):
            pyfits.setval(f, 'GAIN', ext=i, value=1.0)

    # saltxtalk
    iraf.unlearn(iraf.saltxtalk)
    iraf.saltxtalk(images='gpP*.fits', clobber=True, usedb=True,
                   xtalkfile=pysaltpath + '/data/rss/RSSxtalk.dat', mode='h')
    for f in glob('gpP*.fits'):
        os.remove(f)

    # saltbias
    iraf.unlearn(iraf.saltbias)
    iraf.saltbias(images='xgpP*.fits', clobber=True, mode='h')
    for f in glob('xgpP*.fits'):
        os.remove(f)

    # Put all of the newly created files into the pysalt directory
    if not os.path.exists('pysalt'):
        os.mkdir('pysalt')
    for f in glob('bxgpP*.fits'):
        shutil.move(f, 'pysalt')
    os.chdir('..')

    # Hold off on the the mosaic step for now. We want to do some processing on
    # the individual chips


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


def makeflats(fs=None):
    # Note the list of files need to not include any paths relative to
    # the work directory.
    # Maybe not the greatest convention, but we can update this later
    os.chdir('work')
    if fs is None:
        fs = glob('pysalt/bxgp*.fits')
    if len(fs) == 0:
        print "WARNING: No flat-fields to combine and normalize."
        # Fail gracefully by going up a directory
        os.chdir('..')
        return
    # make a flats directory
    if not os.path.exists('flats'):
        os.mkdir('flats')

    # Figure out which images are flats and which grating angles were used
    allflats, grangles = get_ims(fs, 'flat')

    # For each grating angle
    for ga in np.unique(grangles):
        # grab the flats for this gr angle
        flats = allflats[grangles == ga]

        # For each chip
        for c in range(1, 7):
            # run imcombine with average and sigclip, weighted by exposure time
            flatlist = ''
            for f in flats:
                flatlist += '%s[%i],' % (f, c)
                # Add the exptime keyword to each extension
                pyfits.setval(f, 'EXPTIME', ext=c,
                              value=pyfits.getval(f, 'EXPTIME'))

            # set the output combined file name
            combineoutname = 'flats/flt%0.2fcomc%i.fits' % (ga, c)
            if os.path.exists(combineoutname):
                os.remove(combineoutname)
            # initialize the iraf command
            iraf.unlearn(iraf.imcombine)

            # don't forget to remove the last comma in the filelist
            iraf.imcombine(input=flatlist[:-1], output=combineoutname,
                           combine='average', reject='sigclip', lsigma=3.0,
                           hsigma=3.0, weight='exposure', expname='EXPTIME')

            pyfits.setval(combineoutname, 'DISPAXIS', value=1)
            # We want to make an illumination correction file
            # before running response:
            illumoutname = 'flats/flt%0.2fillc%i.fits' % (ga, c)
            iraf.unlearn(iraf.illumination)
            iraf.illumination(images=combineoutname,
                              illuminations=illumoutname, interactive=False,
                              naverage=-40, order=11, low_reject=3.0,
                              high_reject=3.0, niterate=5, mode='hl')

            # Flag any pixels in the illumination correction< 0.1
            illumhdu = pyfits.open(illumoutname, mode='update')
            illumhdu[0].data[illumhdu[0].data <= 0.1] = 0.0
            illumhdu.flush()

            # Get 40 pixels out of the middle of the image and
            # median them to run response
            combinehdu = pyfits.open(combineoutname)
            ny = combinehdu[0].data.shape[0]
            # divide out the illumination correction before running response
            flat1d = np.median(combinehdu[0].data[ny / 2 - 21: ny / 2 + 20, :]
                               / illumhdu[0].data[ny / 2 - 21: ny / 2 + 20, :],
                               axis=0)
            # close the illumination file because we don't need it anymore
            illumhdu.close()

            # File stage m1d for median 1-D
            flat1dfname = 'flats/flt%0.2fm1dc%i.fits' % (ga, c)
            tofits(flat1dfname, flat1d, hdr=combinehdu[0].header.copy())

            # run response
            # r1d = response1d
            resp1dfname = 'flats/flt%0.2fr1dc%i.fits' % (ga, c)
            iraf.response(flat1dfname, flat1dfname, resp1dfname, order=31,
                          interactive=False, naverage=-5, low_reject=3.0,
                          high_reject=3.0, niterate=5, mode='hl')

            resp1dhdu = pyfits.open(resp1dfname)
            resp1d = resp1dhdu[0].data.copy()
            resp1dhdu.close()

            # After response divide out the response function
            # normalize the 1d resp to its median
            resp1d /= np.median(resp1d)

            # Chuck any outliers
            flatsig = np.std(resp1d - 1.0)
            resp1d[abs(resp1d - 1.0) > 5.0 * flatsig] = 1.0
            resp = flat1d / resp1d

            resp2dfname = 'flats/flt%0.2fresc%i.fits' % (ga, c)
            resp2d = combinehdu[0].data.copy() / resp
            tofits(resp2dfname, resp2d, hdr=combinehdu[0].header.copy())
            combinehdu.close()

            # close the combined flat because we don't need it anymore
            combinehdu.close()

            pyfits.setval(resp2dfname, 'DISPAXIS', value=1)

            # Reset any pixels in the flat field correction< 0.1
            # We could flag bad pixels here if we want, but not right now
            flathdu = pyfits.open(resp2dfname, mode='update')
            flathdu[0].data[flathdu[0].data <= 0.1] = 0.0
            flathdu.flush()
            flathdu.close()
    # Step back up to the top directory
    os.chdir('..')


def flatten(fs=None):
    os.chdir('work')
    if fs is None:
        fs = glob('pysalt/bxgpP*.fits')
    if len(fs) == 0:
        print "WARNING: No images to flat-field."
        # Change directories to fail more gracefully
        os.chdir('..')
        return
    if not os.path.exists('flts'):
        os.mkdir('flts')
    # Make sure there are science images or arcs and what grating angles were
    # used
    scifs, scigas = get_ims(fs, 'sci')
    arcfs, arcgas = get_ims(fs, 'arc')

    ims = np.append(scifs, arcfs)
    gas = np.append(scigas, arcgas)
    # For each science and arc image
    for i, f in enumerate(ims):
        thishdu = pyfits.open(f)
        ga = gas[i]
        # For each chip
        for c in range(1, 7):
            # open the corresponding response file
            resphdu = pyfits.open('flats/flt%0.2fresc%i.fits' % (ga, c))
            # divide out the illumination correction and the flatfield
            # make sure divzero = 0.0
            thishdu[c].data /= resphdu[0].data.copy()
            # replace the infinities with 0.0
            thishdu[c].data[np.isinf(thishdu[c].data)] = 0.0
            resphdu.close()

        # save the updated file
        if f in scifs:
            typestr = 'sci'
        else:
            typestr = 'arc'
        # get the image number
        # by salt naming convention, these should be the last 3 characters
        # before the '.fits'
        imnum = f[-8:-5]
        outname = 'flts/' + typestr + '%0.2fflt%03i.fits' % (float(ga),
                                                             int(imnum))
        thishdu.writeto(outname)
        thishdu.close()

    os.chdir('..')


def mosaic(fs=None):

    os.chdir('work')
    # If the file list is not given, grab the default files
    if fs is None:
        fs = glob('flts/*.fits')
    # Abort if there are no files
    if len(fs) == 0:
        print "WARNING: No flat-fielded images to mosaic."
        os.chdir('..')
        return

    if not os.path.exists('mos'):
        os.mkdir('mos')

    # Get the images to work with
    ims, gas = get_scis_and_arcs(fs)

    for i, f in enumerate(ims):
        ga = gas[i]
        fname = f.split('/')[1]
        typestr = fname[:3]
        # by our naming convention, imnum should be the last 3 characters
        # before the '.fits'
        imnum = fname[-8:-5]
        outname = 'mos/' + typestr
        outname += '%0.2fmos%03i.fits' % (float(ga), int(imnum))
        # prepare to run saltmosaic
        iraf.unlearn(iraf.saltmosaic)
        iraf.flpr()
        iraf.saltmosaic(images=f, outimages=outname, outpref='',
                        geomfile=pysaltpath + '/data/rss/RSSgeom.dat',
                        clobber=True, mode='h')

        # Make a bad pixel mask marking where there is no data.
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
        h.close()

    os.chdir('..')


def identify2d(fs=None):
    os.chdir('work')
    if fs is None:
        fs = glob('nrm/arc*nrm*.fits')
    if len(fs) == 0:
        print "WARNING: No slit normalized arcs for (2D) specidentify."
        # Change directories to fail gracefully
        os.chdir('..')
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
        ccdsum = int(pyfits.getval(f, 'CCDSUM').split()[1])

        # linelistpath is a global variable defined in beginning, path to
        # where the line lists are.
        lamplines = pysaltpath + '/data/linelists/' + lampfiles[lamp]

        # img num should be right before the .fits
        imgnum = f[-8:-5]
        # run pysalt specidentify
        idfile = 'id2/arc%0.2fid2%03i' % (float(ga), int(imgnum)) + '.db'
        iraf.unlearn(iraf.specidentify)
        iraf.flpr()
        iraf.specidentify(images=f, linelist=lamplines, outfile=idfile,
                          guesstype='rss', inter=True, rstep=600 / ccdsum,
                          rstart=200 / ccdsum, startext=1, clobber='yes',
                          verbose='yes', mode='hl', logfile='salt.log',
                          mdiff=2, function='legendre')
    os.chdir('..')


def get_chipgaps(hdu):
        # Get the x coordinages of all of the chip gap pixels
        # recall that pyfits opens images with coordinates y, x
        # get the BPM from 51-950 which are the nominally good pixels
        # (for binning = 4 in the y direction)
        # (the default wavelength solutions are from 50.5 - 950.5)
        # Note this throws away one extra pixel on either side but it seems to
        # be necessary.
        ccdsum = int(hdu[0].header['CCDSUM'].split()[1])

        ypix = slice(200 / ccdsum + 1, 3800 / ccdsum)
        d = hdu[1].data[ypix].copy()
        bpm = hdu[2].data[ypix].copy()

        w = np.where(np.logical_or(bpm > 0, d == 0))[1]

        # Note we also grow the chip gap by 1 pixel on each side
        # Chip 1
        chipgap1 = (np.min(w[w > 10]) - 1, np.max(w[w < 1500]) + 1)
        # Chip 2
        chipgap2 = (np.min(w[w > 1500]) - 1, np.max(w[w < 2500]) + 1)
        # edge of chip 3=
        chipgap3 = (np.min(w[w > 2500]) - 1, hdu[2].data.shape[1] + 1)
        return (chipgap1, chipgap2, chipgap3)


def rectify(ids=None, fs=None):
    os.chdir('work')
    if ids is None:
        ids = np.array(glob('id2/arc*id2*.db'))
    if fs is None:
        fs = glob('mos/*mos*.fits')
    if len(ids) == 0:
        print "WARNING: No wavelength solutions for rectification."
        os.chdir('..')
        return
    if len(fs) == 0:
        print "WARNING: No images for rectification."
        os.chdir('..')
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
    if not os.path.exists('rec'):
        os.mkdir('rec')
    for i, f in enumerate(ims):
        fname = f.split('/')[1]
        typestr = fname[:3]
        ga, imgnum = gas[i], fname[-8:-5]

        outfile = 'rec/' + typestr + '%0.2frec' % (ga) + imgnum + '.fits'
        iraf.unlearn(iraf.specrectify)
        iraf.flpr()
        idfile = ids[np.array(idgas) == ga][0]
        iraf.specrectify(images=f, outimages=outfile, solfile=idfile,
                         outpref='', function='legendre', order=3,
                         inttype='interp', conserve='yes', clobber='yes',
                         verbose='yes')

        # Update the BPM to mask any blank regions
        h = pyfits.open(outfile, 'update')
        # Cover the chip gaps. The background task etc do better if the chip
        # gaps are straight
        # To deal with this we just throw away the min and max of each side of
        # the curved chip gap
        chipgaps = get_chipgaps(h)

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
    os.chdir('..')


def slitnormalize(fs=None):
    os.chdir('work')
    if fs is None:
        fs = glob('rec/*rec*.fits')
    if len(fs) == 0:
        print "WARNING: No mosaiced files for slitnormalize."
        # Change directories to fail gracefully
        os.chdir('..')
        return
    if not os.path.exists('nrm'):
        os.mkdir('nrm')

    for f in fs:
        outname = f.replace('rec', 'nrm')
        iraf.unlearn(iraf.specslitnormalize)
        iraf.specslitnormalize(images=f, outimages=outname, outpref='',
                               order=5, clobber=True, mode='h')

    os.chdir('..')


def background(fs=None):
    os.chdir('work')
    # Get rectified science images
    if fs is None:
        fs = glob('nrm/sci*nrm*.fits')
    if len(fs) == 0:
        print "WARNING: No rectified images for background-subtraction."
        os.chdir('..')
        return

    if not os.path.exists('bkg'):
        os.mkdir('bkg')

    for f in fs:
        print("Subtracting background for %s" % f)
        # Make sure dispaxis is set correctly
        pyfits.setval(f, 'DISPAXIS', value=1)

        # the outfile name is very similar, just change folder prefix and
        # 3-char stage substring
        outfile = 'bkg/' + f[4:12] + 'bkg' + f[15:]
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
    os.chdir('..')


def lax(fs=None):
    os.chdir('work')
    if fs is None:
        fs = glob('bkg/*bkg*.fits')
    if len(fs) == 0:
        print "WARNING: No background-subtracted files for Lacosmicx."
        os.chdir('..')
        return

    if not os.path.exists('lax'):
        os.mkdir('lax')
    for f in fs:
        outname = 'lax/' + f[4:12] + 'lax' + f[15:]
        hdu = pyfits.open(f)

        # Add a CRM extension
        hdu.append(pyfits.ImageHDU(data=hdu['BPM'].data.copy(),
                                   header=hdu['BPM'].header.copy(),
                                   name='CRM'))
        # Set all of the pixels in the CRM mask to zero
        hdu['CRM'].data[:, :] = 0

        chipgaps = get_chipgaps(hdu)
        chipedges = [[0, chipgaps[0][0]], [chipgaps[0][1] + 1, chipgaps[1][0]],
                     [chipgaps[1][1] + 1, chipgaps[2][0]]]

        # Run each chip separately
        for chip in range(3):
            # Use previously subtracted sky level = 0 as we have already added
            # a constant sky value in the background task
            # Gain = 1, readnoise should be small so it shouldn't matter much.
            # Default value seems to work.
            chipinds = slice(chipedges[chip][0], chipedges[chip][1])
            lacosmicx.run(inmat=hdu[1].data[:, chipinds],
                          inmask=hdu[2].data[:, chipinds],
                          outmaskfile='tmpmsk.fits', sigclip=4.0, objlim=1.0,
                          sigfrac=0.1, gain=1.0, pssl=0.0, robust=False)

            # Open the cr mask file
            maskhdu = pyfits.open('tmpmsk.fits')

            # Update the image
            hdu['CRM'].data[:, chipinds][:, :] = maskhdu[0].data[:, :]
            # Flag the cosmic ray pixels with a large negative number
            hdu['SCI'].data[:, chipinds][maskhdu[0].data == 1] = -1000000

            # clean up
            os.remove('tmpmsk.fits')

        # Save the file
        hdu.writeto(outname, clobber=True)
        hdu.close()

    os.chdir('..')


def fixpix(fs=None):
    os.chdir('work')
    if fs is None:
        fs = glob('nrm/sci*nrm*.fits')
    if len(fs) == 0:
        print "WARNING: No rectified images to fix."
        os.chdir('..')
        return
    if not os.path.exists('fix'):
        os.mkdir('fix')
    for f in fs:
        outname = f.replace('nrm', 'fix')
        # Copy the file to the fix directory
        shutil.copy(f, outname)
        # Set all of the BPM pixels = 0
        h = pyfits.open(outname, mode='update')
        h['SCI'].data[h['BPM'] == 1] = 0
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
    os.chdir('..')


def extract(fs=None, interactive=True):
    os.chdir('work')
    if fs is None:
        fs = glob('fix/*fix*.fits')
    if len(fs) == 0:
        print "WARNING: No fixpixed images available for extraction."
        os.chdir('..')
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
        if interactive:
            ds9display(f.replace('fix', 'bkg'))
        # set dispaxis = 1 just in case
        pyfits.setval(f, 'DISPAXIS', extname='SCI', value=1)
        iraf.unlearn(iraf.apall)
        iraf.flpr()
        iraf.apall(input=f + '[SCI]', output=outbase, interactive='yes',
                   review='no', line='INDEF', nsum=-1000, lower=-5, upper=5,
                   b_function='legendre', b_order=5,
                   b_sample='-400:-200,200:400', b_naverage=-10, b_niterate=5,
                   b_low_reject=3.0, b_high_reject=3.0, nfind=1, t_nsum=15,
                   t_step=15, t_nlost=100, t_function='legendre', t_order=5,
                   t_niterate=5, t_low_reject=3.0, t_high_reject=3.0,
                   background='fit', weights='variance', pfit='fit1d',
                   clean='no', readnoise=readnoise, gain=1.0, lsigma=4.0,
                   usigma=4.0, mode='hl')

        # Copy the CCDSUM keyword into the 1d extraction
        pyfits.setval(outbase + '.fits', 'CCDSUM',
                      value=pyfits.getval(f, 'CCDSUM'))

        # Extract the corresponding arc
        arcname = glob('nrm/arc' + f.split('/')[1][3:8] + '*.fits')[0]
        # set dispaxis = 1 just in case
        pyfits.setval(arcname, 'DISPAXIS', extname='SCI', value=1)
        iraf.unlearn(iraf.apsum)
        iraf.flpr()
        iraf.apsum(input=arcname + '[SCI]', output='auxext_arc',
                   references=f[:-5] + '[SCI]', interactive='no', find='no',
                   edit='no', trace='no', fittrace='no', extras='no',
                   review='no', background='no', mode='hl')
        # copy the arc into the 5 column of the data cube
        arcfs = glob('auxext_arc*.fits')
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
        # Add the airmasa and exptime keywords back into the
        # extracted spectrum header
        pyfits.setval(outbase + '.fits', 'AIRMASS',
                      value=pyfits.getval(f, r'AIRMASS'))
        pyfits.setval(outbase + '.fits', 'EXPTIME',
                      value=pyfits.getval(f, 'EXPTIME'))
    os.chdir('..')


def split1d(fs=None):
    os.chdir('work')
    if fs is None:
        fs = glob('x1d/sci*x1d*.fits')
    if len(fs) == 0:
        print "WARNING: No extracted spectra to split."
        os.chdir('..')
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
                       format='multispec', rebin='no')
        hdu.close()
    os.chdir('..')


def isstdstar(f):
    # get the list of standard stars
    stdslist = glob(pysaltpath + '/data/standards/spectroscopic/*')
    objname = pyfits.getval(f, 'OBJECT').lower()
    for std in stdslist:
        if objname in std:
            return True

    # Otherwise not in the list so return false
    return False


def spectoascii(fname, asciiname, ap=0):
    hdu = pyfits.open(fname)
    w = WCS(fname)
    # get the wavelengths of the pixels
    npix = hdu[0].data.shape[2]
    lam = w.all_pix2world(np.linspace(0, npix - 1, npix), 0, 0, 0)[0]
    spec = hdu[0].data[0, ap]
    specerr = hdu[0].data[2, ap]
    np.savetxt(asciiname, np.array([lam, spec, specerr]).transpose())
    hdu.close()


def stdsensfunc(fs=None):
    os.chdir('work')
    if fs is None:
        fs = glob('x1d/sci*x1d*c?.fits')
    if len(fs) == 0:
        print "WARNING: No extracted spectra to create sensfuncs from."
        os.chdir('..')
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
            stdfile = pysaltpath + '/data/standards/spectroscopic/m%s.dat' % pyfits.getval(f, 'OBJECT').lower()
            extfile = pysaltpath + '/data/site/suth_extinct.dat'
            iraf.unlearn(iraf.specsens)
            iraf.specsens(asciispec, outfile, stdfile, extfile,
                          airmass=pyfits.getval(f, 'AIRMASS'),
                          exptime=pyfits.getval(f, 'EXPTIME'), function='poly',
                          order=11, clobber=True, mode='h', thresh=1e10)
            # delete the ascii file
            os.remove(asciispec)
    os.chdir('..')


def fluxcal(stdsfolder='./std', fs=None):
    os.chdir('work')
    if fs is None:
        fs = glob('x1d/sci*x1d*c*.fits')
    if len(fs) == 0:
        print "WARNING: No science chip spectra to flux calibrate."
        os.chdir('..')
        return

    if not os.path.exists('flx'):
        os.mkdir('flx')
    extfile = pysaltpath + '/data/site/suth_extinct.dat'
    stdfiles = glob(stdsfolder + '/*sens*c?.dat')
    for f in fs:
        outfile = f.replace('x1d', 'flx')
        chip = outfile[-6]
        hdu = pyfits.open(f)
        ga = f.split('/')[1][3:8]
        # Get the standard sensfunc with the same grating angle
        stdfile = None
        for stdf in stdfiles:
            if ga in stdf:
                # Get the right chip number
                if chip == stdf[-5]:
                    stdfile = stdf
                    break
        if stdfile is None:
            print('No standard star with grating-angle %s' % ga)
            continue
        # for each extracted aperture
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
    os.chdir('..')


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


def speccombine():
    os.chdir('work')
    nsteps = 8001
    lamgrid = np.linspace(2000.0, 10000.0, nsteps)
    fs = glob('flx/sci*c?.fits')
    nfs = len(fs)
    # for each aperture
    # get all of the science images
    specs = np.zeros((nfs, nsteps))
    specerrs = np.zeros((nfs, nsteps))
    ap = 0
    for i, f in enumerate(fs):
        hdu = pyfits.open(f)
        w = WCS(f)
        # get the wavelengths of the pixels
        npix = hdu[0].data.shape[2]
        lam = w.all_pix2world(np.linspace(0, npix - 1, npix), 0, 0, 0)[0]
        # interpolate each spectrum onto a comman wavelength scale

        specs[i] = interp(lamgrid, lam, hdu[0].data[0][ap],
                          left=0.0, right=0.0)
        # Also calculate the errors. Right now we assume that the variances
        # interpolate linearly. This is not stricly correct but it should be
        # close. Also we don't include terms in the variance for the
        # uncertainty in the wavelength solution.
        specerrs[i] = interp(lamgrid, lam, hdu[0].data[2][ap] ** 2.0) ** 0.5

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
        lines.append('%f\n' % p)
    f = open('flx/scales.dat', 'w')
    f.writelines(lines)
    f.close()
    # run scombine after multiplying the spectra by the best fit parameters
    if os.path.exists('out.fits'):
        os.remove('out.fits')
    iraf.scombine(iraf_filelist, 'sci.fits', scale='@flx/scales.dat',
                  reject='avsigclip', lthreshold=1e-19)
    # Remove the other apertures
    # remove the sky and arc bands from the combined spectra.
    return specs


# Define the telluric B and A band begin:end wavelengths
telluricWaves = {'B': (6855, 6935), 'A': (7590, 7685)}
#    w =  where( (wave GE 3190) AND (wave LE 3216),  nw)
#     if (nw GT 0) then bstar(w) =  1
# ;    w =  where( (wave GE 3420) AND (wave LE 5600),  nw)
#     w =  where( (wave GE 3420) AND (wave LE 5500),  nw)
#     if (nw GT 0) then bstar(w) =  1
#     w =  where( (wave GE 6050) AND (wave LE 6250),  nw)
#     if (nw GT 0) then bstar(w) =  1
#     w =  where( (wave GE 6360) AND (wave LE 6450),  nw)
#     if (nw GT 0) then bstar(w) =  1
#     w =  where( (wave GE 6530) AND (wave LE 6840),  nw)
#     if (nw GT 0) then bstar(w) =  1
#     w =  where( (wave GE 7410) AND (wave LE 7560),  nw)
#     if (nw GT 0) then bstar(w) =  1
#     w =  where( (wave GE 8410) AND (wave LE 8800),  nw)
#     if (nw GT 0) then bstar(w) =  1
#     w =  where( (wave GE 9900),  nw)
#     if (nw GT 0) then bstar(w) =  1


def mktelluric(fs=None):
    # for each file
    # if it is a standard star combined file
    # fit a smooth function over the telluric regions
    # Replace the telluric with the smoothed function
    # Divide the original and the telluric corrected spectra to
    # get the correction factor
    return


def telluric(stdsfolder='./tel/', fs=None):
    os.chdir('work')
    if fs is None:
        fs = glob('sci.fits')
    if len(fs) == 0:
        print "WARNING: No flux-calibrated spectra to telluric-correct."
        os.chdir('..')
        return

    if not os.path.exists('tel'):
        os.mkdir('tel')
    for f in fs:
        outfile = f.replace('final.fits')
        # Get the standard to use for telluric correction
        stdfile = glob(stdsfolder + '/*.fits')[0]

        # Loop over the telluric regions? Maybe don't even need a loop
        # Cross-correlate the standard star and the sci spectra
        # to find wavelength shift of standard star.
        # Scale, shift, and stretch standard star spectrum to match science
        # spectrum.
        # Divide science spectrum by transformed standard star sub-spectrum
        # Copy telluric-corrected data to new file.


if __name__ == '__main__':
    # Parse the input arguments.
    import argparse
    parser = argparse.ArgumentParser(description='Reduce long-slit RSS SALT data.')
    parser.add_argument('stdfolder', metavar='stdfolder', default='std',
               help='Path to the standard star file folder to use for flux calibration and telluric correction.')
    args = parser.parse_args()
    run(args.stdfolder)
    sys.exit("Thanks for using this pipeline!")
