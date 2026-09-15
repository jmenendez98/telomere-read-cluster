"""TeloBP, vendored: the array/subtelomere boundary caller module 1 runs on.

This is TeloBP's code, lightly adapted, carried here so that `trc` has no
import to satisfy outside its own package.  It was a real operational problem:
TeloBP is not on PyPI in the form this pipeline needs, the fork lives in a
checkout whose path differs per machine, and `environment.yml` could not
express that -- so every environment built from the file came out silently
missing its boundary caller and died at module 1.

    TeloBP -- https://github.com/GEN-DBIO/TeloBP
    MIT License, Copyright (c) 2024 Ramin Kahidi

Taken from the `itsfix` fork, whose addition is the second half of this file:
`getTeloNPArrayBoundary` bounds the scan by the array's own extent and snaps
the call onto the last run of the caller's own telomere pattern.  The
unpatched TeloBP cannot bound the
scan, which is what lets a distant interstitial telomeric block capture the
boundary -- 20q at 108 kb, Xp at 216 kb, on real ONT reads.

WHAT WAS ADAPTED
    * Only the code reached from `getTeloNPArrayBoundary` and
      `getIsGStrandFromSeq` is here.  Dropped: `trimTeloReferenceGenome` and
      `refRecordTeloLengths`, which want biopython; `makeOffsetPlot`; the
      CHM13 manual-label tables and the two functions that test against them;
      `descriptionToChr`, `write_bed_file`, `recordBedData`, and
      `isGStrand(chrArm, strand)`, which are reference-genome utilities; and
      an `import pandas as pd` that upstream never uses.
    * `graphLine` imports pyplot in its body rather than at module scope.
      Upstream imports it on the way in, which is the only reason matplotlib
      was ever a dependency of this pipeline -- nothing in `trc` plots.
    * The three upstream modules are one file, so the cross-imports between
      them are gone.  Nothing else was touched: the algorithm, the parameter
      defaults, the numbered error returns and the upstream naming convention
      -- camelCase, which is not this repo's -- are as they came.

Re-syncing is a diff against the checkout, by construction: every block below
was cut from it by line range rather than retyped.

`errorReturns` is the vocabulary the caller needs.  `getTeloBoundary` reports
failure in band, as a negative int, not by raising:

    init         -1     read shorter than teloWindow, or validation failed
    fusedRead   -10     telomeric repeat at BOTH ends of the read
    strandType  -20     strand could not be called from composition at all
    seqNotFound -1000   unused on this path

`trc.intake` turns the middle two into its `both_ends` and `no_array`
rejections and treats every other negative as `no_boundary`.
"""
import logging
import re

import numpy as np

# Verbatim from upstream, and load-bearing: `getTeloBoundary` reports ordinary
# per-read outcomes -- a read too short to scan, a fused read -- through
# `logging.warning`.  Without this line the root logger's last-resort handler
# prints every one of them to stderr, which on a rejection-heavy sample is
# tens of thousands of lines.  `trc` does not use `logging` itself, so nothing
# of ours is being reconfigured here.
logging.basicConfig(level=logging.ERROR)


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

expectedTeloCompositionQ = [
    ["GGG", 3/6],
]

expectedTeloCompositionP = [
    ["CCC", expectedTeloCompositionQ[0][1]],
]

# Recommended composition for nanopore reads called with Guppy
teloNPTeloCompositionGStrand = [["(?!GGG)[ATC]{2}.GGG|(?!AAA)[AT]{3}AAA|TTAGG.", 6/6, 6]]

teloNPTeloCompositionCStrand = [["CTTCTT|CCTGG|CCC(?!CCC)[ATCG]{3}", 6/6, 6]]

# When the area under the curve of the offsets is calculated, this
# constant is used to mark the point where the slope of the curve
# starts to plateau. Basically, once the difference between two
# values in the area under the curve is less than this, the slope
# is considered to be plateauing.
areaDiffsThreshold = 0.2


errorReturns = {"init": -1, "fusedRead": -10,"strandType": -20, "seqNotFound": -1000}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def is_regex_pattern(input_string):
    return bool(re.search(r'[^a-zA-Z]', input_string))


def getGraphArea(offsets, targetColumn, windowSize):
    data = np.array(offsets)
    transposed_data = data.T
    areaList = []
    row = transposed_data[targetColumn, :]
    
    # print(row)
    for i in range(0, len(row) - windowSize, 1):
        area = row[i:i + windowSize].sum()
        areaList.append((area / windowSize))
        # areaList.append(area)
    return areaList


def graphLine(rowIn, labelIn, windowStep, boundaryPoint=-1, pdfOut=None):
    # ADAPTED: pyplot imported here rather than at module scope, so trc
    # does not carry matplotlib for a path it never takes.
    import matplotlib.pyplot as plt

    row = rowIn
    # if boundaryPoint is less than half the x axis, only show boundaryPoint x2 of the data
    if boundaryPoint < 0.5 * len(rowIn):
        row = rowIn[:int((2 * boundaryPoint)/windowStep)]

    x = np.arange(len(row))
    x = x * windowStep
    fig, ax = plt.subplots()

    ax.plot(x, row, label=labelIn)

    # Set labels and title
    ax.set_xlabel('Distance from end of sequence')
    ax.set_ylabel('Nucleotide offset')
    ax.set_title('Nucleotide Offsets from Expected Telomere Composition')

    if boundaryPoint != -1:
        ax.axvline(boundaryPoint, color='red', label='Boundary Point')

    # Show legend
    ax.legend()

    if pdfOut is not None:
        pdfOut.savefig(fig)
    elif pdfOut is None:
        # Display the graph
        plt.show()


def validate_parameters(seq, isGStrand, composition, teloWindow=100, windowStep=6, plateauDetectionThreshold=-15, changeThreshold=-5, targetPatternIndex=-1, nucleotideGraphAreaWindowSize=500, showGraphs=False):
    
    validate_seq_teloWindow(seq, teloWindow)

    if not isinstance(isGStrand, bool) and not isinstance(isGStrand, np.bool_):
        raise ValueError("isGStrand should be a boolean, or numpy boolean")

    if not isinstance(windowStep, int) or windowStep < 1:
        raise ValueError(
            "windowStep should be an int greater than or equal to 1")

    if not isinstance(targetPatternIndex, int) or targetPatternIndex > len(composition):
        raise ValueError(
            "targetPatternIndex should be an int and within the range of the composition list")

    if not isinstance(nucleotideGraphAreaWindowSize, int) or nucleotideGraphAreaWindowSize < 1:
        raise ValueError(
            "nucleotideGraphAreaWindowSize should be an int greater than or equal to 1")

    if not isinstance(showGraphs, bool):
        raise ValueError("showGraphs should be a boolean")

def validate_seq_teloWindow(seq, teloWindow):
    if not isinstance(teloWindow, int) or teloWindow < 6:
        raise ValueError(
            "teloWindow should be an int greater than or equal to 6")

    if len(seq) < teloWindow:
        raise Warning("sequence length is less than teloWindow")


# --------------------------------------------------------------------------
# the boundary caller
# --------------------------------------------------------------------------

# The following function takes in a sequence, and returns the index of the telomere boundary.
def getTeloBoundary(seq, isGStrand = None, compositionGStrand=[], compositionCStrand = [], teloWindow=100, windowStep=6, changeThreshold=-20, plateauDetectionThreshold=-50, targetPatternIndex=-1, nucleotideGraphAreaWindowSize=500, showGraphs=False, pdf=None, returnLastDiscontinuity=False, secondarySearch = False, maxScanBP=None):
    """
    This function takes in a sequence, and returns the index of the telomere boundary.

    :param seq: The sequence to be analyzed
    :param isGStrand: True if the sequence is the G strand (has TTAGGG telomeres), False if it is the C strand (has CCCTAA telomeres). 
           Is None by default, and will be determined using the composition lists (count of G vs C repeats) if not specified.
    :param compositionGStrand: A list of lists, where each list contains a nucleotide pattern representing the expected telomere pattern on the G Strand.
    :param compositionCStrand: A list of lists, where each list contains a nucleotide pattern representing the expected telomere pattern on the C Strand.
    :param teloWindow: The size of the window to be used when calculating the offset of the nucleotide composition from the expected 
           telomere composition.
    :param windowStep: The step size to be used when moving through the sequence in 'windows' of size teloWindow.
    :param changeThreshold: The sequence difference threshold where we can assume that we are approaching the telomere boundary. 
           This value is only used if returnLastDiscontinuity is true. 
    :param plateauDetectionThreshold: The sequence difference threshold at which the algorithm starts scanning for the point where the 
           slope of discontinuity plateaus. In this context, discontinuity means when the sequence no longer has a pattern similar to a telomere. 
    :param targetPatternIndex: The index of the pattern in the composition list that we want to use to calculate the telomere boundary. 
           This is primarily for testing purposes, when we want to visualize the offsets of multiple patterns, but only use 1 for finding the boundary.
    :param nucleotideGraphAreaWindowSize: The size of the window to be used when calculating the area under the curve of the offsets. Generally speaking, 
           a smaller value gives greater precision, but less accuracy if the inputted sequence is noisy. A larger value gives greater accuracy, but less precision.
           More specifically, this value must be large enough that it can "look into" the area well past the telomere boundary, making sure that the 
           discontinuity in the telomere pattern is sustained.
    :param showGraphs: Boolean value, if true, the graphs will be shown.
    :param pdf: A pdf object, if provided, the graphs will be saved to the pdf.
    :param returnLastDiscontinuity: Boolean value, if true, the algorithm will return the last possible point of discontinuity, rather than the first.
           For sequences which are very noisy, this may be necessary, as the first point of discontinuity may be a false positive.
    :param maxScanBP: If set, only the telomere-ward `maxScanBP` bases are scanned for the boundary; the rest of the read is
           ignored.  This exists because `returnLastDiscontinuity` searches for the *last* point still above `changeThreshold`
           over the whole read, so any distant region whose composition happens to match the telomere pattern -- an interstitial
           telomeric block, or merely a C-rich subtelomeric tandem repeat -- recaptures the search and drags the boundary out to
           it.  Capping the scan keeps the noise tolerance inside the array without letting it reach sequence that cannot be
           telomere.  Strand determination still sees the whole read, so fused-read detection is unaffected.  None (the default)
           reproduces the unbounded behaviour exactly.
    """

    boundaryPoint = -1
    ntOffsets = []
    graphAreaWindowSize = int(nucleotideGraphAreaWindowSize / windowStep)
    try:
        validate_seq_teloWindow(seq, teloWindow)
    except Warning as w:
        logging.warning(f"Initial validation failed for read, returning {errorReturns['init']}: {w}")
        return errorReturns['init']

    if len(compositionGStrand) == 0:
        compositionGStrand = expectedTeloCompositionQ
    if len(compositionCStrand) == 0:
        compositionCStrand = expectedTeloCompositionP

    # calculate telomere strand type
    if isGStrand == None:
        isGStrand = getIsGStrandFromSeq(seq, compositionGStrand[targetPatternIndex], compositionCStrand[targetPatternIndex])
        if isGStrand < 0 or not isinstance(isGStrand, bool) and not isinstance(isGStrand, np.bool_):
            # print("Could not determine telomere strand type, returning -1")
            if isGStrand == errorReturns['fusedRead']:
                logging.warning(f"Warning: fused strand likely, returning {errorReturns['fusedRead']}")
                return errorReturns['fusedRead']
            logging.warning(f"Warning: could not determine telomere strand type from sequence, returning {errorReturns['strandType']}")
            return errorReturns['strandType']
        
    # Bound the scan.  Done here, after the strand call and before anything
    # reads the sequence again, so every later stage -- the window sweep, the
    # plateau search and the secondary search -- sees one consistent sequence
    # and the returned index keeps its meaning: an offset from the start of the
    # read on the C strand, a distance from its end on the G strand.  Both are
    # preserved by trimming the far end.
    if maxScanBP is not None and len(seq) > maxScanBP:
        seq = seq[-maxScanBP:] if isGStrand else seq[:maxScanBP]

    composition = []
    if isGStrand:
        composition = compositionGStrand
    else:
        composition = compositionCStrand

    '''
    validate_parameters(seq, isGStrand, composition, teloWindow, windowStep, plateauDetectionThreshold,
                        changeThreshold, targetPatternIndex, nucleotideGraphAreaWindowSize, showGraphs)
    '''

    try:
        validate_parameters(seq, isGStrand, composition, teloWindow, windowStep, plateauDetectionThreshold, changeThreshold, targetPatternIndex, nucleotideGraphAreaWindowSize, showGraphs)
    except Warning as w:
        logging.warning(f"Initial validation failed for read, returning {errorReturns['init']}: {w}")
        return errorReturns['init']

    # Move through the sequence in windows of size teloWindow, and step size windowStep,
    # and calculate the offset of the nucleotide composition from the expected telomere composition
    for i in range(0, len(seq) - teloWindow, windowStep):
        teloSeq = ""
        if isGStrand == True:
            teloSeq = seq[(len(seq) - i) - teloWindow:len(seq) - i]
        else:
            teloSeq = seq[i:i + teloWindow]
        teloLen = len(teloSeq)
        teloSeqUpper = str(teloSeq.upper())

        currentOffsets = []
        # Calculate the offset of the nucleotide composition from the expected telomere composition
        for ntPatternEntry in composition:
            ntPattern = ntPatternEntry[0]
            patternComposition = ntPatternEntry[1]
            patternCount = len(re.findall(ntPattern, teloSeqUpper))
            if is_regex_pattern(ntPattern) == True:
                if len(ntPatternEntry) != 3:
                    raise ValueError("Error: a target length must be specified as a third list item if using a regex pattern. Example: ['GGG|AAA', 3/6, 3], where the third item is the target length.")
                regexTargetLength = ntPatternEntry[2]
                rawOffsetValue = (
                    (patternCount * regexTargetLength) / teloLen - patternComposition)
                if patternComposition != 0:
                    percentOffsetValue = (
                        rawOffsetValue / patternComposition) * 100
                    currentOffsets.append(percentOffsetValue)
                else:
                    percentOffsetValue = (rawOffsetValue) * 100
                    currentOffsets.append(percentOffsetValue)
            else:
                # patternCount = the number of times we see the pattern
                # len(ntPattern) = the length of the pattern, so GGG is 3
                # teloLen = the length of the telomere window
                # patternComposition = the expected composition of the pattern, so 3/6 for GGG
                # Here we get an offset score based on the number of nucleotides we expect to see, given the pattern composition
                rawOffsetValue = (
                    (patternCount * len(ntPattern)) / teloLen - patternComposition)
                if patternComposition != 0:
                    # Here we convert the raw offset score to a percentage
                    percentOffsetValue = (
                        rawOffsetValue / patternComposition) * 100
                    currentOffsets.append(percentOffsetValue)
                else:
                    # Incase the expected composition is 0, we just return the raw offset score
                    percentOffsetValue = (rawOffsetValue) * 100
                    currentOffsets.append(percentOffsetValue)

        ntOffsets.append(currentOffsets)

    areaList = getGraphArea(ntOffsets, targetPatternIndex, graphAreaWindowSize)
    areaDiffs = np.diff(areaList)
    indexAtThreshold = -1

    # If returnLastDiscontinuity is true, we will scan for the
    # last point where we are above the changeThreshold, then look ahead for the first point where we
    # go below the plateauDetectionThreshold. This is because the area under the curve is not always monotonically
    # decreasing.

    if returnLastDiscontinuity:
        # Here, we grab the last point where the area is below the changeThreshold, and the slope is negative
        indexAtThreshold = (next((y for y in range(len(areaList) - 2, 0, -1) if (
            areaList[y] > changeThreshold and (0 > (areaList[y + 1] - areaList[y])))), indexAtThreshold))
        if indexAtThreshold != -1:
            # The min threshold was reached, and the slope was negative, so we can look for the max threshold ahead of it
            indexAtThreshold = (next((y for y in range(indexAtThreshold, len(areaList) - 2) if (
                areaList[y] < plateauDetectionThreshold and (0 > (areaList[y + 1] - areaList[y])))), indexAtThreshold))
        else:
            # Didn't find a point above the changeThreshold, so we just scan for the first point past the maxThreshold
            indexAtThreshold = (next((y for y in range(len(areaList) - 2) if (
                areaList[y] < plateauDetectionThreshold and (0 > (areaList[y + 1] - areaList[y])))), indexAtThreshold))
    else:
        indexAtThreshold = (next((y for y in range(len(areaList) - 2) if (
            areaList[y] < plateauDetectionThreshold and (0 > (areaList[y + 1] - areaList[y])))), indexAtThreshold))

    if indexAtThreshold == -1:
        logging.warning(f"No telo boundary found, returning {errorReturns['init']}")
        if showGraphs:
            print(f"showGraph:  {showGraphs}")
            graphLine(
                areaList, composition[targetPatternIndex][0] + " Area", windowStep, pdfOut=pdf)
            # makeOffsetPlot(ntOffsets, composition,
            #                offsetIndexToBPConstant=windowStep)
        return errorReturns['init']

    # Look through areaDiffs to find point where areaDiffs plateau
    for x in range(indexAtThreshold, len(areaDiffs)-1):
        # if it plateaus, or in the rare case that the diff jumps over the threshold, we have found the boundary point
        if abs(areaDiffs[x]) < areaDiffsThreshold or areaDiffs[x] < areaDiffsThreshold and areaDiffs[x+1] > areaDiffsThreshold :
            boundaryPoint = x * windowStep
            break
    if boundaryPoint == -1:
        # This means we have reached the end of the telomere
        # but we didn't fine the point at which the telomere offset stopped changing.
        logging.debug("Warning: Sequence was not long enough to find a telomere boundary, returning end of sequence as boundary point")
        boundaryPoint = len(areaDiffs) * windowStep
        
    if secondarySearch == True:
        # Lower being towards the telomere, upper being towards the centromere
        telomereOffset = 500
        subTelomereOffset = 1000
        telomereOffsetRE = 30
        subTelomereOffsetRE = 30
        ntPattern = ntPatternEntry[0]

        if isGStrand == True:
            # scanSeq = seq[boundaryPoint-telomereOffset:boundaryPoint+subTelomereOffset]
            lowerIndex = (len(seq) - (boundaryPoint+subTelomereOffset))
            upperIndex = len(seq) - (boundaryPoint-telomereOffset)            
            if lowerIndex < 0:
                lowerIndex = 0
            if upperIndex > len(seq):
                upperIndex = len(seq)
            
            scanSeq = seq[lowerIndex:upperIndex]
            if len(scanSeq) < 100:
                logging.debug("Warning: Sequence was not long enough to perform secondary search, returning original boundary point")
            else:
                secBoundary = getTeloBoundary(scanSeq, isGStrand, composition, teloWindow=90, windowStep=6, changeThreshold=changeThreshold, plateauDetectionThreshold=-60, targetPatternIndex=-1, nucleotideGraphAreaWindowSize=100, showGraphs=False, returnLastDiscontinuity=returnLastDiscontinuity, secondarySearch=False)

                tempBoundary = boundaryPoint + secBoundary - telomereOffset
                if upperIndex == len(seq):
                    tempBoundary = secBoundary

                # ***( upper and lower here might be wrong, check this)
                scanSeq = seq[(len(seq) - (tempBoundary+subTelomereOffsetRE)):len(seq) - (tempBoundary-telomereOffsetRE)]
                scan_pattern = "("+ntPattern+")" + "("+ntPattern+")"
                match = re.search(str(scan_pattern), str(scanSeq))
                if match:
                    secBoundary = len(scanSeq) -match.span()[0]
                    boundaryPoint = tempBoundary + secBoundary - telomereOffsetRE
                else:
                    boundaryPoint = tempBoundary
                    logging.debug("Secondary search failed to find a match, returning original boundary point")
        else:
            # print("Performing secondary search")
            lowerIndex = boundaryPoint-telomereOffset
            upperIndex = boundaryPoint+subTelomereOffset
            if lowerIndex < 0:
                lowerIndex = 0
            if upperIndex > len(seq):
                upperIndex = len(seq)
            scanSeq = seq[lowerIndex:upperIndex]
            if len(scanSeq) < 100:
                logging.debug("Warning: Sequence was not long enough to perform secondary search, returning original boundary point")
            else:
                secBoundary = getTeloBoundary(scanSeq, isGStrand, composition, teloWindow=90, windowStep=6, changeThreshold=changeThreshold, plateauDetectionThreshold=-60, targetPatternIndex=-1, nucleotideGraphAreaWindowSize=100, showGraphs=False, returnLastDiscontinuity=returnLastDiscontinuity, secondarySearch=False)

                tempBoundary = boundaryPoint + secBoundary - telomereOffset
                if lowerIndex == 0:
                    tempBoundary = secBoundary
                scanSeq = seq[tempBoundary-telomereOffsetRE:tempBoundary+subTelomereOffsetRE]
                
                if patternComposition != 1:
                    logging.warning("Warning: Secondary search is not fully compatible with telomere compositions less than 1. Please provide a telomere pattern that covers 6/6 of the expected telomere nucleotides, like 'TTAGGG' or 'GGG...'.")
                else:
                    
                    scan_pattern = "("+ntPattern+")" + "("+ntPattern+")"
                    matches = [match for match in re.finditer(scan_pattern, str(scanSeq))]
                    if matches:
                        teloEnd = matches[-1].end()
                        boundaryPoint = tempBoundary + teloEnd - telomereOffsetRE
                    else:
                        boundaryPoint = tempBoundary
                        logging.debug("Secondary search failed to find a match, returning original boundary point")

    if showGraphs:
        graphLine(areaList, composition[targetPatternIndex]
                  [0] + " Area", windowStep, boundaryPoint=boundaryPoint, pdfOut=pdf)
        # makeOffsetPlot(ntOffsets, composition, windowStep)

    return boundaryPoint


def getIsGStrandFromSeq(seq, GStrandPatternIn, CStrandPatternIn, searchStrandRepeats = 4, minTeloCountDiff = 1, fusedReadTeloRepeatThreshold = 20, maxTelomereGap = 150):
    # We will look at the beginning of the seq and count for C strands, then look at the end and count for G strands
    # the compare the counts to see which is greater and return the result
    
    CStrandPattern = CStrandPatternIn[0]
    GStrandPattern = GStrandPatternIn[0]

    boundaryReg = ".{0,6}"

    CStrandPattern =  ("("+CStrandPattern+")"+boundaryReg) * searchStrandRepeats
    GStrandPattern =  ("("+GStrandPattern+")"+boundaryReg) * searchStrandRepeats
    CStrandPattern = CStrandPattern[:-len(boundaryReg)]
    GStrandPattern = GStrandPattern[:-len(boundaryReg)]

    allCStrands = [match for match in re.finditer(CStrandPattern, str(seq.upper()))]
    # get start index of each match. For C strand start at the beginning and count forward
    cStrandCount = 0
    cStrandMatchLengths = 0
    lastMatch = 0
    for match in allCStrands:
        if lastMatch == 0:
            lastMatch = match.start()

        if match.start() - lastMatch <= maxTelomereGap:
            cStrandCount += 1
            cStrandMatchLengths += match.end() - match.start()
        lastMatch = match.start()
        
    allGStrands = [match for match in re.finditer(GStrandPattern, str(seq.upper()))]

    gStrandCount = 0
    gStrandMatchLengths = 0
    lastMatch = 0
    for match in allGStrands[::-1]:
        if lastMatch == 0:
            lastMatch = match.start()
        if lastMatch - match.start() <= maxTelomereGap:
            gStrandCount += 1
            gStrandMatchLengths += match.end() - match.start()
        lastMatch = match.start()

    if max(cStrandCount, gStrandCount) <= minTeloCountDiff and abs(cStrandCount - gStrandCount) <= minTeloCountDiff:

        # The intention here is to give a last chance to classify the read, as we want to keep as many short reads as possible
        if cStrandMatchLengths > gStrandMatchLengths:
            return False
        elif gStrandMatchLengths > cStrandMatchLengths:
            return True
        return errorReturns['strandType']
    
    if min(cStrandCount, gStrandCount) > 100 or abs(cStrandCount - gStrandCount) <= 0.6 * min(cStrandCount, gStrandCount) or min(cStrandCount, gStrandCount) >= fusedReadTeloRepeatThreshold:
        return errorReturns['fusedRead']

    if cStrandCount > gStrandCount:
        return False
    else:
        return True


def getTeloNPBoundary(seq, isGStrand=None, compositionCStrandIn=teloNPTeloCompositionCStrand, compositionGStrandIn=teloNPTeloCompositionGStrand, teloWindow=100, windowStep=6, changeThreshold=-20, plateauDetectionThreshold=-60, targetPatternIndex=-1, nucleotideGraphAreaWindowSize=750, showGraphs=False, pdf=None, returnLastDiscontinuity=True, secondarySearch = True, maxScanBP=None):
    return getTeloBoundary(seq, isGStrand, compositionCStrand=compositionCStrandIn, compositionGStrand=compositionGStrandIn, teloWindow=teloWindow, windowStep=windowStep, changeThreshold=changeThreshold, plateauDetectionThreshold=plateauDetectionThreshold, targetPatternIndex=targetPatternIndex, nucleotideGraphAreaWindowSize=nucleotideGraphAreaWindowSize, showGraphs=showGraphs, pdf=pdf, returnLastDiscontinuity=returnLastDiscontinuity, secondarySearch=secondarySearch, maxScanBP=maxScanBP)


# --------------------------------------------------------------------------
# itsfix: bounding the scan and anchoring on the array
# --------------------------------------------------------------------------

# Where the telomere array actually stops, and how to anchor on it.
#
#
# `getTeloNPBoundary` finds the boundary from a smoothed composition curve, which
# is the right instrument for "has the telomere pattern broken" but has two
# consequences for anything that then compares reads to each other.
#
# It can be captured from arbitrarily far away.  `returnLastDiscontinuity=True`
# searches for the *last* window still above `changeThreshold` anywhere in the
# read, so a region whose composition matches the telomere pattern -- a real
# interstitial telomeric block, or just a C-rich subtelomeric tandem repeat, since
# the C-strand pattern `CCC(?!CCC)[ATCG]{3}` scores C-richness rather than
# telomere -- pulls the boundary out to it.  Observed on ONT reads: 20q captured
# at 108 kb by a degenerate `CCCTGA`/`CCCCGA` block, Xp at 216 kb by
# `CCTCTCTCCCCGTCCCCCCTCCCT` with no canonical repeat in it at all.
#
# And it carries the smoothing width as slop.  The boundary is where a 750 bp
# window's area plateaus, snapped by a regex on that same permissive pattern, so
# it lands within a couple of hundred bases of the array's end but not on it, and
# not in repeat phase.  Fine for a length; visible as jitter when reads are stacked
# against it and their variant repeats are supposed to line up.
#
# `telomereExtent` answers the first problem by giving `maxScanBP` a value that
# cannot be a distant repeat, and `snapToArrayEnd` answers the second by moving
# the call onto the end of the last run of the CALLER'S OWN telomere pattern.
# `getTeloNPArrayBoundary` is the two of them around a normal boundary call.
#
# The extent walk folds each hexamer to its minimal rotation, so a tandem
# repeat reads the same whatever frame it is seen in and the array's phase
# never has to be recovered -- which matters on ONT, where indels have
# usually destroyed it.
# Runs of any class count, canonical or variant, so an array finishing in a
# kilobase of `CCCGAA` is not cut short at the point the canonical repeat stops.
#
# The snap does not fold, and does not use that walk at all.  Folding is the
# right answer to "how far does the array reach" and the wrong one to "where
# shall every read agree to put its boundary": a rotation class agrees with
# itself in six places, and the run end it hands back is wherever that read's
# own repeat happened to start.  `snapToArrayEnd` takes the run ends of the
# pattern the CALLER is judging every read by -- `trc`'s `--telo-regex` -- so
# the landmark is one the reads share rather than one each read has privately.
#
# Measured on HG08434 LCL-ONT-UL, the 2,512 clustered reads of 92 clusters,
# scored as mean per-column plurality agreement over the 2,500 bp of array
# behind b0 with the clusters held FIXED, so only b0 moves.  Over the 300 bp
# nearest the boundary, where the variant repeats that identify an arm sit:
#
#     0.859   unsnapped, teloBP's call as it comes
#     0.959   snapped onto the last TANDEM run end       (the old behaviour)
#     0.964   snapped onto the last --telo-regex run end (this one)
#
# 76 of the 92 clusters improve against no snap and 10 lose, the worst by
# 0.027; against the tandem walk it is 66 better and 19 worse.  Moving b0 by
# the same distances at RANDOM scores 0.52, so the metric is measuring
# alignment and not movement.  Forcing the reads with no run end within
# `snapBP` onto their nearest single match instead of leaving them alone was
# tried and is worse (0.955), which is why the give-up branch stays.

UNIT_W = 6
MIN_UNITS = 3          # two identical hexamers can be one substitution in canonical
MAX_GAP = 1_000        # non-repeat this long ends the array; shorter is crossed
MARGIN = 3_000         # scanned past the extent, so the boundary call still has
                       # subtelomere to see the pattern break against
SNAP_BP = 300          # how far from the boundary call a run end may be taken

_COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")

_FOLD = {}


def _rc(s):
    return s.translate(_COMP)[::-1]


def _fold(u):
    """Minimal rotation of one hexamer, memoised on the 4096 possibilities."""
    r = _FOLD.get(u)
    if r is None:
        r = _FOLD[u] = min(u[i:] + u[:i] for i in range(len(u)))
    return r


def tandemRunEnds(seq, w=UNIT_W, minUnits=MIN_UNITS, maxGap=MAX_GAP):
    """Ends of the tandem runs reachable from position 0, in order.

    A run is a maximal stretch of one rotation class of at least `minUnits`
    units.  Walking outward from the telomere end, a run is reachable if it
    starts no more than `maxGap` past the end of the last reachable one; the
    first run that does not ends the walk, and everything past it is subtelomere
    as far as this function is concerned.  Returns [] if there is no run at all.

    One forward pass with an early exit, so the cost is the length of the array
    plus `maxGap` rather than the length of the read.
    """
    n = len(seq)
    if n < w:
        return []
    ends, end, i = [], 0, 0
    last = n - w
    while i <= last:
        c = _fold(seq[i:i + w])
        j = i
        while j < last and _fold(seq[j + 1:j + 1 + w]) == c:
            j += 1
        span = (j - i) + w
        if span // w >= minUnits:
            if end and i - end > maxGap:
                break
            end = i + span
            ends.append(end)
        elif end and i - end > maxGap:
            break
        i = j + 1
    return ends


def telomereExtent(seq, isGStrand=False, **kw):
    """How far the tandem repeat array reaches from the telomere end, in bp.

    0 when the read has no tandem run to walk at all.  On the G strand the read
    is reversed first, so the answer is a distance from the end of the sequence
    and matches `getTeloBoundary`'s own G-strand convention.
    """
    ends = tandemRunEnds(seq[::-1] if isGStrand else seq, **kw)
    return ends[-1] if ends else 0


def matchRunEnds(seq, rgx, minUnits=MIN_UNITS, maxGap=MAX_GAP):
    """Ends of the `rgx` match runs reachable from position 0, in order.

    A run is a maximal chain of ABUTTING matches.  `finditer` tiles the
    sequence left to right, so a chain is a stretch the pattern covers with no
    gap at all, and it counts as a run at `minUnits` matches -- a lone match is
    a coincidence, three in a row is the array.  Reachability is
    `tandemRunEnds`': a run is reachable if it starts no more than `maxGap`
    past the end of the last reachable one, and the first that does not ends
    the walk.  Returns [] if the pattern never runs.

    Allowing a chain to skip a base or two was tried, on the theory that an ONT
    indel breaks the tiling mid-array; it scores slightly WORSE (0.963 against
    0.964), because a break in the tiling is also how the end of the array
    announces itself.  The walk stops one `maxGap` past the last run, so it
    costs the length of the array rather than the length of the read.
    """
    ends, last, st, en, n = [], 0, None, None, 0
    for m in rgx.finditer(seq):
        if en is not None and m.start() == en:
            en, n = m.end(), n + 1
            continue
        if st is not None and n >= minUnits:
            if ends and st - last > maxGap:
                return ends
            ends.append(en)
            last = en
        if ends and m.start() - last > maxGap:
            return ends
        st, en, n = m.start(), m.end(), 1
    if st is not None and n >= minUnits and not (ends and st - last > maxGap):
        ends.append(en)
    return ends


def snapToArrayEnd(seq, boundaryPoint, rgx, isGStrand=False, snapBP=SNAP_BP,
                   **kw):
    """Move `boundaryPoint` onto the nearest `rgx` run end within `snapBP`.

    The boundary is left alone if no run end is that close, and if `rgx` is
    None there is no landmark to move onto at all.  That is the honest outcome
    for a read whose array does not end in the pattern: a snap that reached
    further would be inventing a landmark rather than recovering one, and the
    benchmark above agrees -- forcing those reads somewhere costs alignment.

    On the G strand the read is reverse-complemented, where `telomereExtent`
    merely reverses it.  Both put the array at the front and both measure from
    the end of the read, but a pattern belongs to a strand and reversing a
    sequence does not give the other one.
    """
    if rgx is None or boundaryPoint is None or boundaryPoint < 0:
        return boundaryPoint
    ends = matchRunEnds(_rc(seq) if isGStrand else seq, rgx, **kw)
    if not ends:
        return boundaryPoint
    best = min(ends, key=lambda e: abs(e - boundaryPoint))
    return best if abs(best - boundaryPoint) <= snapBP else boundaryPoint


def getTeloNPArrayBoundary(seq, isGStrand=False, margin=MARGIN, snap=True,
                           snapBP=SNAP_BP, snapRgx=None, extentKw=None, **kw):
    """`getTeloNPBoundary`, bounded by the array's own extent and snapped to it.

    The scan is capped at `telomereExtent` + `margin`, which is what stops a
    distant repeat from capturing `returnLastDiscontinuity`; the margin has to be
    wide enough that the boundary call still sees the pattern break, and 3 kb is
    comfortably past every boundary observed within a correctly extented array.
    The result is then snapped onto the last run of `snapRgx`, so reads stacked
    against it agree on a landmark they share rather than on the smoothing
    width.  No pattern means no snap: `snapRgx` is the caller's telomere
    pattern, and without one there is nothing for reads to agree about.

    A read with no tandem run has no extent to bound with, so it falls through to
    an ordinary unbounded call rather than being scanned over `margin` bp of
    nothing and reported as boundaryless.
    """
    ek = dict(extentKw or {})
    extent = telomereExtent(seq, isGStrand, **ek)
    cap = extent + margin if extent else None
    b = getTeloNPBoundary(seq, isGStrand=isGStrand, maxScanBP=cap, **kw)
    if b is None or b < 0:
        return b
    if not snap or snapRgx is None:
        return b
    return snapToArrayEnd(seq, b, snapRgx, isGStrand, snapBP)
