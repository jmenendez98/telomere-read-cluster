# TeloBP -- https://github.com/GreiderLab/TeloBP
# MIT License, Copyright (c) 2024 Ramin Kahidi
import logging
import re

import numpy as np

logging.basicConfig(level=logging.ERROR)


expectedTeloCompositionQ = [
    ["GGG", 3/6],
]

expectedTeloCompositionP = [
    ["CCC", expectedTeloCompositionQ[0][1]],
]

teloNPTeloCompositionGStrand = [["(?!GGG)[ATC]{2}.GGG|(?!AAA)[AT]{3}AAA|TTAGG.", 6/6, 6]]

teloNPTeloCompositionCStrand = [["CTTCTT|CCTGG|CCC(?!CCC)[ATCG]{3}", 6/6, 6]]

areaDiffsThreshold = 0.2


errorReturns = {"init": -1, "fusedRead": -10,"strandType": -20, "seqNotFound": -1000}


def is_regex_pattern(input_string):
    return bool(re.search(r'[^a-zA-Z]', input_string))


def getGraphArea(offsets, targetColumn, windowSize):
    data = np.array(offsets)
    transposed_data = data.T
    areaList = []
    row = transposed_data[targetColumn, :]

    for i in range(0, len(row) - windowSize, 1):
        area = row[i:i + windowSize].sum()
        areaList.append((area / windowSize))
    return areaList


def graphLine(rowIn, labelIn, windowStep, boundaryPoint=-1, pdfOut=None):
    import matplotlib.pyplot as plt

    row = rowIn
    if boundaryPoint < 0.5 * len(rowIn):
        row = rowIn[:int((2 * boundaryPoint)/windowStep)]

    x = np.arange(len(row))
    x = x * windowStep
    fig, ax = plt.subplots()

    ax.plot(x, row, label=labelIn)

    ax.set_xlabel('Distance from end of sequence')
    ax.set_ylabel('Nucleotide offset')
    ax.set_title('Nucleotide Offsets from Expected Telomere Composition')

    if boundaryPoint != -1:
        ax.axvline(boundaryPoint, color='red', label='Boundary Point')

    ax.legend()

    if pdfOut is not None:
        pdfOut.savefig(fig)
    elif pdfOut is None:
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


def getTeloBoundary(seq, isGStrand = None, compositionGStrand=[], compositionCStrand = [], teloWindow=100, windowStep=6, changeThreshold=-20, plateauDetectionThreshold=-50, targetPatternIndex=-1, nucleotideGraphAreaWindowSize=500, showGraphs=False, pdf=None, returnLastDiscontinuity=False, secondarySearch = False, maxScanBP=None):
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

    if isGStrand == None:
        isGStrand = getIsGStrandFromSeq(seq, compositionGStrand[targetPatternIndex], compositionCStrand[targetPatternIndex])
        if isGStrand < 0 or not isinstance(isGStrand, bool) and not isinstance(isGStrand, np.bool_):
            if isGStrand == errorReturns['fusedRead']:
                logging.warning(f"Warning: fused strand likely, returning {errorReturns['fusedRead']}")
                return errorReturns['fusedRead']
            logging.warning(f"Warning: could not determine telomere strand type from sequence, returning {errorReturns['strandType']}")
            return errorReturns['strandType']

    if maxScanBP is not None and len(seq) > maxScanBP:
        seq = seq[-maxScanBP:] if isGStrand else seq[:maxScanBP]

    composition = []
    if isGStrand:
        composition = compositionGStrand
    else:
        composition = compositionCStrand

    try:
        validate_parameters(seq, isGStrand, composition, teloWindow, windowStep, plateauDetectionThreshold, changeThreshold, targetPatternIndex, nucleotideGraphAreaWindowSize, showGraphs)
    except Warning as w:
        logging.warning(f"Initial validation failed for read, returning {errorReturns['init']}: {w}")
        return errorReturns['init']

    for i in range(0, len(seq) - teloWindow, windowStep):
        teloSeq = ""
        if isGStrand == True:
            teloSeq = seq[(len(seq) - i) - teloWindow:len(seq) - i]
        else:
            teloSeq = seq[i:i + teloWindow]
        teloLen = len(teloSeq)
        teloSeqUpper = str(teloSeq.upper())

        currentOffsets = []
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
                rawOffsetValue = (
                    (patternCount * len(ntPattern)) / teloLen - patternComposition)
                if patternComposition != 0:
                    percentOffsetValue = (
                        rawOffsetValue / patternComposition) * 100
                    currentOffsets.append(percentOffsetValue)
                else:
                    percentOffsetValue = (rawOffsetValue) * 100
                    currentOffsets.append(percentOffsetValue)

        ntOffsets.append(currentOffsets)

    areaList = getGraphArea(ntOffsets, targetPatternIndex, graphAreaWindowSize)
    areaDiffs = np.diff(areaList)
    indexAtThreshold = -1

    if returnLastDiscontinuity:
        indexAtThreshold = (next((y for y in range(len(areaList) - 2, 0, -1) if (
            areaList[y] > changeThreshold and (0 > (areaList[y + 1] - areaList[y])))), indexAtThreshold))
        if indexAtThreshold != -1:
            indexAtThreshold = (next((y for y in range(indexAtThreshold, len(areaList) - 2) if (
                areaList[y] < plateauDetectionThreshold and (0 > (areaList[y + 1] - areaList[y])))), indexAtThreshold))
        else:
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
        return errorReturns['init']

    for x in range(indexAtThreshold, len(areaDiffs)-1):
        if abs(areaDiffs[x]) < areaDiffsThreshold or areaDiffs[x] < areaDiffsThreshold and areaDiffs[x+1] > areaDiffsThreshold :
            boundaryPoint = x * windowStep
            break
    if boundaryPoint == -1:
        logging.debug("Warning: Sequence was not long enough to find a telomere boundary, returning end of sequence as boundary point")
        boundaryPoint = len(areaDiffs) * windowStep
        
    if secondarySearch == True:
        telomereOffset = 500
        subTelomereOffset = 1000
        telomereOffsetRE = 30
        subTelomereOffsetRE = 30
        ntPattern = ntPatternEntry[0]

        if isGStrand == True:
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

    return boundaryPoint


def getIsGStrandFromSeq(seq, GStrandPatternIn, CStrandPatternIn, searchStrandRepeats = 4, minTeloCountDiff = 1, fusedReadTeloRepeatThreshold = 20, maxTelomereGap = 150):
    CStrandPattern = CStrandPatternIn[0]
    GStrandPattern = GStrandPatternIn[0]

    boundaryReg = ".{0,6}"

    CStrandPattern =  ("("+CStrandPattern+")"+boundaryReg) * searchStrandRepeats
    GStrandPattern =  ("("+GStrandPattern+")"+boundaryReg) * searchStrandRepeats
    CStrandPattern = CStrandPattern[:-len(boundaryReg)]
    GStrandPattern = GStrandPattern[:-len(boundaryReg)]

    allCStrands = [match for match in re.finditer(CStrandPattern, str(seq.upper()))]
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


# trc additions: cap teloBP's scan near the array and snap its call onto
# the end of a telomeric run

# repeat unit, min copies per run, max gap between runs
UNIT_W = 6
MIN_UNITS = 3
MAX_GAP = 1_000
# scan this far past the array; snap within this far
MARGIN = 3_000
SNAP_BP = 300

_COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")

# unit -> its smallest rotation
_FOLD = {}


def _rc(s):
    return s.translate(_COMP)[::-1]


def _fold(u):
    """Smallest rotation of u, so every phase of a repeat compares equal."""
    r = _FOLD.get(u)
    if r is None:
        r = _FOLD[u] = min(u[i:] + u[:i] for i in range(len(u)))
    return r


def tandemRunEnds(seq, w=UNIT_W, minUnits=MIN_UNITS, maxGap=MAX_GAP):
    """Ends of tandem runs of any w-bp unit (>= minUnits copies) from
    seq's start, stopping at the first gap over maxGap."""
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
    """bp from the telomeric end to the end of the last tandem run; 0 if
    none."""
    ends = tandemRunEnds(seq[::-1] if isGStrand else seq, **kw)
    return ends[-1] if ends else 0


def matchRunEnds(seq, rgx, minUnits=MIN_UNITS, maxGap=MAX_GAP):
    """Ends of runs of >= minUnits back-to-back rgx matches from seq's
    start, stopping at the first gap over maxGap."""
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
    """boundaryPoint moved to the nearest matchRunEnds end within snapBP,
    else unchanged."""
    if rgx is None or boundaryPoint is None or boundaryPoint < 0:
        return boundaryPoint
    ends = matchRunEnds(_rc(seq) if isGStrand else seq, rgx, **kw)
    if not ends:
        return boundaryPoint
    best = min(ends, key=lambda e: abs(e - boundaryPoint))
    return best if abs(best - boundaryPoint) <= snapBP else boundaryPoint


def getTeloNPArrayBoundary(seq, isGStrand=False, margin=MARGIN, snap=True,
                           snapBP=SNAP_BP, snapRgx=None, extentKw=None, **kw):
    """teloNP boundary scanned no further than margin past the tandem
    array, then snapped onto snapRgx if snap; negative if teloBP finds
    none."""
    ek = dict(extentKw or {})
    extent = telomereExtent(seq, isGStrand, **ek)
    cap = extent + margin if extent else None
    b = getTeloNPBoundary(seq, isGStrand=isGStrand, maxScanBP=cap, **kw)
    if b is None or b < 0:
        return b
    if not snap or snapRgx is None:
        return b
    return snapToArrayEnd(seq, b, snapRgx, isGStrand, snapBP)
