__version__ = '1.0.0'


import datetime
import multiprocessing
import argparse

import decimal as dc
dc.getcontext().prec = 5

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import copy
import subprocess
import os
import gzip
import timeit
import shutil

from tqdm import tqdm
from collections import Counter
from Bio import Align

def save_config(config, filename):
    """
    Write timestamp and commandline arguments to file.

    Args:
        config (dict): Config parameters and values to write.
        filename (str): Name of the file to write to.
    """
    with open(filename, "w") as f:
        f.write("Commandline_argument\tValue\n")
        f.write("Time\t{}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%d")))
        f.write("getITD_version\t{}\n".format(__version__))
        for param in sorted(config.keys()):
            if param not in ["ANNO", "DOMAINS"]:
                f.write("{}\t{}\n".format(param, config[param]))

def load_config(filename):
    """
    Load config parameters from file.

    Args:
        filename (str): Name of the file to read config from.

    Returns:
        Dictionary with config parameter - value pairs.
    """
    config = {}
    with open(filename, "r") as f:
        for line in f:
            key, val = line.strip("\n").split("\t")
            if key not in ["Time", "Commandline_argument"]:
                try:
                    config[key] = int(val)
                except:
                    try:
                        config[key] = float(val)
                    except:
                        config[key] = val

    # recognize string as dict
    if "COST_ALIGNED" in config:
        config["COST_ALIGNED"] = eval(config["COST_ALIGNED"])
    return config



# child processes spawned on Windows by multiprocessing do not
# receive variables set in __main__ of parent process
# -->  they cannot access config {} values set in __main__
# --> to circumvent this, __main__ saves config and children
#     spawned by multiprocessing load it from file
if __name__ in ['__mp_main__', 'getitd']:
    try:
        current_dir = os.getcwd()
        config = load_config(os.path.join(current_dir, "config.txt"))
    except OSError:
        print("NO CONFIG FOUND")
        config = {}
else:
    # mimic Windows style process spawning on Linux:
    # multiprocessing.set_start_method("spawn")
    try:
        config
    except NameError:
        config = {}


def parallelize(function, args, cores):
    """
    Parallelize a given function across a given number of cores.

    Args:
        function (function): Function or method to parallelize.

        args (tuple): Tuple of function's arguments.

        cores (int): Number of cores to utilize.

    Returns:
        List of function's outputs.
    """
    with multiprocessing.Pool(cores) as p:
        return p.map(function, args)

def bbmap_process_quick(fastq1, fastq2, bbmap_path, temp_path):
    assert os.path.isdir(bbmap_path)
    assert os.path.isfile(f"{bbmap_path}/bbmerge.sh")
    assert os.path.isfile(f"{bbmap_path}/bbduk.sh")
    
    start_time = timeit.default_timer()
    
    if os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
        
    subprocess.run(["mkdir", temp_path])
    
    bbmap_log = ""
    
    # Phase 1 merging
    ret = subprocess.run([
        f"{bbmap_path}/bbmerge.sh", 
        f"in1={fastq1}", f"in2={fastq2}",
        f"out={temp_path}/merged.fastq", 
        f"outu={temp_path}/unmerged.fastq",
        f"ihist={temp_path}/hist.tsv"
    ], capture_output=True, text=True)
    bbmap_log += ret.stderr + '\n'

    # Phase 3 - average bqs filtering
    ret = subprocess.run([
        f"{bbmap_path}/bbduk.sh", 
        f"in={temp_path}/merged.fastq",
        f"out={temp_path}/cleaned.fastq", 
        "maq=25"
    ], capture_output=True, text=True)
    bbmap_log += ret.stderr + '\n'
    
    print(f"BBmap time taken - {round(timeit.default_timer() - start_time, 2)} sec")
    return bbmap_log

def bbmap_process(fastq1, fastq2, bbmap_path, temp_path):
    assert os.path.isdir(bbmap_path)
    assert os.path.isfile(f"{bbmap_path}/bbmerge.sh")
    assert os.path.isfile(f"{bbmap_path}/bbduk.sh")
    
    start_time = timeit.default_timer()
    
    if os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
        
    subprocess.run(["mkdir", temp_path])
    
    bbmap_log = ""
    
    # Phase 1 merging
    ret = subprocess.run([
        f"{bbmap_path}/bbmerge.sh", 
        f"in1={fastq1}", f"in2={fastq2}",
        f"out={temp_path}/merged.fastq", 
        f"outu={temp_path}/unmerged.fastq",
        f"ihist={temp_path}/hist.tsv"
    ], capture_output=True, text=True)
    bbmap_log += ret.stderr + '\n'
    
    # Phase 2 merging
    ret = subprocess.run([
        f"{bbmap_path}/bbduk.sh", 
        f"in={temp_path}/unmerged.fastq",
        f"out={temp_path}/qtrimmed.fastq", 
        "qtrim=r", "trimq=20"
    ], capture_output=True, text=True)
    bbmap_log += ret.stderr + '\n'
    
    ret = subprocess.run([
        f"{bbmap_path}/bbmerge.sh", 
        f"in={temp_path}/qtrimmed.fastq",
        f"out={temp_path}/merged2.fastq",
        f"ihist={temp_path}/hist2.tsv"
    ], capture_output=True, text=True)
    bbmap_log += ret.stderr + '\n'
    
    # Concatenate into first file
    f1 = open(f"{temp_path}/merged.fastq", 'a+')
    f2 = open(f"{temp_path}/merged2.fastq", 'r')
    f1.write(f2.read())
    f1.close()
    f2.close()

    # Phase 3 - average bqs filtering
    ret = subprocess.run([
        f"{bbmap_path}/bbduk.sh", 
        f"in={temp_path}/merged.fastq",
        f"out={temp_path}/cleaned.fastq", 
        "maq=30"
    ], capture_output=True, text=True)
    bbmap_log += ret.stderr + '\n'
    
    print(f"BBmap time taken - {round(timeit.default_timer() - start_time, 2)} sec")
    return bbmap_log

def is_gz_file(filename):
    """
    Check whether a given file is gzipped or not,
    using its magic number.

    Args:
        filename: Name of the file to read.

    Returns:
        bool, True when gzipped, False otherwise.
    """
    with open(filename, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

def read_fastq(fastq_file):
    """
    Read sequence fastq file and extract sequences and BQS.

    Args:
        fastq_file: Name of the fastq file to read, R1 or R2.

    Returns:
        List of Read() objects.
    """
    reads = [] # simple list of fastq sequences
    try:
        if is_gz_file(fastq_file):
            open_fct = gzip.open
        else:
            open_fct = open
        
        with open_fct(fastq_file, 'rt') as f:
            line = f.readline()
            while line:
                _ = line
                read_seq = f.readline().rstrip(os.linesep)
                _ = f.readline()
                _ = f.readline().rstrip(os.linesep)
                reads.append(read_seq)
                line = f.readline()
    except IOError as e:
        print("---\nCould not read fastq file {}!\n---".format(fastq_file))
    
    return reads

def read_reference(filename):
    """
    Read in WT reference sequence.

    Args:
        filename (str): Name of the file to be read.

    Returns:
        Reference sequence, stripped of trailing newlines.
    """
    with open(filename, 'r') as f:
        ref = f.read()
    ref = ref.splitlines()
    assert len(ref) == 1
    return ref[0]

# add column names!
def read_annotation(filename):
    """
    Read in WT reference annotation file.

    For each bp of the WT reference, provides genomic, transcriptomic
    and proteomic coordinate, exon/intron annotation and the respective
    reference bp.

    Args:
        filename (str): Name of the file to be read.

    Returns:
        pd.DataFrame of the annotation.
    """
    try:
        return pd.read_csv(filename, sep='\t')
    except IOError as e:
        print("\nAnnotation file was not provided or cannot be accessed!\n")
        return None

def annotateCoords(anno_df):
    """
    Annotate HGVS coordinates to given reference file
    """
    df = anno_df.copy(deep=True)
    
    # find first exon
    firstExonCoord = 0
    for i in range(len(df)):
        if df.iloc[i]["region"].find("exon") > -1:
            firstExonCoord = i
            break

    assert firstExonCoord > -1
    assert int(df.iloc[firstExonCoord]["transcript_bp"]) > 0
    df["HGVScoord"] = ""
    
    # Annotate upstream intron if required
    if firstExonCoord > 0:
        cdot = int(df.iloc[i]["transcript_bp"])
        for i in range(firstExonCoord, -1, -1):
            df.loc[i, "HGVScoord"] = f"c.{int(cdot)}{int(i)-int(firstExonCoord)}"
    
    # Now annotate first exon
    i = firstExonCoord
    inIntron = False
    while i < len(anno_df):
        if df.iloc[i]["region"].find("exon") > -1:
            if inIntron:
                inIntron = False
            df.loc[i, "HGVScoord"] = f"c.{int(df.iloc[i]['transcript_bp'])}"
        elif not inIntron:
            inIntron = True
            cdot = int(df.iloc[i-1]['transcript_bp'])
            # find next exon coord
            nextExonCoord = 0
            lastExonCoord = i-1
            for j in range(i, len(df)):
                if df.iloc[j]["region"].find("exon") > -1:
                    nextExonCoord = j
                    break
            # annotate first base of intron
            df.loc[i, "HGVScoord"] = f"c.{int(cdot)}+{1}"
        elif nextExonCoord > 0:
            # need to find whether position is closer to donor or acceptor
            if i - lastExonCoord < nextExonCoord - i:
                df.loc[i, "HGVScoord"] = f"c.{int(cdot)}+{i - lastExonCoord}"
            else:
                df.loc[i, "HGVScoord"] = f"c.{int(cdot)+1}-{nextExonCoord - i}"
        else:
            assert inIntron
            df.loc[i, "HGVScoord"] = f"c.{int(cdot)}+{i - lastExonCoord}"
        i += 1

    return df

def ar_to_vaf(ar):
    """
    Convert AR to VAF.

    VAF (variant allele frequency) = V-AF
    AR (allele ratio) = V-AF / WT-AF
    V-AF + WT-AF = 100 (%)

    Args:
        ar (float): AR to convert.

    Returns:
        VAF (float)

    """
    return ar/(ar + 1) * 100 # * 100 because VAF is in %

def vaf_to_ar(vaf):
    """
    Convert VAF to AR.

    VAF (variant allele frequency) = V-AF
    AR (allele ratio) = V-AF / WT-AF
    V-AF + WT-AF = 100 (%)

    Note:
        if VAF == 100:
            AR = -1
            (instead of 100 / 0)

    Args:
        vaf (dc.Decimal): VAF to convert.

    Returns:
        AR (dc.Decimal)
    """
    if vaf == 100:
        return -1
    return vaf/(100 - vaf)

def save_stats(stat, filename):
    """
    Write statistics to file.

    Args:
        stat (str): Statistic to save.
        filename (str): Name of the file to write to.
    """
    print(stat)
    with open(filename, "a") as f:
        f.write(stat + "\n")

def str_to_bool(string):
    """
    Convert literal str to bool, such that
        'True' ->  True
        'False' -> False
    This is required to parse boolean command line
    arguments, which are all initially of type str.
    Calling `bool(str)` evaluates to `True` whenever
    the str is not empty. Thus, without this function,
    'False' would also evaluate to `True`.

    Args:
        string (str): To be converted.

    Returns:
        Literal boolean (bool) of string.
    """
    if string.lower() in ('true', 'True'):
        return True
    elif string.lower() in ('false', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value (True or False) expected.')

class Mutation(object):
    def __init__(
        self,
        mutType = "ins",
        pos_str = None,
        ins_str = "",
        counts = 0):
        
        self.mutType = mutType
        assert pos_str is not None
        self.pos = [int(i) for i in pos_str.split("-")]
        if len(self.pos) == 1:
            self.pos.append(self.pos[0])
        self.ins_str = ins_str
        self.counts = counts
        self.comutations = {}
        
    def add(self, count):
        self.counts += count
        return(self)
    
    def nameMut(self):
        if self.mutType == "snp":
            return(f"{self.pos[0]}{self.ins_str}")
        elif self.pos[0] == self.pos[1]:
            return(f"{self.pos[0]}{self.mutType}{self.ins_str}")
        return(f"{self.pos[0]}-{self.pos[1]}{self.mutType}{self.ins_str}")
    
    def addComutation(self, mutName, counts):
        if mutName in self.comutations.keys():
            self.comutations[mutName] += counts
        else:
            self.comutations[mutName] = counts
        return(self)
    
    def netInsert(self):
        if self.mutType[0:3] == "ins":
            return len(self.ins_str)
        elif self.mutType == "del":
            return -self.pos[1] + self.pos[0] - 1
        elif self.mutType[0:6] == "delins":
            return len(self.ins_str) - self.pos[1] + self.pos[0] - 1
        elif self.mutType == "dup":
            return self.pos[1] - self.pos[0] + 1
        else:
            return 0
        
    def getInsertPos(self):
        if self.mutType[0:3] == "ins":
            return self.pos[0]
        elif self.mutType == "del":
            return self.pos[0]
        elif self.mutType[0:6] == "delins":
            return self.pos[0]
        elif self.mutType == "dup":
            return self.pos[1] + 1
        else:
            return self.pos[0]

def getHGVS(seq, ref, config, verbose = False):
    """
    Aligns a sequence with respect to the reference
    - returns 3 lists: coordinates, operations, insert sequence
    - Also returns start / end coordinates of alignment (-1 if not aligned)
    """
    
    q_alns = []
    r_alns = []

    aligner2 = config["ALIGNER"]

    aln = aligner2.align(seq, ref)
    if not aln:
        return [],[],[], -1, -1
    if verbose:
        print(aln[-1])
        
    coords = aln[-1].coordinates
    # print(coords)
    minAlignLen = 6
    for j in range(len(coords[0]) - 1):
        if coords[0][j+1] - coords[0][j] >= minAlignLen and coords[1][j+1] - coords[1][j] >= minAlignLen:
            # if match must be a block of at least 6
            q_alns.append(coords[0][j:j+2])
            r_alns.append(coords[1][j:j+2])
            
    q_alns = np.array(q_alns)
    r_alns = np.array(r_alns)

    # No alignment
    if q_alns.shape[0] == 0:
        return [],[],[], -1, -1

    # Boundaries of alignment
    ref_start, ref_end = r_alns[0][0], r_alns[-1][1]

    # Insufficient alignment to reference
    minRefAlignFraction = 0.4
    if (ref_end - ref_start) < minRefAlignFraction * len(ref):
        return [],[],[], -1, -1                
    
    # Single alignment - likely WT
    if q_alns.shape[0] == 1:
        return [],[],[], ref_start, ref_end
   
    if verbose:
        print(q_alns)
        print(r_alns)

    # calculate the sum inserted / deleted. Abort if this exceeds 70% of length of sequence
    nTotIndel = 0    
    ops, rC, rSeq = [],[],[]
    for i in range(len(q_alns) - 1):
        if q_alns[i+1][0] == q_alns[i][1]:
            # no gaps in query sequence, i.e. no insertion
            if r_alns[i+1][0] > r_alns[i][1]:
                # gap in reference alignment deletion
                ops.append("del")
                rSeq.append("")
                rC.append(f"{str(r_alns[i][1])}-{str(r_alns[i+1][0]-1)}")
                nTotIndel += (r_alns[i][1] - r_alns[i+1][0])
            else:
                pass
        else:
            if r_alns[i+1][0] == r_alns[i][1]:
                # simple insertion
                insSeq = seq[q_alns[i][1]:q_alns[i+1][0]]
                nTotIndel += len(insSeq)
                
                # check if duplication
                isDup = False
                if len(insSeq) <= q_alns[i][1]:
                    dupSeq = seq[(q_alns[i][1] - len(insSeq)):(q_alns[i+1][0] - len(insSeq))]
                    if insSeq == dupSeq:
                        isDup = True
                        ops.append("dup")
                        rSeq.append("")
                        rC.append(f"{str(r_alns[i][1] - len(insSeq))}-{str(r_alns[i][1] - 1)}")
                if not isDup:
                    rC.append(f"{str(r_alns[i][1])}-{str(r_alns[i+1][0]+1)}")
                    ops.append(f"ins[{len(insSeq)}]")
                    rSeq.append(insSeq)

            elif r_alns[i+1][0] > r_alns[i][1]:
                # delins
                insSeq = seq[q_alns[i][1]:q_alns[i+1][0]]
                rC.append(f"{str(r_alns[i][1])}-{str(r_alns[i+1][0]-1)}")
                ops.append(f"delins[{r_alns[i+1][0] - r_alns[i][1]},{len(insSeq)}]")
                rSeq.append(insSeq)
                nTotIndel += len(insSeq) + (r_alns[i+1][0] - r_alns[i][1]) # del + ins
            else:
                # overlapping alignment preceeded by novel insert, treat as insertion
                # true insert length is longer than mapped
                # attach duplicated alignment to end of novel insert
                dupLen = r_alns[i][1] - r_alns[i+1][0]
                insSeq = seq[q_alns[i][1]:(q_alns[i+1][0] + dupLen)]
                rC.append(f"{str(r_alns[i][1])}-{str(r_alns[i][1]+1)}")
                ops.append(f"ins[{len(insSeq)}]")
                rSeq.append(insSeq)

    maxFracIsIndel = 0.7
    if (nTotIndel + ref_end - ref_start) * maxFracIsIndel < nTotIndel:
        return [],[],[], -1, -1

    if verbose:
        print([f"{c}{o}{s}" for c, o, s in zip(rC, ops, rSeq)])
    
    return rC, ops, rSeq, ref_start, ref_end

def generateMutSeq(ref, hgvs, returnComplex = False):
    """
    Generates a sequence which is a mutated sequence from the given reference,
    using the given hgvs instructions (inserts / deletions only, not SNPs)
    """
    
    refStarts = []
    refEnds = []
    insSeqs = []
    
    for hg in hgvs:
        posStart = -1
        posEnd = -1
        seq = ""
        if hg.find("delins") > -1:
            mutStart = hg.find("delins")
            pos = hg[0:mutStart]
            seq = hg[mutStart+6:]
            if seq.find("]") > -1:
                seq = seq.split("]")[1]
            if pos.find("-") > 0:
                posStart, posEnd = pos.split("-")
                posStart = int(posStart)
                posEnd = int(posEnd) + 1
            else:
                posStart = int(pos)
                posEnd = int(pos) + 1
        elif hg.find("ins") > -1:
            mutStart = hg.find("ins")
            pos = hg[0:mutStart]
            seq = hg[mutStart+3:]
            if seq.find("]") > -1:
                seq = seq.split("]")[1]
            if pos.find("-") > 0:
                posStart, posEnd = pos.split("-")
                posStart = int(posStart)
                posEnd = int(posEnd) - 1
            else:
                posStart = int(pos)
                posEnd = int(pos) - 1
        elif hg.find("del") > -1:
            mutStart = hg.find("del")
            pos = hg[0:mutStart]
            if pos.find("-") > 0:
                posStart, posEnd = pos.split("-")
                posStart = int(posStart)
                posEnd = int(posEnd) + 1
            else:
                posStart = int(pos)
                posEnd = int(pos) + 1
        elif hg.find("dup") > -1:
            mutStart = hg.find("dup")
            pos = hg[0:mutStart]
            if pos.find("-") > 0:
                posStart, posEnd = pos.split("-")
                posStart = int(posStart)
                posEnd = int(posEnd) + 1
            else:
                posStart = int(pos)
                posEnd = int(pos) + 1
            seq = ref[posStart:posEnd]
            posEnd = posStart # as this is duplication
        else:
            pass
        
        if posStart > -1:
            refStarts.append(posStart)
            refEnds.append(posEnd)
            insSeqs.append(seq)
        
    refStarts = np.array(refStarts)
    refEnds = np.array(refEnds)
    insSeqs = np.array(insSeqs)
    
    arg_sort = np.argsort(refStarts)
    refStarts = refStarts[arg_sort]
    refEnds = refEnds[arg_sort]
    insSeqs = insSeqs[arg_sort]
    
    refPos = 0
    newRef = ""
    for rS, rE, iS in zip(refStarts, refEnds, insSeqs):
        if rS > refPos:
            newRef += ref[refPos:rS]
            refPos = rE
            newRef += iS
    
    newRef += ref[refPos:]
    
    if returnComplex:
        return newRef, refStarts, refEnds, insSeqs
    
    return newRef

def getRefLoc(pos, refStarts, refEnds, insSeqs):
    # gets the true reference position given the outputs of generateMutSeq
    
    fudgeF = 0
    for rS, rE, iS in zip(refStarts, refEnds, insSeqs):
        if pos + fudgeF < rS:
            return pos + fudgeF
        altF = rE - rS + len(iS)
        if altF > 0:
            # net insertion
            if pos + fudgeF - rS < altF:
                # inside insert
                return -1
        fudgeF -= altF
    return pos + fudgeF

def findSNP(seq, ref, delins_hgvs):
    """
    Finds and describes any SNPs of sequence, with respect to reference
    mutated with deletion / insertion hgvs instructions
    """
    
    synRef, refStarts, refEnds, insSeqs = generateMutSeq(ref, delins_hgvs, returnComplex = True)
    aln = aligner2.align(seq, synRef)
    
    q_alns, r_alns = [], []
    coords = aln[-1].coordinates
    minAlignLen = 6
    for j in range(len(coords[0]) - 1):
        if coords[0][j+1] - coords[0][j] >= minAlignLen and coords[1][j+1] - coords[1][j] >= minAlignLen:
            # if match must be a block of at least 6
            q_alns.append(coords[0][j:j+2])
            r_alns.append(coords[1][j:j+2])
            
    q_alns = np.array(q_alns)
    r_alns = np.array(r_alns)

    snpCoords = []
    snpRefs = []
    snpSubs = []
    
    for i in range(len(q_alns)):
        qlen = q_alns[i][1] - q_alns[i][0]
        rlen = r_alns[i][1] - r_alns[i][0]

        qseq = seq[q_alns[i][0]:q_alns[i][1]]
        rseq = synRef[r_alns[i][0]:r_alns[i][1]]
        assert len(qseq) == len(rseq)
            
        r_start = r_alns[i][0]
        for j in range(qlen):
            if qseq[j] != rseq[j]:
                snpCoords.append(r_start + j)
                snpRefs.append(rseq[j])
                snpSubs.append(qseq[j])
    
    # Correct snpCoords
    realCoords, realRefs, realSubs = [], [], []
    for i in range(len(snpCoords)):
        pos = getRefLoc(snpCoords[i], refStarts, refEnds, insSeqs)
        if pos > 0:
            realCoords.append(pos)
            realRefs.append(snpRefs[i])
            realSubs.append(snpSubs[i])

    return realCoords, realRefs, realSubs

def alignITD(prealigns_df, config):
    REF = config["REF"]
    
    start_time = timeit.default_timer()
    print(f"Processing sample {sampleName}")
    
    df = prealigns_df.copy(deep=True)

    allow_insert_mismatch = 3

    df["Aligned"] = False
    df["alignRefCoords"] = ""
    df["HGVS"] = ""
    df["SNP"] = ""
    df["net_insertSize"] = 0
    df["Ns"] = ""
    # df["idealSequence"] = ""
    # df["gaps_and_mismatches"] = -1
    # df["Ns"] = -1
    # df["nonN_gaps_and_mismatches"] = -1

    covIncrement = np.zeros((len(REF) + 1,))
    mutList = []
    mutNames = []

    for i in range(len(df)):
        seq = df.iloc[i]["Sequence"]
        rC_S, ops_S, rSeq, startC, endC = getHGVS(seq, REF)
        if startC == -1 or endC == -1:
            continue

        seqCount = int(df.iloc[i]["Counts"])
            
        covIncrement[startC] += seqCount
        covIncrement[endC] -= seqCount
        
        seqMutList = []
        for rC, ops, rS in zip(rC_S, ops_S, rSeq):
            tmpMut = Mutation(ops, rC, rS, seqCount)
            seqMutList.append(tmpMut)
        
        if any(m.netInsert() > 2 for m in seqMutList):
            for ii in range(len(seqMutList)):
                netInsert = seqMutList[ii].netInsert()
                if netInsert > 2:
                    thisMutName = seqMutList[ii].nameMut()
                    # find highest count mutation with match under threshold
                    mutOpts = [m for m in mutList if (m.netInsert() >= netInsert - 3 and m.netInsert() <= netInsert + 3)]

                    if len(mutOpts) > 0:                                              
                        mutOpts.sort(key=lambda x: x.counts, reverse=True)

                        hasScore = False
                        for mut in mutOpts:
                            if mut.nameMut() == thisMutName:
                                seqMutList[ii] = mut
                                break
                                
                            if not hasScore:
                                # get ideal scoring
                                mutNameList = [m.nameMut() for m in seqMutList]
                                synSeq = generateMutSeq(REF, mutNameList)
                                aln = aligner2.align(seq, synSeq)
                                cts_ideal = aln[-1].counts()
                                hasScore = True
                                
                            tmpHGVS = copy.deepcopy(mutNameList)
                            tmpHGVS[ii] = mut.nameMut()
                            synSeq = generateMutSeq(REF, tmpHGVS)
                            aln = aligner2.align(seq, synSeq)
                            cts = aln[-1].counts()
                            if cts.gaps + cts.mismatches <= cts_ideal.gaps + cts_ideal.mismatches + allow_insert_mismatch:
                                seqMutList[ii] = mut
                                break

        HGVSMutNames = [m.nameMut() for m in seqMutList]
                                                
        # Infer SNPs
        snpNameList = []
        N_List = []

        snpCoords, snpRefs, snpSubs = findSNP(seq, REF, HGVSMutNames)

        for sC, sR, sS in zip(snpCoords, snpRefs, snpSubs):
            if sS == "N":
                N_List.append(sC)
            else:
                snp = f"{sC}{sR}>{sS}"
                snpNameList.append(snp)
                snpMut = Mutation("snp", str(sC), f"{sR}>{sS}", seqCount)
                seqMutList.append(snpMut)
        
        # add mutations to main list
        for ii in range(len(seqMutList)):
            thisMutName = seqMutList[ii].nameMut()
            mutIdx = [i for i, m in enumerate(mutNames) if m == thisMutName]
            assert len(mutIdx) < 2
            if len(mutIdx) == 1:
                mutList[mutIdx[0]].add(seqCount)
            else:
                mutList.append(seqMutList[ii])
                mutNames.append(seqMutList[ii].nameMut())                    
                    
        # determine comutations
        if len(seqMutList) > 1:
            seqMutNames = [m.nameMut() for m in seqMutList]
            for ii in range(len(seqMutNames)):
                mutIdx = [i for i, m in enumerate(mutNames) if m == seqMutNames[ii]]
                assert len(mutIdx) == 1
                for iii in range(len(seqMutNames)):
                    if iii == ii:
                        continue
                    mutList[mutIdx[0]].addComutation(seqMutNames[iii], seqCount)                    
                    
        df.loc[i,"Aligned"] = True
        df.loc[i, "alignRefCoords"] = f"{startC}-{endC}"
        mutNameList = [m.nameMut() for m in seqMutList]
        df.loc[i, "HGVS"] = ";".join(HGVSMutNames)
        df.loc[i, "SNP"] = ";".join(snpNameList)

        # newSeq = generateMutSeq(REF, mutNameList)
        # df.loc[i, "net_insertSize"] = len(newSeq) - len(REF)
        df.loc[i, "net_insertSize"] = sum([int(m.netInsert()) for m in seqMutList])
        
        df.loc[i, "Ns"] = ",".join([str(i) for i in N_List])
        
        # df.loc[i, "idealSequence"] = newSeq
        # aln2 = aligner2.align(df.iloc[i]["Sequence"], newSeq)
        # cts = aln2[-1].counts()
        # df.loc[i, "gaps_and_mismatches"] = cts.gaps + cts.mismatches
        # df.loc[i, "Ns"] = df.iloc[i]["Sequence"].count('N')
        # df.loc[i, "nonN_gaps_and_mismatches"] = cts.gaps + cts.mismatches - df.iloc[i]["Sequence"].count('N')

    ######################################################################
    # Write results
    df.to_csv(config["ALIGN_FILE"], sep = ",", index=False)
    
    mutList.sort(key=lambda x: x.counts, reverse=True)

    mutName = [m.nameMut() for m in mutList]
    mutCount = [m.counts for m in mutList]
    netIns = [m.netInsert() for m in mutList]

    totCov = []
    incCov = 0
    for i in range(len(covIncrement)):
        incCov += covIncrement[i]
        totCov.append(incCov)

    insertPos = []
    insertRegion = []
    mutNorm = []
    mutVaf = []
    coMuts = []
    for i in range(len(mutList)):
        pos = mutList[i].getInsertPos()
        assert pos > -1
        insertPos.append(pos)
        insertRegion.append(anno.iloc[pos]["region"])
        mutNorm.append(totCov[pos])
        mutVaf.append(100 * mutCount[i] / totCov[pos])
        
        cM_dict = {k: v for k, v in sorted(mutList[i].comutations.items(), key=lambda x: x[1], reverse = True)}
        cM_dict_pc = {}
        for k, v in zip(cM_dict.keys(), cM_dict.values()):
            cM_dict_pc[k] = f"{round(v * 100/ mutCount[i], 2)}%"
        coMuts.append(cM_dict_pc)

    dict = {'name': mutName, 'netInsert': netIns, 'counts': mutCount, 'vaf_percent': mutVaf, 'coverage': mutNorm, 
            'insertPos': insertPos, 'insertRegion' : insertRegion, 'co_mutations' : coMuts} 
    res = pd.DataFrame(dict)
    print("Top inserts")
    res_s = res[res["netInsert"] > 5]
    print(res_s.head(10))
    
    ######################################################################
    # Write results    
    res.to_csv(config["MUTATION_FILE"], sep = ",", index=False)
    ######################################################################

    summa = res.groupby(["netInsert"])[["vaf_percent", "counts"]].sum().reset_index()
    summa = summa.sort_values(by = 'vaf_percent', ascending = False)
    print("Top insert lengths")
    print(summa.head(10))
    
    ######################################################################
    # Write results    
    summa.to_csv(config["NETINSERT_FILE"], sep = ",", index=False)
    ######################################################################
    
    print(f"mergeITD time taken - {round(timeit.default_timer() - start_time, 2)} sec")
    print("\n")
    
    return 0

def parse_config_from_cmdline(config):
    """
    Get analysis parameters from commandline.

    Args:
        config (dict): Dict to save parameters to

    Returns:
        Filled config dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("sampleID", help="sample ID used as output folder prefix (REQUIRED)")
    parser.add_argument("fastq1", help="FASTQ file (optionally gzipped) of forward reads (REQUIRED)")
    parser.add_argument("fastq2", help="FASTQ file (optionally gzipped) of reverse reads (optional)", nargs="?")
    parser.add_argument("-bbmap", help="Path to bbmap directory (default ~/bin/bbmap)", default="~/bin/bbmap", type=str)

    parser.add_argument("-reference", help="WT amplicon sequence as reference for read alignment (default ./anno/amplicon.txt)", default="./anno/amplicon.txt", type=str)
    parser.add_argument("-anno", help="WT amplicon sequence annotation (default ./anno/amplicon_kayser.tsv)", default="./anno/amplicon_kayser.tsv", type=str)
    # parser.add_argument("-forward_primer", help="Forward primer gene-specific sequence(s) as present at the 5' end of supplied forward reads. Separate by space when supplying more than one (default GCAATTTAGGTATGAAAGCCAGCTAC)", default=["GCAATTTAGGTATGAAAGCCAGCTAC"], type=str, nargs="+")
    # parser.add_argument("-reverse_primer", help="Reverse primer gene-specific sequence(s) as present at the 5' end of supplied reverse reads. Separate by space when supplying more than one (default CTTTCAGCATTTTGACGGCAACC)", default=["CTTTCAGCATTTTGACGGCAACC"], type=str, nargs="+")
    # parser.add_argument("-require_indel_free_primers", help="If True, discard i) reads containing insertions or deletions within the primer sequence and ii) reads not containing any primer sequence. Set to False if these have been trimmed (default True)", default=True, type=str_to_bool)
    # parser.add_argument("-forward_adapter", help="Sequencing adapter of the forward reads' primer as (potentially) present at the 5' end of the supplied forward reads, 5' of the gene-specific primer sequence (default TCGTCGGCAGCGTCAGATGTGTATAAGAGACAGA)", default="TCGTCGGCAGCGTCAGATGTGTATAAGAGACAGA", type=str)
    # parser.add_argument("-reverse_adapter", help="Sequencing adapter of the reverse reads' primer as (potentially) present at the 5' end of the supplied reverse reads, 5' of the gene-specific primer sequence (default GTCTCGTGGGCTCGGAGATGTGTATAAGAGACAGA)", default="GTCTCGTGGGCTCGGAGATGTGTATAAGAGACAGA", type=str)
    parser.add_argument("-plot_coverage", help="If True, plot read coverage across the reference to 'coverage.png' in the respective output folder (default False)", default=False, type=str_to_bool)
    # parser.add_argument("-technology", help="Sequencing technology used, options are '454' or 'Illumina' (default). '454' sets -infer_sense_from_alignment to True and -min_read_copies to 1, regardless of the respective command line options used; 'Illumina' will instead use these command line options or their respective defaults.", default="Illumina", type=str, choices=['Illumina', '454'])
    # parser.add_argument("-infer_sense_from_alignment", help="If True, infer each read's sense by aligning it as a forward and reverse read and keeping the better alignment (default False).", default=False, type=str_to_bool)
    parser.add_argument('-nkern', help="number of cores to use for parallel tasks (default 12)", default="12", type=int)
    parser.add_argument('-gap_open', help="alignment cost of gap opening (default -36)", default="-36", type=int)
    parser.add_argument('-gap_extend', help="alignment cost of gap extension (default -0.5)", default="-0.5", type=float)
    parser.add_argument('-match', help="alignment cost of base match (default 5)", default="5", type=int)
    parser.add_argument('-mismatch', help="alignment cost of base mismatch (default -15)", default="-15", type=int)
    # parser.add_argument('-max_trailing_bp', help="maximum number of aligned bp between the start / end of an insertion and the start / end of the read to consider the insertion 'trailing'. Trailing insertions are not required to be in-frame and will be considered ITDs even if the matching WT tandem is not directly adjacent. Set this to 0 to disable (default 0).", default="0", type=int)
    # parser.add_argument('-minscore_inserts', help="fraction of max possible alignment score required for ITD detection and insert collapsing (default 0.5)", default="0.5", type=float)
    # parser.add_argument('-minscore_alignments', help="fraction of max possible alignment score required for a read to pass when aligning reads to amplicon reference (default 0.4)", default="0.4", type=float)
    # parser.add_argument("-min_bqs", help="minimum average base quality score (BQS) required by each read (default 30)", type=int, default=30)
    # parser.add_argument('-min_read_length', help="minimum read length in bp required after N-trimming (default 100)", default="100", type=int)
    parser.add_argument('-min_read_copies', help="minimum number of copies of each read required for processing (1 to turn filter off, 2 (default) to discard unique reads)", default="2", type=int)
    parser.add_argument('-min_insert_seq_length', help="minimum number of insert basepairs which must be sequenced of each insert for it to be considered by getITD. For non-trailing ITDs, this is the minimum insert length; for trailing ITDs, it is the minimum number of bp of a potentially longer ITD which have to be sequenced (default 6).", default="6", type=int)
    # parser.add_argument("-max_seq_Ns", help="maximum number of N's before these are filtered prior to alignment", type=int, default=-1)
    parser.add_argument('-filter_ins_unique_reads', help="minimum number of unique reads required to support an insertion for it to be considered 'high confidence' (default 2)", default="2", type=int)
    parser.add_argument('-filter_ins_total_reads', help="minimum number of total reads required to support an insertion for it to be considered 'high confidence' (default 1)", default="1", type=int)
    parser.add_argument('-filter_ins_vaf', help="minimum variant allele frequency (VAF) required for an insertion to be considered 'high confidence' (default 0.006)", default="0.006", type=float)
    cmd_args = parser.parse_args()

    config["R1"] = cmd_args.fastq1
    config["R2"] = cmd_args.fastq2
    config["SAMPLE"] = cmd_args.sampleID
    config["NKERN"] = cmd_args.nkern

    config["REF_FILE"] = cmd_args.reference
    config["ANNO_FILE"] = cmd_args.anno
    
    config["BBMAP_PATH"] = cmd_args.bbmap
    assert os.path.isdir(config["BBMAP_PATH"])
    assert os.path.isfile(f"{config["BBMAP_PATH"]}/bbmerge.sh")
    assert os.path.isfile(f"{config["BBMAP_PATH"]}/bbduk.sh")
    
    # config["TECH"] = cmd_args.technology
    # if config["TECH"] == "454":
        # config["INFER_SENSE_FROM_ALIGNMENT"] = True
    # else:
        # config["INFER_SENSE_FROM_ALIGNMENT"] = cmd_args.infer_sense_from_alignment
    config["PLOT"] = cmd_args.plot_coverage

    # R2 reads are reverse-complemented prior to alignment to the WT reference sequence
    # --> reverse-complement any sequence later to be found within reverse-complemented R2 reads
    # config["FORWARD_PRIMERS"] = [primer.upper() for primer in cmd_args.forward_primer]
    # config["REVERSE_PRIMERS"] = [primer.upper().translate(str.maketrans('ATCGatcg','TAGCtagc'))[::-1] for primer in cmd_args.reverse_primer]
    # config["FORWARD_ADAPTER"] = cmd_args.forward_adapter.upper()
    # config["REVERSE_ADAPTER"] = cmd_args.reverse_adapter.upper().translate(str.maketrans('ATCGatcg','TAGCtagc'))[::-1]

    config["COST_MATCH"] = cmd_args.match
    config["COST_MISMATCH"] = -abs(cmd_args.mismatch)
    config["COST_GAPOPEN"] = -abs(cmd_args.gap_open)
    config["COST_GAPEXTEND"] = -abs(cmd_args.gap_extend)
    # config["MIN_SCORE_INSERTS"] = cmd_args.minscore_inserts
    # config["MIN_SCORE_ALIGNMENTS"] = cmd_args.minscore_alignments

    # config["MIN_BQS"] = cmd_args.min_bqs
    # config["MAX_NS"] = cmd_args.max_seq_Ns
    # config["MIN_READ_LENGTH"] = cmd_args.min_read_length
    config["MIN_INSERT_SEQ_LENGTH"] = cmd_args.min_insert_seq_length
    # if config["TECH"] == "454":
        # config["MIN_READ_COPIES"] = 1
    # else:
    config["MIN_READ_COPIES"] = cmd_args.min_read_copies
    # config["REQUIRE_INDEL_FREE_PRIMERS"] = cmd_args.require_indel_free_primers
    # config["MAX_TRAILING_BP"] = cmd_args.max_trailing_bp

    config["MIN_TOTAL_READS"] = cmd_args.filter_ins_total_reads
    config["MIN_UNIQUE_READS"] = cmd_args.filter_ins_unique_reads
    config["MIN_VAF"] = cmd_args.filter_ins_vaf

    return config

def make_file_path_absolute(file_):
    if not os.path.isabs(file_):
        file_ = os.path.join(os.getcwd(), file_)
    return file_


def save_coverage(iref_coverage, config):
    """
    Write coverage distribution per inter-bp space
    to file `config["OUT_COV_FILE"]` in the `config["OUT_DIR"]` folder.

    Args:
        iref_coverage ([dict]): List oft three dictionaries which each contain
                the inter-bp coverage of the reference for i) forward reads only,
                ii) reverse reads only and iii) all reads, merged at the fragment
                level so that paired reads of the same DNA fragments are not counted
                twice at any given position.
        config (dict): Dictionary containing analysis parameters.
    """
    cov = pd.DataFrame(iref_coverage)
    cov.to_csv(config["OUT_COV_FILE"], sep="\t")


def plot_coverage(iref_coverage, config):
    """
    Plot coverage distribution per inter-bp space
    to file `config["OUT_COV_PLOT"]` in the `config["OUT_DIR"]` folder.

    Args:
        iref_coverage ([dict]): List oft three dictionaries which each contain
                the inter-bp coverage of the reference for i) forward reads only,
                ii) reverse reads only and iii) all reads, merged at the fragment
                level so that paired reads of the same DNA fragments are not counted
                twice at any given position.
        config (dict): Dictionary containing analysis parameters.
    """
    # import only when plotting is desired to avoid depending on matplotlib install?
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')

    fig, axs = plt.subplots(3, figsize=(20, 8), sharex=True, sharey=True)
    fig.suptitle("Final coverage achieved for " + config["SAMPLE"], fontsize=20)

    forward_plot = axs[0].bar(
            iref_coverage["all_reads"].keys(),
            iref_coverage["all_reads"].values(),
            label="total fragments",
            linewidth=0,
            width=1,
            color="dimgray")
    forward_plot = axs[1].bar(
            iref_coverage["forward_reads"].keys(),
            iref_coverage["forward_reads"].values(),
            label="forward reads",
            linewidth=0,
            width=1,
            color="tab:blue")
    forward_plot = axs[2].bar(
            iref_coverage["reverse_reads"].keys(),
            iref_coverage["reverse_reads"].values(),
            label="reverse reads",
            linewidth=0,
            width=1,
            color="tab:orange")

    for ax in axs:
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.legend()

    axs[2].set_xlabel('reference bp', fontsize=18)
    axs[1].set_ylabel('# of reads aligned', fontsize=18)

    plt.tight_layout()
    plt.savefig(config["OUT_COV_PLOT"], dpi=300)


def main(config):

    # PROCESS INPUTS
    config["OUT_DIR"] = '_'.join([config["SAMPLE"], "mergeitd"])
    config["TMP_DIR"] = '_'.join([config["SAMPLE"], "mergeitd", "temp_fastq"])

    config["OUT_COV_PLOT"] = os.path.join(config["OUT_DIR"], "coverage.png")
    config["OUT_COV_FILE"] = os.path.join(config["OUT_DIR"], "coverage.txt")
    config["STATS_FILE"] = os.path.join(config["OUT_DIR"], "stats.txt")
    config["CONFIG_FILE"] = os.path.join(config["OUT_DIR"], "config.txt")

    config["BBLOG"] = "bbmap.log"
    config["ALIGN_FILE"] = "alignClasses.csv"
    config["MUTATION_FILE"] = "mutation_vaf.csv"
    config["NETINSERT_FILE"] = "netInserts_vaf.csv"
    
    # make all input & output file / folder names absolute paths
    for file_ in ["R1", "R2", "REF_FILE", "ANNO_FILE", "OUT_DIR", 
        "OUT_COV_PLOT", "OUT_COV_FILE", "STATS_FILE", "CONFIG_FILE",
        "BBLOG", "ALIGN_FILE", "MUTATION_FILE", "NETINSERT_FILE"
    ]:
        if config[file_]:
            config[file_] = make_file_path_absolute(config[file_])

    config["ANNO"] = read_annotation(config["ANNO_FILE"])
    config["ANNO"] = annotateCoords(config["ANNO"])
    
    # config["DOMAINS"] = get_domains(config["ANNO"])
    config["REF"] = read_reference(config["REF_FILE"]).upper()
    # config["COST_ALIGNED"] = {(c1, c2): get_alignment_score(c1, c2, config) for c1, c2 in itertools.product(["A","T","G","C","Z","N"], repeat=2)}

    ## CREATE OUTPUT FOLDER
    if not os.path.exists(config["OUT_DIR"]):
        os.makedirs(config["OUT_DIR"])

    ## CREATE TEMP DIRECTORY FOR MERGED FASTQ
    if not os.path.exists(config["TMP_DIR"]):
        os.makedirs(config["TMP_DIR"])
        
    ## CHANGE TO OUTPUT FOLDER
    #  this is required for parallel child processes to retrieve
    #  the correct config.txt file later on despite static / constant filename
    os.chdir(config["OUT_DIR"])
    save_config(config, config["CONFIG_FILE"])

    ## REMOVE OLD STATS & LOG FILE & START CREATING A NEW ONE
    try:
        os.remove(config["STATS_FILE"])
        # os.remove(os.path.join(config["OUT_DIR"], "incomplete-wt-tandem.log"))
    except OSError:
        pass
    save_stats("\n==== PROCESSING SAMPLE {} ====".format(config["SAMPLE"]), config["STATS_FILE"])

    ### NEW MERGEITD PIPELINE

    config["ALIGNER"] = Align.PairwiseAligner()

    config["ALIGNER"].mode = 'global'
    config["ALIGNER"].match_score = config["COST_MATCH"]
    config["ALIGNER"].mismatch_score = config["COST_MISMATCH"]
    config["ALIGNER"].open_gap_score = config["COST_GAPOPEN"]
    config["ALIGNER"].extend_gap_score = config["COST_GAPEXTEND"]
    config["ALIGNER"].target_end_gap_score = 0.0
    config["ALIGNER"].query_end_gap_score = 0.0

    bbmap_log = bbmap_process_quick(
        config["R1"], config["R2"], 
        bbmap_path = config["BBMAP_PATH"], 
        temp_path = config["TMP_DIR"])
    
    with open(config["BBLOG"], 'w') as f:
        f.write(bbmap_log)

    ### READS MERGED & CLEANED FASTQ READS
    reads = read_fastq(f"{config["TMP_DIR"]}/cleaned.fastq")

    ### GET UNIQUE READS
    unique_reads = Counter(reads)

    ### MAKE PANDAS DF OF UNIQUE READS AND COUNTS
    prealigns = pd.DataFrame({
        "Sequence": list(Counter(unique_reads).keys()),
        "Counts" : list(Counter(unique_reads).values())
    })
    prealigns = prealigns.sort_values(by = "Counts", ascending = False).reset_index(drop = True)
    prealigns = prealigns[prealigns["Counts"] >= config["MIN_READ_COPIES"]]

    ### MEASURE SEQUENCE LENGTH
    prealigns["SeqLength"] = 0
    for i in range(len(prealigns)):
        prealigns.loc[i, "SeqLength"] = len(prealigns.iloc[i]["Sequence"])

    alignITD(prealigns, config)

    ### END MERGEITD PIPELINE

    ########################################
    # CHANGE BACK TO ORIGINAL / PARENT DIRECTORY
    os.chdir("..")

    shutil.rmtree(config["TMP_DIR"])

########## MAIN ####################
if __name__ == '__main__':

    config = parse_config_from_cmdline(config)
    main(config)
