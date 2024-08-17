__version__ = '1.5.16'


import Bio.pairwise2 as bio
import timeit
import collections
import itertools
import datetime
import multiprocessing
import argparse
import pandas as pd
from numpy import inf # to read config["COST_ALIGNED"] from file
import numpy as np
import decimal as dc
dc.getcontext().prec = 5
import pprint
import os
import copy
import gzip


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

def flatten_list(list_):
    """
    Turn list of lists into list of sublists' elements.

    Args:
        list_ (list): List to flatten.

    Returns:
        Flattened list.
    """
    return [item for sublist in list_ for item in sublist]

def get_gaps(seq):
    """
    Extract gap indices from alignment string.

    Args:
        seq (str): Alignment string of one of the aligned
                sequences.

    Returns:
        List of lists, each containing indices of consecutive gaps.

    """
    gap_idxs_sep = []

    if '-' in seq:
        seqn = np.array(list(seq))
        gap_idxs_all = np.where(seqn == '-')[0]
        # x = i,e from enumerate(), lambda return diff between e and i, groupby breaks up
        #   list whenever this difference changes (changes between non-consecutive indices)
        for key, group in itertools.groupby(enumerate(gap_idxs_all), lambda x: x[1]-x[0]):
            gap_idxs_sep.append([e for i,e in group])
        assert np.all(np.concatenate(gap_idxs_sep) == gap_idxs_all)
    return gap_idxs_sep


def get_first_aligned_bp_index(alignment_seq):
    """
    Given an alignment string, return the index of the first aligned,
    i.e. non-gap position (0-indexed!).

    Args:
        alignment_seq (string): String of aligned sequence, consisting of
            gaps ('-') and non-gap characters, such as "HA-LO" or "----ALO".

    Returns:
        Integer, >= 0, indicating the first non-gap character within alignment_seq.
    """
    index_of_first_aligned_bp =  [i for i,bp in enumerate(alignment_seq) if bp != '-'][0]
    return index_of_first_aligned_bp



def average_bqs(bqs):
    """
    Calculate the mean BQS of a given string of quality scores.
    Assumes BQS are in Sanger format, encoded as Phred +33.

    Args:
        bqs (str): String of base quality scores.

    Returns:
        Mean BQS of that string.
    """
    return sum([ord(x) - 33 for x in bqs]) / len(bqs)

def connect_bases(char1, char2):
    """
    For two aligned bases, get connecting symbol.
    Bases on whether bases match, mismatch or are part of a gap.

    Args:
        char1, char2 (str): Two aligned bases.
    """
    if char1 == '-' or char2 == '-':
        return ' '
    if char1 == char2:
        return '|'
    return '.'

def connect_alignment(seq1, seq2):
    """
    For two aligned sequences, get connecting symbols.
    Based on whether bases match, mismatch or are part of a gap.

    Args:
        seq1, seq2 (str): Two aligned sequences.
    """
    return ''.join([connect_bases(char1,char2) for char1,char2 in zip(seq1,seq2)])

def get_number_of_digits(number):
    """
    Count the number of digits in a given number.

    Use this, to print that many less spaces in front of each
    part of the well-formatted read-to-reference alignment.

    Args:
        number (int): Number whose digits to count.

    Returns:
        The number of digits.
    """
    if number == 0:
        return 1
    return int(np.log10(number)) +1

def print_alignment_connection(connection, pre_width, f):
    """
    Print the symbols connecting aligned read and reference, encoding
    match, mismatch and gap at each bp.

    Args:
        connection (str): String of connecting symbols.
        pre_width (int): Number of spaces printed before the alignment to keep everything formatted.
        f (file_object): Output file to contain the alignment.
    """
    f.write(' ' * (pre_width +2))
    f.write(connection)
    f.write('\n')

def print_alignment_seq(seq, seq_coord, pre_width, post_width, f):
    """
    Print part of an alignment.

    Args:
        seq (str): The part of the read's sequence to be printed.
        seq_coord (int): Start coordinate of the part of the alignment printed.
        pre_width (int): Number of spaces printed before the alignment to keep everything formatted.
        post_width (int): Number of spaces printed after it.
        f (file_object): Output file to contain the alignment as a whole.

    Return:
       Start coordinate of the next part of this alignment to be printed.
    """
    seq_coord += int(len(seq) > seq.count('-'))
    f.write(' ' * (pre_width - get_number_of_digits(seq_coord) +1))
    f.write(str(seq_coord) + ' ')
    f.write(seq)
    seq_coord = seq_coord + len(seq) - seq.count('-') -1 +int(len(seq) == seq.count('-'))
    f.write(' ' * (post_width - get_number_of_digits(seq_coord)))
    f.write(str(seq_coord) + '\n')
    return seq_coord


def get_alignment_score(char1,char2, config):
    """
    Calculate the alignment score of two aligned bases.

    When realigning an insert to the WT reference, the actual insert
    is masked by 'Z'. Return the maximum penalty (- np.inf) to probibit
    realignment of the insert to itself.

    Args:
        char1, char2 (str): The two aligned bases.

    Returns:
        Alignment score (float).
    """
    if char1 == char2:
        return config["COST_MATCH"]
    elif char1 == 'Z' or char2 == 'Z':
        return -np.inf
    else:
        return config["COST_MISMATCH"]

def get_min_score(seq1, seq2, min_score):
    """
    For two sequences, calculate the minimum alignment score required
    to pass the respective alignment filter, which is a percentage of the
    maximum possible alignment score between these sequences - and
    therefore dependent on the two aligned sequences' length.

    Args:
        seq1, seq2 (str): The two aligned sequences.

        min_score (float): Proportion of the maximum possible alignment score
                that is required to pass the alignment score filter.

    Returns:
        Minimum required alignment score (float).
    """
    return (min(len(seq1),len(seq2)) * config["COST_MATCH"]) * min_score
    # min score with penalize_trailing_gaps=True:
    #return (min(len(seq1),len(seq2)) * config["COST_MATCH"] + (abs(len(seq1) - len(seq2)) -1) * config["COST_GAPEXTEND"] + config["COST_GAPOPEN"]) * min_score


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
    reads = []
    read_index = 0
    try:
        if is_gz_file(fastq_file):
            open_fct = gzip.open
        else:
            open_fct = open

        with open_fct(fastq_file,'rt') as f:
            line = f.readline()
            while line:
                read_id = line
                read_seq = f.readline().rstrip(os.linesep)
                read_desc = f.readline()
                read_bqs = f.readline().rstrip(os.linesep)
                assert len(read_seq) == len(read_bqs)
                reads.append(Read(seq=read_seq, index=read_index, bqs=read_bqs))
                line = f.readline()
                read_index += 1
    # catch missing file or permissions
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

def merge(inserts, condition, iref_coverage, config):
    """
    Merge insertions describing the same mutation.

    Args:
        inserts ([InsertCollection]): List of insertions to merge.
        condition (str): Encodes condition to determine whether
                insertions do describe the same mutation. One of
                "is-same", "is-similar", "is-close", "is-same_trailing"

    Returns:
        InsertCollection of merged insertions.
    """
    still_need_to_merge = True
    while still_need_to_merge:
        still_need_to_merge = False
        merged = []
        for insert_collection in inserts:
            was_merged = False
            for minsert_collection in merged[::-1]:
                if minsert_collection.should_merge(insert_collection, condition, config):
                    minsert_collection = minsert_collection.merge(insert_collection, iref_coverage)
                    was_merged = True
                    still_need_to_merge = True
                    break
            if not was_merged:
                merged.append(insert_collection)
        inserts = merged
    return merged

def save_to_file(inserts, filename, config):
    """
    Write insertions detected to TSV file.

    Add additional columns with sample ID, actual coordinate of the
    insertion site, allelic ratio (MUT : WT) and counts and alignment
    filenames of each of the distinct supporting reads. If annotation
    was supplied, add that as well.

    Args:
        inserts ([InsertCollection]): List of inserts to save.
        filename (str): Name of the file, will be saved in specified OUT_DIR.
    """
    if inserts:
        to_save = [insert.prep_for_save(config) for insert in inserts]
        if config["ANNO"] is not None:
            for insert in to_save:
                insert = insert.set_insertion_site()
                insert = insert.annotate_domains(config["DOMAINS"])
                cols = ["domains"]
                for to_annotate in ["start", "end", "insertion_site"]:
                    for coord in ["chr13_bp", "transcript_bp", "protein_as"]:
                        insert = insert.annotate(to_annotate, coord, config)
                        cols.append(to_annotate + "_" + coord)
                insert = insert.annotate("insertion_site", "region", config)
                cols.append("insertion_site_domain") # rename in df...
                #cols = ["domains", "start_chr13_bp", "start_transcript_bp", "start_protein_as", "end_chr13_bp", "end_transcript_bp", "end_protein_as", "insertion_site_protein_as"]

        dict_ins = {}
        for key in vars(to_save[0]):
            dict_ins[key] = tuple(vars(insert)[key] for insert in to_save)

        df_ins =  pd.DataFrame(dict_ins)
        df_ins["sample"] = [config["SAMPLE"]] * len(to_save)
        df_ins["ar"] = [vaf_to_ar(insert.vaf) for insert in to_save]
        df_ins["file"] = [[read.al_file for read in insert.reads] for insert in to_save]
        df_ins["counts_unique_each"] = [[read.counts for read in insert.reads] for insert in to_save]
        df_ins["counts_unique_total"] = [len([read.counts for read in insert.reads]) for insert in to_save]

        # sort files and read counts per alignment file based on the number of represented reads
        df_ins["file"]               = [[f2 for c2,f2 in sorted(zip(c,f), reverse=True)] for f,c in zip(df_ins["file"], df_ins["counts_unique_each"])]
        df_ins["counts_unique_each"] = [[c2 for c2,f2 in sorted(zip(c,f), reverse=True)] for f,c in zip(df_ins["file"], df_ins["counts_unique_each"])]

        if 'external_bp' in df_ins:
            cols = ['external_bp'] + cols

        cols = ['sample','length', 'start', 'vaf', 'ar', 'coverage', 'counts', 'trailing', 'seq', 'sense'] + cols + ['file', 'counts_unique_each', 'counts_unique_total']
        df_ins = df_ins.rename(index=str, columns={"insertion_site_region": "insertion_site_domain"})
        df_ins[cols].sort_values(by=['length','start','vaf']).to_csv(os.path.join(config["OUT_DIR"],filename), index=False, float_format='%.2e', sep='\t')

def get_unique_reads(reads):
    """
    Merge reads with identical read sequences.

    Create a Read() object and sum up supporting read counts
    in Read.counts for each Read.seq, keep reads of different
    orientation distinct, i.e. create one Read for each combination
    of Read.seq and Read.sense.

    This is really slow. Come up with something better!
    Would it help to process forward/reverse reads separately?

    Args:
        reads ([Read]): Reads to merge, may share the same Read.seq.

    Returns:
        Merged reads ([Read]), each with a unique Read.seq.

    """
    seqs = [read.seq for read in reads]
    unique_seqs, inverse_indices = np.unique(seqs, return_inverse=True)
    nreads = np.array(reads)
    unique_reads = []
    for inverse_index, seq in enumerate(unique_seqs):
        list_reads = nreads[inverse_indices == inverse_index]
        list_reads_index = [read.index for read in list_reads]
        list_reads_sense = set([read.sense for read in list_reads])
        for sense in list_reads_sense:
            unique_reads.append(
                    Read(
                        seq=seq,
                        sense=sense,
                        bqs=None,
                        counts=len([this_read for this_read in list_reads if this_read.sense == sense]),
                        index=[this_read.index for this_read in list_reads if this_read.sense == sense]))
    return unique_reads


def filter_alignment_score(reads, config):
    """
    Filter reads based on alignment score.

    Keep reads with an alignment score above the specified fraction of
    the max achievable alignment score. Failed alignments' score is None.

    Args:
        reads ([Read]): Reads to filter.
    Returns:
        Passing reads ([Read]).
    """
    reads_filtered = [
        read for read in reads
        if read.al_score is not None and read.al_score >= get_min_score(
                read.seq, config["REF"], config["MIN_SCORE_ALIGNMENTS"])]
    save_stats("Filtering {} / {} low quality alignments with a score < {} % of max".format(
            len(reads) - len(reads_filtered), len(reads), config["MIN_SCORE_ALIGNMENTS"] *100), config["STATS_FILE"])
    return reads_filtered


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


def get_reads(config):
    """
    Read in FASTQ files.

    Args:
        config (dict): Dict containing analysis parameters.

    Returns:
        List of Read objects, one for each read from input FASTQ files.
    """
    save_stats("-- Reading FASTQ files --", config["STATS_FILE"])
    start_time = timeit.default_timer()
    reads = read_fastq(config["R1"])
    # if sense is to be set based on file of origin, set now
    if not config["INFER_SENSE_FROM_ALIGNMENT"]:
        for read in reads:
            read.sense = 1

    # IF IT EXISTS:
    # --> reverse-complement R2 reads so that all reads can be aligned to the same reference
    if config["R2"]:
        reads_rev = read_fastq(config["R2"])
        # if sense is to be set based on file of origin,
        #   rev-complement reads and set sense accordingly
        if not config["INFER_SENSE_FROM_ALIGNMENT"]:
            reads_rev = parallelize(Read.reverse_complement, reads_rev, config["NKERN"])
            for read in reads_rev:
                read.sense = -1
        reads = reads + reads_rev
    print("Reading FASTQ files took {} s".format(round(timeit.default_timer() - start_time, 2)))
    save_stats("Number of total reads: {}".format(len(reads)), config["STATS_FILE"])
    return reads


def get_merged_inserts(inserts, type_, iref_coverage, config):
    """
    Merge Inserts based on different conditions.

    Args:
        inserts: List of Inserts to merge.
        type_ (str): "insertions" or "itds",
                     description for output files.
        config (dict): Dict containing analysis parameters.

    Returns:
        List of merged Inserts.
    """
    save_stats("\n-- Merging {} --".format(type_), config["STATS_FILE"])

    merged = []
    # turn Insert objects into InsertCollection to keep merging methods simple and not have to distinguish between the two
    to_merge = [InsertCollection(insert) for insert in inserts]
    suffix = ""
    for condition in [
            "is-same",
            "is-similar",
            "is-close",
            "is-same_trailing"]:
        to_merge = merge(to_merge, condition, iref_coverage, config)
        merged.append(to_merge)
        save_stats("{} {} remain after merging".format(len(to_merge), type_), config["STATS_FILE"])
        suffix = suffix + condition
        save_to_file([insert.rep for insert in to_merge], type_ + "_collapsed-" + suffix + ".tsv", config)
        suffix = suffix + "_"

    # convert InsertCollection back to list of (representative) Inserts
    return [insert.rep for insert in merged[-1]]


def get_hc_inserts(inserts, type_, config, suffix=""):
    """
    Filter for high-confidence (hc) inserts only.

    Args:
        inserts: List of Insertion() objects to filter

        type_ (str): Description of type of inserts,
                     "insertions" or "itds",
                     for output files.

        config (dict): Dict containing analysis parameters.

    Returns:
        Dict with hc inserts, same format as input dict
    """
    save_stats("\n-- Filtering {} --".format(type_), config["STATS_FILE"])

    filter_dic = {
        "number of unique supporting reads": Insert.filter_unique_supp_reads,
        "number of total supporting reads": Insert.filter_total_supp_reads,
        "vaf": Insert.filter_vaf}

    filtered = copy.deepcopy(inserts)
    for filter_type, filter_ in filter_dic.items():
        passed = [filter_(insert, config) for insert in filtered]
        filtered = [insert for (insert, pass_) in zip(filtered, passed) if pass_]

        save_stats("Filtered {} / {} {} based on the {}".format(
                len(passed) - sum(passed), len(passed), type_, filter_type), config["STATS_FILE"])
    save_stats("{} {} remain after filtering!".format(len(filtered), type_), config["STATS_FILE"])
    save_to_file(filtered, type_ + suffix + ".tsv", config)

    return filtered


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

    config["ALIGN_FILE"] = "alignClasses.csv"
    config["MUTATION_FILE"] = "mutation_vaf.csv"
    config["NETINSERT_FILE"] = "netInserts_vaf.csv"
    
    # make all input & output file / folder names absolute paths
    for file_ in ["R1", "R2", "REF_FILE", "ANNO_FILE", "OUT_DIR", 
        "OUT_COV_PLOT", "OUT_COV_FILE", "STATS_FILE", "CONFIG_FILE",
        "ALIGN_FILE", "MUTATION_FILE", "NETINSERT_FILE"
    ]:
        if config[file_]:
            config[file_] = make_file_path_absolute(config[file_])

    config["ANNO"] = read_annotation(config["ANNO_FILE"])
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

    ### END MERGEITD PIPELINE

    ########################################
    # CHANGE BACK TO ORIGINAL / PARENT DIRECTORY
    os.chdir("..")



########## MAIN ####################
if __name__ == '__main__':

    config = parse_config_from_cmdline(config)
    main(config)
