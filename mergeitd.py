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
