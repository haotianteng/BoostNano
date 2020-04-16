import h5py
import os
import shutil
import mappy
import re
import argparse
import numpy as np
import subprocess
import shlex
import sys
from multiprocessing import Pool
import tqdm
DATA_FORMAT = np.dtype([('raw','<i2'),
                        ('norm_raw','<f8'),
                        ('norm_trans','<f8'),
                        ('start','<i4'),
                        ('length','<i4'),
                        ('base','S1')]) 
BASECALL_ENTRY = '/Analyses/Basecall_1D_000'
RESQUIGGLE_METHODS = {'raw','cwdtw'}
class RUN_RECORD():
    def __init__(self):
        self.fail_align = []
        self.poor_qc = []
        self.sucess = []

def fast5s_iter(dest_link,tsv_table):
    """
    An iterator iterate over the fast5 files.
    Args:
        dest_link: readname -> fast5 path
        tsv_table: Polya segmentation information.
    Return:
        An interator of (start_position, transcript start position)
    """
    if args.eval:
        accept_tags = [b'PASS',b'SUFFCLIP',b'ADAPTER']
    else:
        accept_tags = [b'PASS']
    for idx,tag in enumerate(tsv_table['qc_tag']):
        if tag in accept_tags:
            yield dest_link[tsv_table['readname'][idx]],tsv_table['transcript_start'][idx]

def extract_fastq(input_f,ref_f,mode = 0,trans_start = None,alignment = True):
    """
    Args:
        input_f: intput fast5 file handle
        ref_f: file name of the reference
        mode: 0-dna, 1-rna, -1-rna 180mV
        trans_start: Start position of the transcription(required in RNA mode).
        alignment: If requrie alignment.
    """
    with h5py.File(input_f,'r') as input_fh:
        raw_entry = list(input_fh['/Raw/Reads'].values())[0]
        raw_signal = raw_entry['Signal'].value
        raw_seq = input_fh[BASECALL_ENTRY+'/BaseCalled_template/Fastq'].value
        event = input_fh[BASECALL_ENTRY+'/BaseCalled_template/Events'].value
        align = None
        ref_seq = None
        if alignment:
            ref = mappy.Aligner(ref_f,preset = "map-ont",best_n = 5)
            aligns = ref.map(raw_seq.split(b'\n')[1])
            maxmapq = -np.inf
            for aln in aligns:
                if aln.mapq > maxmapq:
                    maxmapq = aln.mapq
                    align = aln
            if align is None:
                print("FAIL MAPPING "+input_f)
            else:
                if align.strand == -1:
                    ref_seq = mappy.revcomp(ref.seq(align.ctg,start = align.r_st,end = align.r_en))
                else:
                    ref_seq = ref.seq(align.ctg,start = align.r_st,end = align.r_en)
        if (mode == 1) or (mode == -1):
            raw_signal = raw_signal[::-1]
    if ref_seq is None and alignment:
        print("No Reference sequence found in %s"%(input_f))
    return raw_signal,raw_seq,ref_seq,event

def write_output(prefix,raw_signal,ref_seq):
    signal_fn = prefix+'.signal'
    ref_fn = prefix+'.ref'
    with open(signal_fn,'w+') as sig_f:
        sig_f.write('\n'.join([str(sig) for sig in raw_signal]))
    with open(ref_fn,'w+') as ref_f:
        ref_f.write(">"+ os.path.basename(prefix)+'\n')
        ref_f.write(ref_seq)
    input_command = " -i "+ref_fn+ ' -p '+ signal_fn + ' -o ' + prefix+'.aln'
    return input_command

def parse_cwDTW(f_path):
    """
    f_path: file path of the cwDTW output file.
    """
    segs = list()
    with open(f_path,'r') as out_f:
        for line in out_f:
            split_line = re.split('\s+\|*[diff:]*\s*',line.strip())
            segs.append(split_line)
    segs = np.array(segs)
    _,index = np.unique(segs[:,3],return_index = True)
    index = np.sort(index)
    segs = segs[index,:]
    output = list()
    for idx,seg in enumerate(segs[:-1]):
        current = seg[[0,4,5]].tolist()
        current.append(int(seg[2])-1)
        current.append(int(segs[idx+1,2])-int(seg[2]))
        current.append(seg[7][2])
        output.append(tuple(current))
    return np.array(output,dtype = DATA_FORMAT)

def write_back(fast5_f,aln_matrix,raw,ref,resquiggle_method):
    """
    Args:
        fast5_f: handle of the fast5 files.
        aln_matrix: A data matrix contain the resquiggle information.
        raw: raw fastq.
        ref: Aligned reference fasta/q file.
        resquiggle_method: The resquiggle method.
    """
    with h5py.File(fast5_f,'a') as fast5_fh:
        if resquiggle_method == 'cwdtw':
            data = np.asarray(aln_matrix,dtype = DATA_FORMAT)
            d_format = DATA_FORMAT
        elif resquiggle_method == 'raw':
            data = aln_matrix
            d_format = data.dtype
        if '/Analyses/Corrected_000' in fast5_fh:
            del fast5_fh['/Analyses/Corrected_000']
        event_h = fast5_fh.create_dataset('/Analyses/Corrected_000/BaseCalled_template/Events', shape = (len(data),),maxshape=(None,),dtype = d_format)
        fastq_h = fast5_fh.create_dataset('/Analyses/Corrected_000/BaseCalled_template/Fastq',shape = (),dtype = h5py.special_dtype(vlen=str))
        ref_h = fast5_fh.create_dataset('/Analyses/Corrected_000/BaseCalled_template/Reference',shape = (),dtype = h5py.special_dtype(vlen=str))
        event_h[...] = data
        event_h.attrs['read_start_rel_to_raw'] = 0
        fastq_h[...] = raw
        ref_h[...] = ref
        
def copy_raw(src_fast5,dest_fast5,raw):
    """
    Write the clipped raw signal into fast5 file.
    Args:
        src_fast5: original fast5 file.
        dest_fast5: destination fast5 file.
        raw: The decapped raw signal.
    """
    if args.mode != 0:
        raw = raw[::-1]
    with h5py.File(src_fast5,'r') as root:
     with h5py.File(dest_fast5,'a') as w_root:
        if '/Raw' in w_root:
            del w_root['/Raw']
        if '/UniqueGlobalKey' in w_root:
            del w_root['/UniqueGlobalKey']
        raw_attrs = list(root['/Raw/Reads'].values())[0].attrs
        read_number = raw_attrs['read_number']
        raw_h = w_root.create_dataset('/Raw/Reads/Read_'+ str(read_number)+'/Signal',shape = (len(raw),),dtype = np.int16)
        raw_h[...] = raw
        w_attrs = list(w_root['/Raw/Reads'].values())[0].attrs
        w_attrs.create('duration',data = len(raw),dtype = np.uint32)
        w_attrs.create('median_before',data = raw_attrs['median_before'],dtype = np.float64)
        w_attrs.create('read_id',data = raw_attrs['read_id'])
        w_attrs.create('read_number',data = raw_attrs['read_number'],dtype = np.uint32)
        w_attrs.create('start_mux',data = raw_attrs['start_mux'])
        w_attrs.create('start_time',data = raw_attrs['start_time'])
        h5py.h5o.copy(root.id,b'UniqueGlobalKey',w_root.id,b'UniqueGlobalKey')
    return None

def label(abs_fast5):
    trans_start = abs_fast5[1]
    abs_fast5 = abs_fast5[0]
    if abs_fast5.endswith("fast5"):
        filename = os.path.basename(abs_fast5)
        align = True
        if args.resquiggle_method == 'raw':
            align = False
        raw_signal,raw_seq,ref_seq,decap_event = extract_fastq(abs_fast5,args.ref,args.mode,trans_start,align)
        prefix = os.path.join(args.saving,'resquiggle',os.path.splitext(filename)[0])
        fast5_save = os.path.join(args.saving,'fast5s',filename)
        if args.copy_original:
            shutil.copyfile(abs_fast5,fast5_save)
        else:
            copy_raw(abs_fast5,fast5_save,raw_signal)
        
        ######Begin cwDTW pipeline
        if args.resquiggle_method == 'cwdtw':
            if ref_seq is not None:
                input_cmd = write_output(prefix,raw_signal,ref_seq)
                cmd = os.path.dirname(os.path.realpath(__file__))+"/utils/cwDTW_nano " + input_cmd +' -R ' + str(args.mode)
                print(cmd)
                raise ValueError
                args_cmd = shlex.split(cmd)
                p = subprocess.Popen(args_cmd,stdout = subprocess.PIPE,stderr = subprocess.STDOUT)
                p_out,_ = p.communicate()
                p.wait()
                align_matrix = parse_cwDTW(prefix+'.aln')
            else:
                pass
#                return
        elif args.resquiggle_method == 'raw':
            align_matrix = decap_event
        ######End cwDTW pipeline
        write_back(fast5_save,align_matrix,raw_seq,ref_seq,args.resquiggle_method)

def run():
#    pool = Pool(args.thread)
    filelist = []
    for path , _ , files in os.walk(args.input):
        for file in files:
            if file.endswith('fast5'):
                filelist.append((os.path.join(path,file),None))
    for file in filelist:
        label(file)
#    for _ in tqdm.tqdm(pool.imap_unordered(label,filelist),total = len(filelist)):
#        pass
#    pool.close()
#    pool.join()          
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='boostnano',
                                     description='A preprocesser for Nanopore RNA basecall and RNA model training.')
    parser.add_argument('-i', '--input', required = True,
                        help="Directory of the fast5 files.")
    parser.add_argument('-r', '--ref', required = True,
                        help="Reference file name")
    parser.add_argument('-m','--mode',default = 0,type = int,
                        help="If RNA pore model is used, 0 for DNA pore model, 1 for 200mV RNA pore model, -1 for 180mV RNA pore model, DEFAULT is 0.")
    parser.add_argument('-s','--saving',
                        help="Output saving folder.")
    parser.add_argument('-t','--thread',default = 1,type = int,
                        help="Thread number.")
    parser.add_argument('--copy',dest = 'copy_original',action = 'store_true', 
                        help="If set, copy the original file else create a new fast5 file with raw_signal and resquiggle only.")
    parser.add_argument('--resquiggle_method',default = 'raw',choices = RESQUIGGLE_METHODS, 
                        help="Resquiggle method, can be only chosen from %s"%(RESQUIGGLE_METHODS))
    parser.add_argument('--for_eval',dest = 'eval',action = 'store_true',
                        help="If set, the SUFFCLIP and ADAPTER reads will also be included.")
    args = parser.parse_args(sys.argv[1:])
    
    if not os.path.isdir(args.saving):
        os.mkdir(args.saving)
    fast5_dir = os.path.join(args.saving,'fast5s')
    resquiggle_dir = os.path.join(args.saving,'resquiggle')
    if not os.path.isdir(fast5_dir):
        os.mkdir(fast5_dir)
    if not os.path.isdir(resquiggle_dir):
        os.mkdir(resquiggle_dir)
    run()

    
