import h5py
import os
import shutil
import re
import argparse
import numpy as np
import subprocess
import shlex
import sys
import queue
import time
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing import Queue
from boostnano.utils.sam_op import Aligner
from boostnano.utils.sam_op import save_aligner
from boostnano.utils.sam_op import load_aligner
from boostnano.utils.progress import multi_pbars

DATA_FORMAT = np.dtype([('raw','<i2'),
                        ('norm_raw','<f8'),
                        ('norm_trans','<f8'),
                        ('start','<i4'),
                        ('length','<i4'),
                        ('base','S1')]) 
BASECALL_ENTRY = '/Analyses/Basecall_1D_000'
RESQUIGGLE_METHODS = {'raw','cwdtw'}
FAIL_ALIGN = "Fail aligning to reference"
POOR_QUALITY = "Poor quality reads"
SUCCESS = "Suceed reads"
QUEUE_WAITING_TIME=2
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

def extract_fastq(input_f):
    """
    Args:
        input_f: intput fast5 file handle
    """
    with h5py.File(input_f,'r') as input_fh:
        raw_entry = list(input_fh['/Raw/Reads'].values())[0]
        read_id = raw_entry.attrs['read_id'].decode('utf-8')
        raw_signal = raw_entry['Signal'][()]
        raw_seq = input_fh[BASECALL_ENTRY+'/BaseCalled_template/Fastq'][()]
        event = input_fh[BASECALL_ENTRY+'/BaseCalled_template/Events'][()]
    return raw_signal,raw_seq,read_id,event

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

def create_aligner(samfile,reference,save_f):
    print("Read reference genome and create hash table.")
    aln = Aligner(reference)
    print("Parse sam file.")
    aln.parse_sam(samfile)
    save_aligner(aln,save_f)
    return None

def label_worker(file_queue,aligner,run_record,p_id):
    while True:
        try:
            abs_fast5 = file_queue.get(timeout = QUEUE_WAITING_TIME)
        except queue.Empty:
            print("No file in the queue, worker %d shut down."%(p_id))
            return None
        if abs_fast5.endswith("fast5"):
            filename = os.path.basename(abs_fast5)
            raw_signal,raw_seq,read_id,decap_event = extract_fastq(abs_fast5)
            prefix = os.path.join(args.saving,'resquiggle',os.path.splitext(filename)[0])
            fast5_save = os.path.join(args.saving,'fast5s',filename)
            if args.copy_original:
                shutil.copyfile(abs_fast5,fast5_save)
            else:
                copy_raw(abs_fast5,fast5_save,raw_signal)
            
            ######Begin cwDTW pipeline
            if args.resquiggle_method == 'cwdtw':
                align_ids = aligner.aln_dict.keys()
                if read_id not in align_ids:
                    run_record[FAIL_ALIGN] = run_record[FAIL_ALIGN] + [read_id]
                    continue
                ref_seqs = aligner.aln_dict[read_id]
                ref_seq = max(ref_seqs,key = lambda x: x['map_score'])
                accum_step = np.pad(np.cumsum(decap_event['move']),(1,0),'constant',constant_values = (0,0))
                accum_loc = np.pad(np.cumsum(decap_event['length']),(1,0),'constant',constant_values = (0,0))
                signal_start = accum_loc[accum_step>=ref_seq['query_offset'][0]][0]
                signal_end = accum_loc[accum_step<=ref_seq['query_offset'][1]][-1]
                decap_signal = raw_signal[signal_start:signal_end]
                if args.mode == 1 or args.mode == -1:
                    decap_signal = decap_signal[::-1]
                ref_seq = ref_seq['reference_sequence']
                if ref_seq is not None:
                    input_cmd = write_output(prefix,decap_signal,ref_seq)
                    cmd = os.path.dirname(os.path.realpath(__file__))+"/utils/cwDTW_nano " + input_cmd +' -R ' + str(args.mode)
                    args_cmd = shlex.split(cmd)
                    with subprocess.Popen(args_cmd,
                                          stdout = subprocess.PIPE,
                                          stderr = subprocess.STDOUT,
                                          close_fds = True) as p:
                        p_out,_ = p.communicate()
                        p.stdout.close()
                        p.terminate()
                    align_matrix = parse_cwDTW(prefix+'.aln')
                else:
                    run_record[FAIL_ALIGN] = run_record[FAIL_ALIGN] + [read_id]
                    continue
            elif args.resquiggle_method == 'raw':
                align_matrix = decap_event
                ref_seq = 'aaa'
            ######End cwDTW pipeline
            write_back(fast5_save,align_matrix,raw_seq,ref_seq,args.resquiggle_method)
            run_record[SUCCESS] = run_record[SUCCESS] + [read_id]

def run():
    print("Create a aligner based on reference genomeand SAM file.")
    aligner_f = os.path.join(args.saving,"aln.bin")
#    create_aligner(args.samfile,args.reference,aligner_f)
    aligner = load_aligner(aligner_f)
    print("Aligner craeated successfully and is stored in %s"%(aligner_f))
    
    print("Run resquiggle.")
    file_queue = Queue()
    file_list =[]
    all_proc = []
    manager = Manager()
    log_dict = manager.dict()
    log_dict[FAIL_ALIGN] = []
    log_dict[POOR_QUALITY] = []
    log_dict[SUCCESS] = []
    titles = [FAIL_ALIGN,POOR_QUALITY,SUCCESS,'Total']
    pbars = multi_pbars(titles)
    print(titles)
    for path , _ , files in os.walk(args.input):
        for file in files:
            if file.endswith('fast5'):
                file_queue.put(os.path.join(path,file))
                file_list.append(file)
#    ## Single thread test code###
#    for file in filelist:
#        label(file,aligner_f,log_dict,sema)
#    ## test code end ###
    
    ### Multiple threads running code ###
    for i in range(args.thread):
        p = Process(target = label_worker, args = (file_queue,aligner,log_dict,i))
        all_proc.append(p)
        p.start()
    while any([x.is_alive() for x in all_proc]):
        time.sleep(0.1)
        total_finish = 0
        for key in log_dict.keys():
            total_finish += len(log_dict[key])
        for key in log_dict.keys():
            pbars.update(titles.index(key),len(log_dict[key]),total_finish)
        pbars.update(titles.index('Total'),total_finish,len(file_list))
        pbars.refresh()
    for p in all_proc:
        p.join()
    for key in log_dict.keys():
        print("%s:%d"%(key,len(log_dict[key])))
        
    ### running code end ###
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='boostnano',
                                     description='A preprocesser for Nanopore RNA basecall and RNA model training.')
    parser.add_argument('-i', '--input', required = True,
                        help="Directory of the fast5 files.")
    parser.add_argument('-r', '--reference', required = True,
                        help="Reference file name")
    parser.add_argument('-a', '--samfile', required = True,
                        help="Alignment sam file that contain all the sequences.")    
    parser.add_argument('-m','--mode',default = 0,type = int,
                        help="If RNA pore model is used, 0 for DNA pore model, 1 for 200mV RNA pore model, -1 for 180mV RNA pore model, DEFAULT is 0.")
    parser.add_argument('-s','--saving',
                        help="Output saving folder.")
    parser.add_argument('-t','--thread',default = 0,type = int,
                        help="Thread number, default is 0, which uses all avamp.cpu_count()ilable cores.")
    parser.add_argument('--copy',dest = 'copy_original',action = 'store_true', 
                        help="If set, copy the original file else create a new fast5 file with raw_signal and resquiggle only.")
    parser.add_argument('--resquiggle_method',default = 'raw',choices = RESQUIGGLE_METHODS, 
                        help="Resquiggle method, can be only chosen from %s"%(RESQUIGGLE_METHODS))
    parser.add_argument('--for_eval',dest = 'eval',action = 'store_true',
                        help="If set, the SUFFCLIP and ADAPTER reads will also be included.")
    args = parser.parse_args(sys.argv[1:])
    if args.thread == 0:
        args.thread = mp.cpu_count()
    if not os.path.isdir(args.saving):
        os.mkdir(args.saving)
    fast5_dir = os.path.join(args.saving,'fast5s')
    resquiggle_dir = os.path.join(args.saving,'resquiggle')
    if not os.path.isdir(fast5_dir):
        os.mkdir(fast5_dir)
    if not os.path.isdir(resquiggle_dir):
        os.mkdir(resquiggle_dir)
    run()

    
