#!/usr/bin/env python3
import utilities as ut
from pudb import set_trace
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
import os
from net import IDENet
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from multiprocessing import Pool, cpu_count, Manager
import pysam
import time
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search import Repeater
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from mmfsv import list2img
from hyperopt import hp
import torchvision.transforms
from multiprocessing.dummy import Pool as ThreadPool
import argparse
import psutil
import time
from concurrent.futures import ThreadPoolExecutor
import sys


# Set environment, limit thread number
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)


my_label = "7+11channel_predict_5fold"
seed_everything(2024)


# Function to process a single image
def process_single_image(args):
    image_type, idx, b_e = args
    img = ut.to_input_image_single(chromosome_sign[:, b_e[0]:b_e[1]])
    img_mid = ut.to_input_image_single(mid_sign[:, b_e[0]:b_e[1]])
    img_i = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))
    img_i = img_i.reshape(1, -1)
    return image_type, idx, img, img_mid, img_i


# Print system start time
current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f"\n[INFO] F-SV begin time: {current_time}")


# Load arguments
parser = argparse.ArgumentParser(description="F-SV Training")
# Add auguments: data_dir, bam_dir, thread_num, batch_size, GPU_index
parser.add_argument('-D', '--data_dir', type=str, required=True, help="Path to the data directory")
parser.add_argument('-B', '--bam_dir', type=str, required=True, help="Path to the BAM directory")
parser.add_argument('-t', '--thread_num', type=int, default=4, help="Number of threads to use (default: 4)")
parser.add_argument('-bs', '--batch_size', type=int, default=16, help="Batch size for training (default: 16)")
parser.add_argument('-GPU', '--GPU_index', type=str, default=0, help="GPU index used for training (default: 0)")

# Analysis arguments
args = parser.parse_args()

# Get arguments
data_dir = args.data_dir
bam_dir = args.bam_dir
num_processes = args.thread_num
bs = args.batch_size
num_cuda = args.GPU_index

# Print argument information
print("\n>>> Arguments Configuration")
print(f"Data directory: {data_dir}")
print(f"BAM directory: {bam_dir}")
print(f"Thread number: {num_processes}")
print(f"Batch size: {bs}")
print(f"GPU index: {num_cuda}")
print()


# Choose GPU index
os.environ["CUDA_VISIBLE_DEVICES"] = num_cuda


# Test: directions for test
# data_dir = "/home/xzy/Desktop/F-SV/data/"
# data_dir = "/mnt/HHD_16T_1/F-SV/HG002_data/"
# data_dir = "/mnt/HHD_16T_1/F-SV/NA12878_data/"
# bam_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/HG002-PacBio-HiFi-minimap2.sorted.bam"
# bam_dir = "/mnt/HHD_16T_1/Alignment_data/NA12878/sorted_final_merged.bam"


# Print Server configuration
print(">>> Server Configuration")
# Get CPU information
cpu_info = psutil.cpu_count(logical=False)  # get physical core number
logical_cores = psutil.cpu_count(logical=True)  # get logical core number
# Get CPU topological information
cpu_topology = os.popen('lscpu').read()
# Get CPU socket number
num_sockets = int(os.popen("lscpu | grep 'Socket(s)' | awk '{print $2}'").read().strip())
# Count physical cores and threads
num_cores_per_socket = cpu_info // num_sockets
threads_per_socket = logical_cores // num_sockets
# Print CPU configuration
print(f"Server has {num_sockets} socket(s).")
print(f"Each socket has {num_cores_per_socket} physical core(s).")
print(f"Each socket has {threads_per_socket} thread(s).")
print()


# Load vcf files
ins_vcf_filename = data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = data_dir + "delete_result_data.csv.vcf"


all_enforcement_refresh = 0
position_enforcement_refresh = 0
img_enforcement_refresh = 0
sign_enforcement_refresh = 0 # attention
cigar_enforcement_refresh = 0


# load bam file and get information
sam_file = pysam.AlignmentFile(bam_dir, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224


# Test: select chromosome
# selected_chromosomes = ['22']
# selected_chromosomes = ['chr1', 'chr2', 'chr3']
# chr_list, chr_length = zip(*[(chr_name, length) for chr_name, length in zip(chr_list, chr_length) if chr_name in selected_chromosomes])
# Print chromosome information
print(f"[INFO] Number of chromosomes: {len(chr_list)}")
print("[INFO] Chromosome Name and Lengths:")
for chr_name, length in zip(chr_list, chr_length):
    print(f"[INFO] Chromosome: {chr_name}, Length: {length}")
print()


print("<=============== F-SV: Image Generation ===============>")
print(f"[INFO] Image generation begin time: {current_time}")

start_time = time.time()

# Check ins/del/n files
# Function: count VCF file numbers
def count_vcf_entries(vcf_file):
    try:
        with open(vcf_file, 'r') as file:
            return sum(1 for line in file if not line.startswith('#') and line.strip())
    except FileNotFoundError:
        print(f"Error: nof found VCF file '{vcf_file}'。")
        return 0
    except Exception as e:
        print(f"Reda '{vcf_file}' error：{e}")
        return 0

# Count VCF file numbers
del_entries = count_vcf_entries(del_vcf_filename) - 1
ins_entries = count_vcf_entries(ins_vcf_filename) - 1

# File directory
ins_dir = os.path.join(data_dir, 'ins')
del_dir = os.path.join(data_dir, 'del')
n_dir = os.path.join(data_dir, 'n')

# Count ins, del, and n file numbers
ins_files_count = len(os.listdir(ins_dir)) if os.path.exists(ins_dir) else 0
del_files_count = len(os.listdir(del_dir)) if os.path.exists(del_dir) else 0
n_files_count = len(os.listdir(n_dir)) if os.path.exists(n_dir) else 0

# Verify file numbers
ins_valid = (ins_files_count == ins_entries)
del_valid = (del_files_count == del_entries)
n_valid = (n_files_count == 2 * del_entries)
# n_valid = (n_files_count == ins_entries)

# if ins_valid and del_valid and n_valid and not all_enforcement_refresh:
# ins_valid = (os.path.exists(os.path.join(data_dir, 'ins')) and os.listdir(os.path.join(data_dir, 'ins')) and
#              all(os.path.exists(os.path.join(data_dir, 'ins', chr)) for chr in chr_list))
# del_valid = (os.path.exists(os.path.join(data_dir, 'del')) and os.listdir(os.path.join(data_dir, 'del')) and
#              all(os.path.exists(os.path.join(data_dir, 'del', chr)) for chr in chr_list))
# n_valid = (os.path.exists(os.path.join(data_dir, 'n')) and os.listdir(os.path.join(data_dir, 'n')) and
#            all(os.path.exists(os.path.join(data_dir, 'n', chr)) for chr in chr_list))

if ins_valid and del_valid and n_valid and not all_enforcement_refresh:
    print("\n > Images files checked, skip to next process.")
else:
    print("\n > Images files does not exist, check positive_img files.")
    print()
    # Load Left/Rigth SR and RD images
    all_ins_img = torch.empty(0, 3, hight, hight)
    all_del_img = torch.empty(0, 3, hight, hight)
    all_negative_img = torch.empty(0, 3, hight, hight)

    # Load CIGAR images
    all_ins_img_mid = torch.empty(0, 4, hight, hight)
    all_del_img_mid = torch.empty(0, 4, hight, hight)
    all_negative_img_mid = torch.empty(0, 4, hight, hight)

    # Load ins signal
    # all_ins_list = torch.empty(0, 512, 11)
    # all_del_list = torch.empty(0, 512, 11)
    # all_negative_list = torch.empty(0, 512, 11)
    all_ins_list = torch.empty(0, 77)
    all_del_list = torch.empty(0, 77)
    all_negative_list = torch.empty(0, 77)

    for chromosome, chr_len in zip(chr_list, chr_length):
        print("========== deal chromosome " + chromosome + " ==========")
        print("[* chromosome " + chromosome + " position start *]")
        # Check whether position files loaded
        position_files = ['insert', 'delete', 'negative']
        position_path = data_dir + 'position/' + chromosome
        if (os.path.exists(position_path) and
            all(os.path.exists(position_path + '/' + pos_file + '.pt') for pos_file in position_files) and
            not position_enforcement_refresh):
            print(" > position/" + chromosome + "/all_position exist, loading...")
            ins_position = torch.load(position_path + '/insert' + '.pt')
            del_position = torch.load(position_path + '/delete' + '.pt')
            n_position = torch.load(position_path + '/negative' + '.pt')
        else:
            print(" > position/" + chromosome + " not exist, execute position generation...")
            ins_position = []
            del_position = []
            n_position = []
            # insert
            insert_result_data = pd.read_csv(ins_vcf_filename, sep = "\t", index_col=0)
            insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
            row_pos = []
            for index, row in insert_chromosome.iterrows():
                row_pos.append(row["POS"])

            set_pos = set()

            for pos in row_pos:
                set_pos.update(range(pos - 100, pos + 100))

            for pos in row_pos:
                gap = 112
                # positive
                begin = pos - 1 - gap
                end = pos - 1 + gap
                if begin < 0:
                    begin = 0
                if end >= chr_len:
                    end = chr_len - 1

                ins_position.append([begin, end])

            # delete
            delete_result_data = pd.read_csv(del_vcf_filename, sep = "\t", index_col=0)
            delete_chromosome = delete_result_data[delete_result_data["CHROM"] == chromosome]
            row_pos = []
            row_end = []
            for index, row in delete_chromosome.iterrows():
                row_pos.append(row["POS"])
                row_end.append(row["END"])

            for pos in row_pos:
                set_pos.update(range(pos - 100, pos + 100))

            for pos, end in zip(row_pos, row_end):
                gap = int((end - pos) / 4)
                if gap == 0:
                    gap = 1
                # positive
                begin = pos - 1 - gap
                end = end - 1 + gap
                if begin < 0:
                    begin = 0
                if end >= chr_len:
                    end = chr_len - 1

                del_position.append([begin, end])

                #negative
                del_length = end - begin
                for _ in range(2):
                    end = begin
                    while end - begin < del_length / 2 + 1:
                        random_begin = random.randint(1, chr_len)
                        while random_begin in set_pos:
                            random_begin = random.randint(1, chr_len)
                        begin = random_begin - 1 - gap
                        end = begin + del_length
                        if begin < 0:
                            begin = 0
                        if end >= chr_len:
                            end = chr_len - 1

                    n_position.append([begin, end])

            save_path = data_dir + 'position/' + chromosome
            ut.mymkdir(save_path)
            torch.save(ins_position, save_path + '/insert' + '.pt')
            torch.save(del_position, save_path + '/delete' + '.pt')
            torch.save(n_position, save_path + '/negative' + '.pt')
        print("[* chromosome " + chromosome + " position end *]")

        print("[* chromosome " + chromosome + " image start *]")
        # Define file list need to load
        positive_img = ['ins_img', 
                        'del_img', 
                        'negative_img', 
                        'ins_img_mid', 
                        'del_img_mid', 
                        'negative_img_mid', 
                        'ins_img_i', 
                        'del_img_i', 
                        'negative_img_i']

        if (os.path.exists(data_dir + 'image/' + chromosome) and
            all(os.path.exists(data_dir + 'image/' + chromosome + '/' + img_name + '.pt') for img_name in positive_img) and
            not img_enforcement_refresh):
            print(" > image/" + chromosome + "/positive_img exist, check next chromosome...")
        else:
            print(" > image/" + chromosome + "/positive_img not exist, execute image generation...")

            # chromosome_sign, mid_sign, mid_sign_img file
            if os.path.exists(data_dir + "chromosome_sign/" + chromosome + ".pt") and not sign_enforcement_refresh:
                print(" > chromosome_sign/" + chromosome + ".pt exist, loading ...")
                chromosome_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + ".pt")
                mid_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")
                mid_sign_img = torch.load(data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")
            else:
                print(" > chromosome_sign/" + chromosome + ".pt not exist, preprocess chromosome " + chromosome + "...")
                ut.mymkdir(data_dir + "chromosome_sign/")

                # Preprocess chromosome_sign, mid_sign, mid_sign_list
                chromosome_sign, mid_sign, mid_sign_list = ut.preprocess(bam_dir, chromosome, chr_len, data_dir, num_processes) # parallel
                # chromosome_sign, mid_sign, mid_sign_list = ut.preprocess(bam_dir, chromosome, chr_len, data_dir) # serial

                # Save chromosome_sign, mid_sign, mid_sign_list
                print(" > save chromosome_sign, mid_sign, mid_sign_list to files...")
                torch.save(chromosome_sign, data_dir + "chromosome_sign/" + chromosome + ".pt")
                torch.save(mid_sign, data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")
                torch.save(mid_sign_list, data_dir + "chromosome_sign/" + chromosome + "_m(i)d_sign.pt")
                
                # Transform ins signal to images
                print(" > transform ins signal to images...")
                mid_sign_img = torch.tensor(list2img.deal_list(mid_sign_list))

                # Save ins signal images to files
                ut.mymkdir(data_dir + "chromosome_img/")
                torch.save(mid_sign_img, data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")
                print()

            # print("Tensor chromosome_sign shape:", chromosome_sign.shape)
            # print("Tensor mid_sign shape:", mid_sign.shape)
            # print("Tensor mid_sign_img shape:", mid_sign_img.shape)

            print(" > initialize tensors to save image datas...")
            # Initialize tensors to save Left/Rigth SR and RD
            ins_img = torch.empty(len(ins_position), 3, hight, hight)
            del_img = torch.empty(len(del_position), 3, hight, hight)
            negative_img = torch.empty(len(n_position), 3, hight, hight)

            # Initialize tensors to save CIGAR
            ins_img_mid = torch.empty(len(ins_position), 4, hight, hight)
            del_img_mid = torch.empty(len(del_position), 4, hight, hight)
            negative_img_mid = torch.empty(len(n_position), 4, hight, hight)

            # Initialize tensors to save ins signal
            # ins_img_i = torch.empty(len(ins_position), 512, 11)
            # del_img_i = torch.empty(len(del_position), 512, 11)
            # negative_img_i = torch.empty(len(n_position), 512, 11)
            ins_img_i = torch.empty(len(ins_position), 77)
            del_img_i = torch.empty(len(del_position), 77)
            negative_img_i = torch.empty(len(n_position), 77)

            # print("Tensor ins_img shape:", ins_img.shape)
            # print("Tensor ins_img_mid shape:", ins_img_mid.shape)
            # print("Tensor ins_img_i shape:", ins_img_i.shape)

            # resize = torchvision.transforms.Resize([512, 11])
            resize = torchvision.transforms.Resize([7, 11])

            # Generate images
            print("[ images generation start ]")
            print(" > Parallel generate images...")

            # Prepare arguments for all image types
            args_list = []

            for idx, b_e in enumerate(ins_position):
                args_list.append(('ins', idx, b_e))

            for idx, b_e in enumerate(del_position):
                args_list.append(('del', idx, b_e))

            for idx, b_e in enumerate(n_position):
                args_list.append(('neg', idx, b_e))

            # Initialize counters for progress tracking
            total_ins = len(ins_position)
            total_del = len(del_position)
            total_neg = len(n_position)

            ins_count = 0
            del_count = 0
            neg_count = 0

            # Print initial progress lines
            print(f"   => finish(ins_img) {chromosome} {ins_count}/{total_ins}")
            print(f"   => finish(del_img) {chromosome} {del_count}/{total_del}")
            print(f"   => finish(neg_img) {chromosome} {neg_count}/{total_neg}")

            # Use a single multiprocessing Pool and imap_unordered for dynamic results
            with Pool(processes=num_processes) as pool:
                results = pool.imap_unordered(process_single_image, args_list)

                for result in results:
                    image_type, idx, img, img_mid, img_i = result
                    if image_type == 'ins':
                        ins_img[idx] = img
                        ins_img_mid[idx] = img_mid
                        ins_img_i[idx] = img_i
                        ins_count += 1
                        sys.stdout.write(f"\033[3F")  # upper cursor move to first line
                        sys.stdout.write(f"\r   => finish(ins_img) {chromosome} {ins_count}/{total_ins}\033[K\n")
                        sys.stdout.write(f"\033[2E")  # down cursor move to last line
                    elif image_type == 'del':
                        del_img[idx] = img
                        del_img_mid[idx] = img_mid
                        del_img_i[idx] = img_i
                        del_count += 1
                        sys.stdout.write(f"\033[3F\033[1B")  # upper cursor move to second line
                        sys.stdout.write(f"\r   => finish(del_img) {chromosome} {del_count}/{total_del}\033[K\n")
                        sys.stdout.write(f"\033[1E")  # down cursor move to last line
                    elif image_type == 'neg':
                        negative_img[idx] = img
                        negative_img_mid[idx] = img_mid
                        negative_img_i[idx] = img_i
                        neg_count += 1
                        sys.stdout.write(f"\033[3F\033[2B")  # upper cursor move to third line
                        sys.stdout.write(f"\r   => finish(neg_img) {chromosome} {neg_count}/{total_neg}\033[K\n")
                    sys.stdout.flush()

            print(" > create image/ dir and save images...")

            save_path = data_dir + 'image/' + chromosome

            ut.mymkdir(save_path)
 
            # Save Left/Rigth SR and RD tensors 
            torch.save(ins_img, save_path + '/ins_img' + '.pt')
            torch.save(del_img, save_path + '/del_img' + '.pt')
            torch.save(negative_img, save_path + '/negative_img' + '.pt')

            # Save CIGAR tensors 
            torch.save(ins_img_mid, save_path + '/ins_img_mid' + '.pt')
            torch.save(del_img_mid, save_path + '/del_img_mid' + '.pt')
            torch.save(negative_img_mid, save_path + '/negative_img_mid' + '.pt')

            # Save ins signal tensors 
            torch.save(ins_img_i, save_path + '/ins_img_i' + '.pt')
            torch.save(del_img_i, save_path + '/del_img_i' + '.pt')
            torch.save(negative_img_i, save_path + '/negative_img_i' + '.pt')

            print("[ images generation end ]")

        print("[* chromosome " + chromosome + " image end *]")

    end_time = time.time()
    img_generation_time = end_time - start_time
    print(f"\n[INFO] Image generation time: {img_generation_time:.2f} seconds")


    # Merge images and tarits
    print("\n<=============== F-SV: Merge All Images and Traits ===============>")
    print(f"[INFO] Images and traits merge begin time: {current_time}")
    start_time = time.time()

    all_ins_img_path = os.path.join(data_dir, 'all_ins_img.pt')
    all_del_img_path = os.path.join(data_dir, 'all_del_img.pt')
    all_n_img_path = os.path.join(data_dir, 'all_n_img.pt')

    all_ins_list_path = os.path.join(data_dir, 'all_ins_list.pt')
    all_del_list_path = os.path.join(data_dir, 'all_del_list.pt')
    all_negative_list_path = os.path.join(data_dir, 'all_negative_list.pt')

    all_merge_files = [
        all_ins_img_path,
        all_del_img_path,
        all_n_img_path,
        all_ins_list_path,
        all_del_list_path,
        all_negative_list_path
    ]

    if all(os.path.exists(f) for f in all_merge_files):
        print("> All images already merged, skipping.")
    else:
        for chromosome, chr_len in zip(chr_list, chr_length):
            print("\n========== deal chromosome " + chromosome + " ==========")
            print("> loaing " + chromosome + " images and traits files ...")

            save_path = data_dir + 'image/' + chromosome

            ins_img = torch.load(save_path + '/ins_img' + '.pt')
            del_img = torch.load(save_path + '/del_img' + '.pt')
            negative_img = torch.load(save_path + '/negative_img' + '.pt')

            ins_img_mid = torch.load(save_path + '/ins_img_mid' + '.pt')
            del_img_mid = torch.load(save_path + '/del_img_mid' + '.pt')
            negative_img_mid = torch.load(save_path + '/negative_img_mid' + '.pt')

            ins_img_i = torch.load(save_path + '/ins_img_i' + '.pt')
            del_img_i = torch.load(save_path + '/del_img_i' + '.pt')
            negative_img_i = torch.load(save_path + '/negative_img_i' + '.pt')

            print("> merge " + chromosome + " images and traits ...")
            all_ins_img = torch.cat((all_ins_img, ins_img), 0)
            all_del_img = torch.cat((all_del_img, del_img), 0)
            all_negative_img = torch.cat((all_negative_img, negative_img), 0)

            all_ins_img_mid = torch.cat((all_ins_img_mid, ins_img_mid), 0)
            all_del_img_mid = torch.cat((all_del_img_mid, del_img_mid), 0)
            all_negative_img_mid = torch.cat((all_negative_img_mid, negative_img_mid), 0)

            all_ins_list = torch.cat((all_ins_list, ins_img_i), 0)
            all_del_list = torch.cat((all_del_list, del_img_i), 0)
            all_negative_list = torch.cat((all_negative_list, negative_img_i), 0)

            print()

        print("> final images and traits merge and save ...")

        all_ins_img = torch.cat([all_ins_img, all_ins_img_mid], 1) # 3, 4, 3
        all_del_img = torch.cat([all_del_img, all_del_img_mid], 1) # 3, 4, 3
        all_n_img = torch.cat([all_negative_img, all_negative_img_mid], 1)

        torch.save(all_ins_img, data_dir + '/all_ins_img' + '.pt')
        torch.save(all_del_img, data_dir + '/all_del_img' + '.pt')
        torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')

        torch.save(all_ins_list, data_dir + '/all_ins_list' + '.pt')
        torch.save(all_del_list, data_dir + '/all_del_list' + '.pt')
        torch.save(all_negative_list, data_dir + '/all_negative_list' + '.pt')

    end_time = time.time()
    img_merge_time = end_time - start_time
    print(f"\n[INFO] Images and traits merge time: {img_merge_time:.2f} seconds")


# Label images: parallel
print("\n<=============== F-SV: Label Images ===============>")
print(f"[INFO] Images label begin time: {current_time}")
start_time = time.time()

# Check exist label datas
if (os.path.exists(data_dir + "ins/") and os.listdir(data_dir + "ins/") and
    os.path.exists(data_dir + "del/") and os.listdir(data_dir + "del/") and
    os.path.exists(data_dir + "n/") and os.listdir(data_dir + "n/")):
    print("\n > Label images already processed, skipping.\n")
else:
    print("\n > Parallel label ins, del and negative datas...")

    # Create direction to save label datas
    ins_label_dir = os.path.join(data_dir, 'ins')
    del_label_dir = os.path.join(data_dir, 'del')
    n_label_dir = os.path.join(data_dir, 'n')

    os.makedirs(ins_label_dir, exist_ok=True)
    os.makedirs(del_label_dir, exist_ok=True)
    os.makedirs(n_label_dir, exist_ok=True)

    # Calculate sample numbers
    all_ins_list = torch.load(data_dir + '/all_ins_list.pt')
    all_del_list = torch.load(data_dir + '/all_del_list.pt')
    all_negative_list = torch.load(data_dir + '/all_negative_list.pt')

    len_ins = len(all_ins_list)
    len_del = len(all_del_list)
    len_negative = len(all_negative_list)

    # Use Manager to share progress info
    manager = Manager()
    progress = manager.dict({"ins": 0, "del": 0, "n": 0})

    # Define image process void
    def label_images(image_type, progress):
        if image_type == 'ins':
            all_ins_img = torch.load(data_dir + '/all_ins_img.pt')
            for index in range(len_ins):
                image = all_ins_img[index].clone()
                list_data = all_ins_list[index].clone()
                torch.save([{"image": image, "list": list_data}, 2],
                           os.path.join(ins_label_dir, f"{index}.pt"))
                progress["ins"] = index + 1
        elif image_type == 'del':
            all_del_img = torch.load(data_dir + '/all_del_img.pt')
            for index in range(len_del):
                image = all_del_img[index].clone()
                list_data = all_del_list[index].clone()
                torch.save([{"image": image, "list": list_data}, 1],
                           os.path.join(del_label_dir, f"{index}.pt"))
                progress["del"] = index + 1
        elif image_type == 'n':
            all_n_img = torch.load(data_dir + '/all_n_img.pt')
            for index in range(len_negative):
                image = all_n_img[index].clone()
                list_data = all_negative_list[index].clone()
                torch.save([{"image": image, "list": list_data}, 0],
                           os.path.join(n_label_dir, f"{index}.pt"))
                progress["n"] = index + 1

    # Use multi process
    image_types = ['ins', 'del', 'n']

    pool = Pool(processes=3)
    for image_type in image_types:
        pool.apply_async(label_images, args=(image_type, progress))
    
    # Close pool, wait all processes complete
    pool.close()

    # Print progress
    while any([p < len_ins if k == "ins" else p < len_del if k == "del" else p < len_negative 
               for k, p in progress.items()]):
        ins_progress = progress["ins"]
        del_progress = progress["del"]
        n_progress = progress["n"]
        sys.stdout.write(f"\r   => processing ins, index = {ins_progress - 1}/{len_ins - 1} | "
                         f"del, index = {del_progress - 1}/{len_del - 1} | "
                         f"neg, index = {n_progress - 1}/{len_negative - 1}")
        sys.stdout.flush()
    pool.join()

end_time = time.time()
img_label_time = end_time - start_time
print(f"\n\n[INFO] Label time: {img_label_time:.2f} seconds")

# Training
print("\n<=============== F-SV: Training ===============>")

logger = TensorBoardLogger(os.path.join(data_dir, "channel_predict"), name=my_label)

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints_predict/" + my_label,
    filename='{epoch:02d}-{validation_f1:.2f}-{train_mean:.2f}',
    monitor="validation_f1",
    verbose=False,
    save_last=None,
    save_top_k=1,
    mode="max",
    auto_insert_metric_name=True,
    every_n_train_steps=None,
    train_time_interval=None,
    every_n_epochs=None,
    save_on_train_epoch_end=None,
    every_n_val_epochs=None
)

def main_train():
    config = {
        "batch_size": bs,
        "beta1": 0.9,
        "beta2": 0.999,
        "lr": 7.187267009530772e-06,
        "weight_decay": 0.0011614665567890423,
        "model_name": "resnet50",
        "KFold":5,
        "KFold_num":0,
    }

    model = IDENet(data_dir, config)

    trainer = pl.Trainer(
        max_epochs=30,
        gpus=1,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)

def train_tune(config, checkpoint_dir=None, num_epochs=200, num_gpus=1):
    model = IDENet(data_dir, config)
    checkpoint_callback = ModelCheckpoint(
        dirpath="../../checkpoints_predict/" + my_label,
        # filename=str(config["KFold_num"]) + "-" + '{epoch:02d}-{validation_f1:.2f}-{validation_mean:.2f}',
        filename='{epoch:02d}-{validation_f1:.2f}-{validation_mean:.2f}',
        monitor="validation_f1",
        verbose=False,
        save_last=None,
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=True,
        every_n_train_steps=None,
        train_time_interval=None,
        every_n_epochs=None,
        save_on_train_epoch_end=None,
        every_n_val_epochs=None
    )
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)

class MyStopper(tune.Stopper):
    def __init__(self, metric, value, epoch = 1):
        self._metric = metric
        self._value = value
        self._epoch = epoch

    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        return (result["training_iteration"] > self._epoch) and (result[self._metric] < self._value)


    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        return False

def gan_tune(num_samples=-1, num_epochs=30, gpus_per_trial=1):
    config = {
        "batch_size": bs,
        "lr": tune.loguniform(1e-7, 1e-5),
        'weight_decay': tune.uniform(0, 0.001),
        # 'weight_decay': tune.uniform(0, 0.01),
        "beta1": 0.9, # tune.uniform(0.895, 0.905),
        "beta2": 0.999, # tune.uniform(0.9989, 0.9991),
        # "lr": 1e-4,
        # 'weight_decay': tune.uniform(0, 0.01),
        # "beta1": tune.uniform(0.895, 0.905),
        # "beta2": tune.uniform(0.9989, 0.9991),
        # "model_name": "resnet50",
        "model_name": "ViT-B/32",
        "use_kfold": False,  # 设置为 False 以禁用 K-Fold
        # 如果需要启用 K-Fold，可以取消以下注释
        # "KFold":5,
        # "KFold_num":tune.choice([1, 3, 4]),
    }

    bayesopt = HyperOptSearch(metric="validation_f1", mode="max")
    re_search_alg = Repeater(bayesopt, repeat=1)

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
        )

    reporter = CLIReporter(
        metric_columns=['train_loss', "train_f1", 'validation_loss', "validation_f1"]
        )

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
        ),
        local_dir="/home/xzy/Desktop/F-SV/log",
        resources_per_trial={
            "cpu": 1,
            "gpu": 1,
        },
        config=config,
        num_samples=num_samples,
        metric='validation_f1',
        mode='max',
        scheduler=scheduler,
        progress_reporter=reporter,
        resume=False,
        search_alg=re_search_alg,
        max_failures = -1,
        name="5fold" + num_cuda)

# ray.init()
# gan_tune()
