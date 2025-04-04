import sys
import os
import pysam
import torch
import time

def is_left_soft_clipped_read(read):
    if read.cigartuples[0][0] == 4:
        return True
    else:
        return False

def is_right_soft_clipped_read(read):
    if read.cigartuples[-1][0] == 4:
        return True
    else:
        return False

def draw_insertion_single(bam_path, chromosome, pic_length, data_dir, num_processes):
    start_time = time.time()
    print(" > Initialize the tensors...")

    split_read_left = torch.zeros(pic_length, dtype=torch.int)
    split_read_right = torch.zeros(pic_length, dtype=torch.int)
    rd_count = torch.zeros(pic_length, dtype=torch.int)
    conjugate_m = torch.zeros(pic_length, dtype=torch.int)
    conjugate_i = torch.zeros(pic_length, dtype=torch.int)
    conjugate_d = torch.zeros(pic_length, dtype=torch.int)
    conjugate_s = torch.zeros(pic_length, dtype=torch.int)
    conjugate_i_list = [[] for _ in range(pic_length)]

    print("[INFO] Tensor initialize done.")
    print(" > Fetch reads from BAM...")
    sam_file = pysam.AlignmentFile(bam_path, "rb")

    processed_read_count = 0

    for read in sam_file.fetch(chromosome):
        if read.is_unmapped or read.is_secondary:
            continue

        start, end = read.reference_start, read.reference_end

        if is_left_soft_clipped_read(read):
            split_read_left[start:end] += 1
        if is_right_soft_clipped_read(read):
            split_read_right[start:end] += 1

        reference_index = start
        for operation, length in read.cigar:
            if operation in [3, 7, 8]:
                reference_index += length
            elif operation == 0:
                conjugate_m[reference_index:reference_index + length] += 1
                reference_index += length
            elif operation == 1:
                # INS
                if reference_index < pic_length:
                    conjugate_i[reference_index] += length
                    conjugate_i_list[reference_index].append(length)
            elif operation == 4:
                # Soft Clip
                left = max(reference_index - int(length / 2), 0)
                right = min(reference_index + int(length / 2), pic_length)
                conjugate_s[left:right] += 1
            elif operation == 2:
                # DEL
                conjugate_d[reference_index:reference_index + length] += 1
                reference_index += length

        processed_read_count += 1

        if processed_read_count % 1000 == 0:
            sys.stdout.write(f"\r [Single-thread] have processed {processed_read_count} reads. ")
            sys.stdout.flush()

    print(f"\r [Single-thread] have processed {processed_read_count} reads. Done.")

    sam_file.close()

    depth_path = os.path.join(data_dir, "depth", chromosome)
    print(f" > Read depth info from {depth_path} ...")
    with open(depth_path, "r") as f:
        for line in f:
            pos_count = line.strip().split("\t")[1:]  
            rd_count[int(pos_count[0]) - 1] = int(pos_count[1])

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[INFO] Single-thread total time: {total_time:.2f} seconds")

    return (
        torch.cat([split_read_left.unsqueeze(0), split_read_right.unsqueeze(0), rd_count.unsqueeze(0)], 0),
        torch.cat([conjugate_m.unsqueeze(0), conjugate_i.unsqueeze(0), conjugate_d.unsqueeze(0), conjugate_s.unsqueeze(0)], 0),
        conjugate_i_list
    )

def trans2img(bam_path, chromosome, chr_len, data_dir, num_processes):
    print("[*] Start preprocess reads")
    chromosome_sign = draw_insertion_single(bam_path, chromosome, chr_len, data_dir, num_processes)
    print("[*] End preprocess reads")
    return chromosome_sign
