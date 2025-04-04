import sys
import os
import numpy as np
import pysam
import torch
from multiprocessing import Pool, Manager, Lock
import time
import subprocess
import time

def get_total_reads_samtools(bam_path, chromosome, num_threads):
    """
    使用 samtools 获取特定染色体的总 reads 数量。

    参数：
    - bam_path: BAM 文件路径
    - chromosome: 要统计的染色体名称
    - num_threads: 使用的线程数量

    返回值：
    - total_reads: 特定染色体中的总 reads 数量
    """
    try:
        # samtools view -@ <num_threads> -c <bam_path> <chromosome>
        cmd = ["samtools", "view", "-@", str(num_threads), "-c", bam_path, chromosome]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        # 检查命令是否成功
        if result.returncode == 0:
            return int(result.stdout.strip())
        else:
            # 如果命令失败，抛出错误并显示错误信息
            raise RuntimeError(f"Error counting reads with samtools: {result.stderr}")
    except FileNotFoundError:
        raise RuntimeError("samtools command not found. Make sure samtools is installed and in your PATH.")


def is_left_soft_clipped_read(read):
    """
    判断 read 是否为左端软剪切

    参数：
    - read: pysam 的 AlignmentSegment 对象

    返回值：
    - True 或 False
    """
    if read.cigartuples[0][0] == 4:
        return True
    else:
        return False

def is_right_soft_clipped_read(read):
    """
    判断 read 是否为右端软剪切

    参数：
    - read: pysam 的 AlignmentSegment 对象

    返回值：
    - True 或 False
    """
    if read.cigartuples[-1][0] == 4:
        return True
    else:
        return False

def process_chunk(args):
    """
    处理分配给此进程的reads，并定期更新进度。
    """
    process_id, bam_path, chromesome, pic_length, progress_dict, num_processes, lock = args
    # 打开 BAM 文件
    sam_file = pysam.AlignmentFile(bam_path, "rb")

    # 初始化变量
    split_read_left_chunk = torch.zeros(pic_length, dtype=torch.int)
    split_read_right_chunk = torch.zeros(pic_length, dtype=torch.int)
    conjugate_m_chunk = torch.zeros(pic_length, dtype=torch.int)
    conjugate_i_chunk = torch.zeros(pic_length, dtype=torch.int)
    conjugate_d_chunk = torch.zeros(pic_length, dtype=torch.int)
    conjugate_s_chunk = torch.zeros(pic_length, dtype=torch.int)
    conjugate_i_list_chunk = [[] for _ in range(pic_length)]

    processed_read_count = 0  # 此进程处理的 reads 数量
    read_count = 0  # 全局reads计数

    # 处理所有reads，按轮询方式分配
    # for read in sam_file.fetch(until_eof=True):
    for read in sam_file.fetch(chromesome):
        if read.is_unmapped or read.is_secondary:
            continue

        # 按轮询方式分配reads
        if read_count % num_processes != process_id:
            read_count += 1
            continue

        start, end = read.reference_start, read.reference_end

        if is_left_soft_clipped_read(read):
            split_read_left_chunk[start:end] += 1

        if is_right_soft_clipped_read(read):
            split_read_right_chunk[start:end] += 1

        reference_index = start  # 参考序列中的位置
        for operation, length in read.cigar:  # 解析 CIGAR 字符串
            if operation in [3, 7, 8]:
                reference_index += length
            elif operation == 0:
                conjugate_m_chunk[reference_index:reference_index + length] += 1
                reference_index += length
            elif operation == 1:
                if reference_index < pic_length:
                    conjugate_i_chunk[reference_index] += length
                    conjugate_i_list_chunk[reference_index].append(length)
            elif operation == 4:
                left = max(reference_index - int(length / 2), 0)
                right = min(reference_index + int(length / 2), pic_length)
                conjugate_s_chunk[left:right] += 1
            elif operation == 2:
                conjugate_d_chunk[reference_index:reference_index + length] += 1
                reference_index += length

        processed_read_count += 1
        read_count += 1

        # 每处理 1000 个 reads 更新一次进度
        if processed_read_count % 1000 == 0:
            progress_dict[process_id] = processed_read_count
            with lock:
                # 更新进度显示
                sys.stdout.write(f"\033[{num_processes}A")  # 上移光标
                for i in range(num_processes):
                    if i == process_id:
                        sys.stdout.write(f"\r [Process #{process_id}] have processed {processed_read_count} reads. \n")
                    else:
                        sys.stdout.write("\n")
                sys.stdout.flush()

    # 最终进度更新
    progress_dict[process_id] = processed_read_count
    with lock:
        sys.stdout.write(f"\033[{num_processes}A")
        for i in range(num_processes):
            if i == process_id:
                sys.stdout.write(f"\r [Process #{process_id}] have processed {processed_read_count} reads. Done. \n")
            else:
                sys.stdout.write("\n")
        sys.stdout.flush()

    sam_file.close()
    # 返回此进程的处理结果
    # TODO: 这个return速度有点慢
    return (split_read_left_chunk, split_read_right_chunk, conjugate_m_chunk, conjugate_i_chunk, conjugate_d_chunk, conjugate_s_chunk, conjugate_i_list_chunk)

def draw_insertion(bam_path, chromosome, pic_length, data_dir, num_processes, progress_dict):
    """
    并行处理 BAM 文件中的 reads，采用多进程方式，轮询分配 reads，确保负载均衡。
    """
    print(" > Initialize the tensors...")
    start_time = time.time()
    split_read_left = torch.zeros(pic_length, dtype=torch.int)
    split_read_right = torch.zeros(pic_length, dtype=torch.int)
    rd_count = torch.zeros(pic_length, dtype=torch.int)
    conjugate_m = torch.zeros(pic_length, dtype=torch.int)
    conjugate_i = torch.zeros(pic_length, dtype=torch.int)
    conjugate_d = torch.zeros(pic_length, dtype=torch.int)
    conjugate_s = torch.zeros(pic_length, dtype=torch.int)
    conjugate_i_list = [[0] for _ in range(pic_length)]
    end_time = time.time()
    tensor_initialize_time = end_time - start_time
    print(f"[INFO] Tensor initialize time: {tensor_initialize_time:.2f} seconds")

    # 使用锁进行同步打印
    manager = Manager()
    lock = manager.Lock()

    # 准备每个进程的参数
    args = []
    for i in range(num_processes):
        args.append((i, bam_path, chromosome, pic_length, progress_dict, num_processes, lock))

    # 开始并行处理
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        print(" > Parallel process reads...")
        # 为进度显示打印空行
        for _ in range(num_processes):
            print("")
        results = pool.map(process_chunk, args)
    end_time = time.time()
    parallel_time = end_time - start_time
    print(f"[INFO] Parallel time: {parallel_time:.2f} seconds")

    # 合并所有进程的结果
    start_time = time.time()
    print("\n > Merge parallel results...")
    for result in results:
        split_read_left += result[0]
        split_read_right += result[1]
        conjugate_m += result[2]
        conjugate_i += result[3]
        conjugate_d += result[4]
        conjugate_s += result[5]
        # merge ins signal list
        for i in range(pic_length):
            conjugate_i_list[i].extend(result[6][i])
    end_time = time.time()
    merge_time = end_time - start_time
    print(f"[INFO] Merge time: {merge_time:.2f} seconds")

    # get read depth info
    # print(" > Process read depth info...")
    # with open(os.path.join(data_dir, "depth", chromosome), "r") as f:
    #     for line in f:
    #         pos_count = line.strip().split("\t")[1:]
    #         rd_count[int(pos_count[0]) - 1] = int(pos_count[1])

    start_time = time.time()
    with open(os.path.join(data_dir, "depth", chromosome), "r") as f:
        for line in f:
            pos_count = line[:-1].split("\t")[1:]
            rd_count[int(pos_count[0]) - 1] = int(pos_count[1])
    end_time = time.time()
    RD_process_time = end_time - start_time
    print(f"[INFO] RD process time: {RD_process_time:.2f} seconds")


    return torch.cat([split_read_left.unsqueeze(0), split_read_right.unsqueeze(0), rd_count.unsqueeze(0)], 0), torch.cat([conjugate_m.unsqueeze(0), conjugate_i.unsqueeze(0), conjugate_d.unsqueeze(0), conjugate_s.unsqueeze(0)], 0), conjugate_i_list

def trans2img(bam_path, chromosome, chr_len, data_dir, num_processes):
    print("[*] Start preprocess reads")
    manager = Manager()
    progress_dict = manager.dict()  # 用于进度的共享字典
    chromosome_sign = draw_insertion(bam_path, chromosome, chr_len, data_dir, num_processes, progress_dict)
    print("[*] End preprocess reads")
    return chromosome_sign
