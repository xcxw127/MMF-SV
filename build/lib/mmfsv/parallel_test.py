import copy
import gc
import logging
import math
import multiprocessing
import os
import threading
import time
from argparse import Namespace
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union, Callable

import pysam

from mmfsv import cluster
from mmfsv import leadprov
from mmfsv import postprocessing
from mmfsv import snf
from mmfsv import sv
from mmfsv.region import Region
from mmfsv.result import Result, ErrorResult, CallResult, GenotypeResult, CombineResult
from mmfsv.snfp import PopulationSNF

import sys
import torch

def to_input_image_single(img):
    ims = torch.empty(len(img), hight, hight)

    for i, img_dim in enumerate(img): #
        img_min = torch.min(img_dim)
        img_hight = torch.max(img_dim) - img_min + 2
        im = torch.zeros(len(img_dim), img_hight)

        for x in range(len(img_dim)):
            im[x, :img_dim[x] - img_min + 1] = img_hight

        ims[i] = resize(im.unsqueeze(0))

    return ims

@dataclass
class Task:
    """
    A task is a generic unit of work sent to a child process to be worked on in parallel. Must be pickleable.
    """
    id: int
    sv_id: int
    contig: str
    start: int
    end: int
    config: Namespace
    assigned_process_id: Optional[int] = None
    lead_provider: leadprov.LeadProvider = None
    bam: object = None
    tandem_repeats: list = None
    genotype_svs: list = None
    regions: list[Region] = None
    _logger = None
    result: Result = None

    def __str__(self):
        return f'Task #{self.id}'

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = logging.getLogger(f'mmfsv.progress')

        return self._logger

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def success(self) -> bool:
        return self.done and not self.result.error

    def add_result(self, result: Result) -> None:
        self.result = result

    def execute(self, worker: 'MMFSVWorker' = None) -> Optional[Result]:
        """
        Execute this Task, returning a Result object
        :param worker is the worker executing this task
        """
        raise NotImplemented

    def build_leadtab(self):
        assert (self.lead_provider is None)

        config = self.config

        if config.input_is_cram and config.reference is not None:
            self.bam = pysam.AlignmentFile(config.input, config.input_mode, require_index=True, reference_filename=config.reference)
        else:
            self.bam = pysam.AlignmentFile(config.input, config.input_mode, require_index=True)
        self.lead_provider = leadprov.LeadProvider(config, self.id * config.task_read_id_offset_mult)
        externals = self.lead_provider.build_leadtab(self.regions if self.regions else [Region(self.contig, self.start, self.end)], self.bam)
        return externals, self.lead_provider.read_count

    def call_candidates(self, keep_qc_fails, config) -> list[sv.SVCall]:
        candidates = []
        for svtype in sv.TYPES:
            for svcluster in cluster.resolve(svtype, self.lead_provider, config, self.tandem_repeats):
                for svcall in sv.call_from(svcluster, config, keep_qc_fails, self):
                    if config.dev_trace_read is not False:
                        cluster_has_read = False
                        for ld in svcluster.leads:
                            if ld.read_qname == config.dev_trace_read:
                                cluster_has_read = True
                        if cluster_has_read:
                            import copy
                            svcall_copy = copy.deepcopy(svcall)
                            svcall_copy.postprocess = None
                            print(f"[DEV_TRACE_READ] [3/4] [Task.call_candidates] Read {config.dev_trace_read} -> Cluster {svcluster.id} -> preliminary SVCall {svcall_copy}")
                    candidates.append(svcall)

        self.coverage_average_fwd, self.coverage_average_rev = postprocessing.coverage(candidates, self.lead_provider, config)
        self.coverage_average_total = self.coverage_average_fwd + self.coverage_average_rev
        return candidates

    def finalize_candidates(self, candidates: list['SVCall'], keep_qc_fails, config):
        passed = []
        for svcall in candidates:
            # svcall.qc = svcall.qc and postprocessing.qc_sv(svcall, config)
            # if not keep_qc_fails and not svcall.qc:
            #     continue
            # svcall.qc = svcall.qc and postprocessing.qc_sv_support(svcall, self.coverage_average_total, config)
            # if not keep_qc_fails and not svcall.qc:
            #     continue

            postprocessing.annotate_sv(svcall, config)

            svcall.qc = svcall.qc and postprocessing.qc_sv_post_annotate(svcall, config)

            if config.dev_trace_read:
                cluster_has_read = False
                for ld in svcall.postprocess.cluster.leads:
                    if ld.read_qname == config.dev_trace_read:
                        cluster_has_read = True
                if cluster_has_read:
                    import copy
                    svcall_copy = copy.deepcopy(svcall)
                    svcall_copy.postprocess = None
                    print(f"[DEV_TRACE_READ] [4/4] [Task.finalize_candidates] Read {config.dev_trace_read} -> Cluster {svcall.postprocess.cluster.id} -> finalized SVCall, QC={svcall_copy.qc}: {svcall_copy}")

            if not keep_qc_fails and not svcall.qc:
                continue

            svcall.finalize()  # Remove internal information (not written to output) before sending to mainthread for VCF writing
            passed.append(svcall)
        return passed


class CallTask(Task):
    """
    """

    def execute(self, worker: 'MMFSVWorker' = None) -> CallResult:
        config = self.config

        MMF_bam = config.input
        MMF_data_dir = config.data_dir
        MMF_model = config.model
        MMF_threads = config.threads
        MMF_bs = config.batch_size
        MMF_GPU_idx = config.GPU_index

        # print(f"MMF_bam type: {type(MMF_bam)}, MMF_bam: ", MMF_bam)
        # print(f"MMF_data_dir type: {type(MMF_data_dir)}, MMF_data_dir: ", MMF_data_dir)
        # print(f"MMF_model type: {type(MMF_model)}, MMF_model: ", MMF_model)
        # print(f"MMF_threads type: {type(MMF_threads)}, MMF_threads: ", MMF_threads)
        # print(f"MMF_bs type: {type(MMF_bs)}, MMF_bs: ", MMF_bs)
        # print(f"MMF_GPU_idx type: {type(MMF_GPU_idx)}, MMF_GPU_idx: ", MMF_GPU_idx)


        if config.snf is not None or config.no_qc:
            qc = False
        else:
            qc = True

        _, read_count = self.build_leadtab()
        svcandidates = self.call_candidates(qc, config)

        # 将 `svcandidates` 写入文件
        # with open(f"svcandidates_{self.id}.txt", "w", encoding="utf-8") as f:
        #     f.write(f"svcandidates 类型: {type(svcandidates)}\n")
        #     f.write(f"svcandidates 长度: {len(svcandidates)}\n")
        #     for i, cand in enumerate(svcandidates):
        #         f.write(f"{i+1}: {cand}\n")  # 逐行写入候选变异
        #         f.write(f"{cand}\n")  # 逐行写入候选变异

        # svcalls = self.finalize_candidates(svcandidates, not qc, config)
        # print(f" [INFO] svcall.contig = {svcalls[0].contig}, svcall_len = {len(svcalls)}.")
        svcalls_unfiltered = self.finalize_candidates(svcandidates, not qc, config)

        # print(f"✅ 原始 svcalls 类型: {type(svcalls)}")
        # print(f"✅ 原始 svcalls 长度: {len(svcalls)}")

        # **写入 TXT 文件**
        # lock = threading.Lock()

        # with lock:
        #     with open("svcalls_all.txt", "a", encoding="utf-8") as f:
        #         for item in svcalls:
        #             f.write(f"{item}\n")

        # with open("svcalls_all.txt", "a", encoding="utf-8") as f:
        #     for item in svcalls:
        #         f.write(f"{item}\n")  # 逐行写入

        # **读取 TXT 文件**
        # with open("svcalls_all.txt", "r", encoding="utf-8") as f:
        #     new_svcalls = [line.strip() for line in f.readlines()]

        # print(f"✅ 重新读取后 new_svcalls 类型: {type(new_svcalls)}")
        # print(f"✅ 重新读取后 new_svcalls 长度: {len(new_svcalls)}")


        # 将 `svcalls` 写入文件
        # with open(f"svcalls_{self.id}.txt", "w", encoding="utf-8") as f:
        #     f.write(f"svcalls 类型: {type(svcalls)}\n")
        #     f.write(f"svcalls 长度: {len(svcalls)}\n")
        #     for i, svcall in enumerate(svcalls):
        #         f.write(f"{i+1}: {svcall}\n")  # 逐行写入最终变异

        parsed_calls = []

        for line in svcalls_unfiltered:
            call_obj = line
            parsed_calls.append(call_obj)

        # load files
        chromosome = parsed_calls[0].contig
        print(f"[DEBUG] > load {chromosome} files...")
        chromosome_sign = torch.load(MMF_data_dir + "chromosome_sign/" + chromosome + ".pt", weights_only=True)
        mid_sign = torch.load(MMF_data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt", weights_only=True)
        mid_sign_img = torch.load(MMF_data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt", weights_only=True)

        # chr_order_map = {str(i): i for i in range(1, 23)}  # "1"->1, "2"->2, ..., "22"->22
        # chr_order_map.update({"X": 23, "Y": 24})  # "X"->23, "Y"->24

        # # sort chr name
        # def sort_key(call_with_index):
        #     i, call = call_with_index
        #     c = call.contig
        #     if c.startswith("chr"):
        #         c = c[3:]
        #     if c in chr_order_map:
        #         return (0, chr_order_map[c], call.pos)
        #     else:
        #         return (1, i)

        # calls_with_idx = list(enumerate(parsed_calls)) 
        # calls_with_idx.sort(key=sort_key)
        # sorted_calls = [x[1] for x in calls_with_idx]

        # parsed_calls = sorted_calls

        svcalls = []
        total_count = len(parsed_calls)
        current_count = 0


        print("[* filter begin *]")
        sam_file = pysam.AlignmentFile(MMF_bam, "rb")
        chr_list_tmp = sam_file.references
        chr_length_tmp = sam_file.lengths
        sam_file.close()

        chr_len_dict = dict(zip(chr_list_tmp, chr_length_tmp))
        chromosome_lengths = chr_len_dict.get(chromosome) 

        for call in parsed_calls:
            # print("Type of call:", type(call))
            # print("call.contig:", call.contig)
            current_count += 1
            sys.stdout.write(f"\r Filtered SV = {current_count}/{total_count}")
            sys.stdout.flush()

            if call.svtype not in ["DEL", "INS"]:
                svcalls.append(call)
                continue

            try:
                pos_begin = call.pos

                if call.svtype == "DEL":
                    end_val = None

                    info_dict = call.info or {}
                    if "end" in info_dict:
                        end_val = int(info_dict["end"])
                    elif "svlen" in info_dict:
                        svlen_str = str(info_dict["svlen"])
                        end_val = pos_begin + int(svlen_str)
                    else:
                        end_val = pos_begin + 1

                    # calculate gap
                    gap = int((end_val - pos_begin) / 4)
                    if gap == 0:
                        gap = 1

                    # final begin/end_
                    begin = pos_begin - 1 - gap
                    end_  = end_val   - 1 + gap

                elif call.svtype == "INS":

                    gap = 112
                    begin = pos_begin - 1 - gap
                    end_  = pos_begin - 1 + gap

                chr_len = chromosome_lengths
                if chr_len is None:
                    print(f"\n[WARNING] Can not find chromosome length, skip.")
                    continue

                # border check
                if begin < 0:
                    begin = 0
                if end_ >= chr_len:
                    end_ = chr_len - 1

                # skip illegal border
                if begin > end_:
                    max_label = "begin"
                    max_value = begin
                else:
                    max_label = "end"
                    max_value = end_

                print(f"\n [DEBUG] call.svtype = {call.svtype}, chr_len = {chr_len}, begin = {begin}, end_ = {end_}")
                if begin >= end_:
                    continue

                # generate images
                print(f" [DEBUG] > generate {call.contig} ins_img...")
                ins_img = to_input_image_single(chromosome_sign[:, begin:end_])
                print(f" [DEBUG] > generate {call.contig} ins_img_mid...")
                ins_img_mid = to_input_image_single(mid_sign[:, begin:end_])
                print(f" [DEBUG] > generate {call.contig} img_i...")
                img_i = resize(mid_sign_img[begin:end_].unsqueeze(0))
                print(f" [DEBUG] > generate {call.contig} ins_img_i...")
                ins_img_i = img_i.reshape(1, -1)

                # model predict
                print(f" [DEBUG] > model predict...")
                all_ins_img = torch.cat([ins_img, ins_img_mid], dim=0)
                y_hat = model(all_ins_img.unsqueeze(dim=0).cuda(), ins_img_i.cuda())
                pred_type = torch.argmax(y_hat, dim=1).item()

                if pred_type == 0:
                    # continue
                    pass 
                elif pred_type == 1:
                    call.svtype = "DEL"
                    call.alt = "<DEL>"
                elif pred_type == 2:
                    call.svtype = "INS"
                    call.alt = "<INS>"

                svcalls.append(call)

            except RuntimeError as e:
                svcalls.append(call)

        print("\n[* filter end *]")

        if not config.no_qc:
            svcalls = [s for s in svcalls if s.qc]

        if config.sort:
            svcalls = sorted(svcalls, key=lambda svcall: svcall.pos)

        from mmfsv.result import CallResult
        result = CallResult(self, svcalls, read_count)

        if config.snf is not None:  # and len(svcandidates):
            snf_filename = f"{config.snf}.tmp_{self.id}.snf"

            with open(snf_filename, "wb") as handle:
                snf_out = snf.SNFile(config, handle)
                for cand in svcandidates:
                    snf_out.store(cand)
                snf_out.annotate_block_coverages(self.lead_provider)
                snf_out.write_and_index()
                handle.close()
            result.snf_filename = snf_filename
            result.snf_index = snf_out.get_index()
            result.snf_total_length = snf_out.get_total_length()
            result.snf_candidate_count = len(svcandidates)
            result.has_snf = True

        result.coverage_average_total = self.coverage_average_total

        return result


class GenotypeTask(Task):
    def execute(self, worker: 'MMFSVWorker' = None) -> Optional[GenotypeResult]:
        config = self.config

        qc = False
        _, read_count = self.build_leadtab()
        svcandidates = self.call_candidates(qc, config=config)
        svcalls = self.finalize_candidates(svcandidates, not qc, config=config)

        binsize = 5000
        binedge = int(binsize / 10)
        genotype_svs_svtypes_bins = {svtype: {} for svtype in sv.TYPES}
        for genotype_sv in self.genotype_svs:
            genotype_sv.genotype_match_sv = None
            genotype_sv.genotype_match_dist = math.inf

            if genotype_sv.svtype not in genotype_svs_svtypes_bins:
                logging.getLogger('mmfsv').warning(f'Unsupported SVTYPE: {genotype_sv.svtype}')
                continue

            bins = [int(genotype_sv.pos / binsize) * binsize]
            if genotype_sv.pos % binsize < binedge:
                bins.append((int(genotype_sv.pos / binsize) - 1) * binsize)
            if genotype_sv.pos % binsize > binsize - binedge:
                bins.append((int(genotype_sv.pos / binsize) + 1) * binsize)

            for bin in bins:
                if bin not in genotype_svs_svtypes_bins[genotype_sv.svtype]:
                    genotype_svs_svtypes_bins[genotype_sv.svtype][bin] = []
                genotype_svs_svtypes_bins[genotype_sv.svtype][bin].append(genotype_sv)

        for cand in svcandidates:
            bin = int(cand.pos / binsize) * binsize
            if bin not in genotype_svs_svtypes_bins[cand.svtype]:
                continue
            if cand.svtype == "BND":
                for genotype_sv in genotype_svs_svtypes_bins[cand.svtype][bin]:
                    dist = abs(genotype_sv.pos - cand.pos)
                    # if minlen>0 and dist < genotype_sv.genotype_match_dist and dist <= config.cluster_merge_bnd * 2:
                    if dist < genotype_sv.genotype_match_dist and dist <= config.cluster_merge_bnd:
                        if cand.bnd_info.mate_contig == genotype_sv.bnd_info.mate_contig:
                            genotype_sv.genotype_match_sv = cand
                            genotype_sv.genotype_match_dist = dist
            else:
                for genotype_sv in genotype_svs_svtypes_bins[cand.svtype][bin]:
                    dist = abs(genotype_sv.pos - cand.pos) + abs(abs(genotype_sv.svlen) - abs(cand.svlen))
                    minlen = float(min(abs(genotype_sv.svlen), abs(cand.svlen)))
                    if minlen > 0 and dist < genotype_sv.genotype_match_dist and dist <= config.combine_match * math.sqrt(
                            minlen) and dist <= config.combine_match_max:
                        genotype_sv.genotype_match_sv = cand
                        genotype_sv.genotype_match_dist = dist

        postprocessing.coverage(self.genotype_svs, self.lead_provider, config)

        # Determine genotypes for unmatched input SVs
        for svcall in self.genotype_svs:
            coverage_list = [svcall.coverage_start, svcall.coverage_center, svcall.coverage_end]
            coverage_list = [c for c in coverage_list if c is not None]
            if len(coverage_list) == 0:
                return
            coverage = round(sum(coverage_list) / len(coverage_list))
            svcall.genotypes = {}
            if coverage > 0:
                svcall.genotypes[0] = (0, 0, 0, coverage, 0, (None, None))
            else:
                svcall.genotypes[0] = config.genotype_none

        from mmfsv.result import GenotypeResult
        return GenotypeResult(self, self.genotype_svs, read_count)


class CombineTask(Task):
    """
    Task to merge/combine multiple SNF files into one.
    """
    # target number of blocks to process in one task. this is the total number of blocks over all input files,
    # i.e. merging 100 files means 100 blocks wrt this value
    TARGET_WORK_PER_TASK = 10000

    result_class = CombineResult

    block_indices: list[int] = None  # List of block starts to process on this task

    def __init__(self, *args, **kwargs):
        self.result_class = kwargs.pop('result_class', None) or self.result_class
        super().__init__(*args, **kwargs)
        self.generate_blocks()

    def generate_blocks(self):
        """
        Generate a set of blocks
        """
        if self.regions:
            block_indices = set()
            for r in self.regions:
                start = r.start // self.config.snf_block_size * self.config.snf_block_size
                block_indices |= set(range(start, r.end + self.config.snf_block_size, self.config.snf_block_size))
            self.block_indices = list(sorted(block_indices))
        else:
            self.block_indices = list(range(self.start, self.end + self.config.snf_block_size, self.config.snf_block_size))

    def __str__(self):
        if len(self.block_indices) > 0:
            return f'''Task {self.id} Contig {self.contig} [{self.start} ({self.block_indices[0]}) .. {self.end} ({self.block_indices[-1]})]'''
        else:
            return f'Task {self.id} [no blocks available]'

    def clone(self, first_block: int, block_count: int, new_id: int = None, emit_first: bool = True) -> 'CombineTask':
        """
        Clone this task with a subset of the blocks to be processed.
        """
        obj = copy.copy(self)
        if new_id is not None:
            obj.id = new_id
        obj.block_indices = self.block_indices[first_block:first_block + block_count]
        obj.start = obj.block_indices[0]
        obj.end = obj.block_indices[-1] + obj.config.snf_block_size
        return obj

    def scatter(self) -> list['CombineTask']:
        """
        Scatter this task to a number of tasks to be run in parallel, on block level:
        - Distribute blocks equally to tasks, each one working on consecutive blocks
        - First task gets any blocks that can not be evenly distributed
        - Tasks other than the first one have to process their first block for kept groups, and
          should not emit calls for this block (as these will be emitted by the previous task)
        """
        total_blocks = len(self.block_indices) * len(self.config.sample_ids_vcf)
        if total_blocks <= self.TARGET_WORK_PER_TASK or self.config.threads <= 1:
            return [self]

        blocks_per_task = (total_blocks // self.TARGET_WORK_PER_TASK)

        return [
            self.clone(
                fb,
                blocks_per_task,
                new_id=self.id + i + 1
            ) for i, fb in enumerate(range(0, len(self.block_indices), blocks_per_task))
        ]

    def execute(self, worker: 'MMFSVWorker' = None):
        samples_headers_snf = {}
        for snf_info in self.config.snf_input_info:
            snf_in = snf.SNFile(self.config, open(snf_info["filename"], "rb"), filename=snf_info["filename"])
            snf_in.read_header()
            samples_headers_snf[snf_info["internal_id"]] = snf_in

            if self.config.combine_close_handles:
                snf_in.close()

        if self.config.combine_population:
            self.config.combine_population = PopulationSNF.open(self.config.combine_population)

        result = self.result_class(self, [], 0)

        # block_groups_keep_threshold=5000
        # TODO: Parameterize
        bin_min_size = self.config.combine_min_size
        bin_max_candidates = max(25, int(len(self.config.snf_input_info) * 0.5))
        overlap_abs = self.config.combine_overlap_abs
        support_threshold = self.config.combine_support_threshold

        sample_internal_ids = set(samples_headers_snf.keys())

        #
        # Load candidate SVs from all samples for each block separately and cluster them based on start position
        #
        candidates_processed = 0
        groups_keep = {svtype: list() for svtype in sv.TYPES}
        calls = []

        for cur, block_index in enumerate(self.block_indices):  # iterate over all blocks
            self.logger.info(f'Processing block {cur + 1}/{len(self.block_indices)} (active calls: {sv.SVCall._counter} groups: {sv.SVGroup._counter})')
            samples_blocks = {}
            if calls:
                result.store_calls(calls)
                calls = []

            for sample_internal_id, sample_snf in samples_headers_snf.items():
                blocks = sample_snf.read_blocks(self.contig, block_index)
                samples_blocks[sample_internal_id] = blocks
                # sample_internal_id is the number of the processed file
                # blocks is a list[dict[str, list[SVCall]]] {'INS': [...], 'DEL': [...], ...}

            for svtype in sv.TYPES:
                bins = {}
                for sample_internal_id, sample_snf in samples_headers_snf.items():  # fetch current block for each file
                    blocks = samples_blocks[sample_internal_id]
                    reqc = sample_snf.reqc

                    if blocks is None:
                        continue
                    for block in blocks:  # usually only 1 block
                        for cand in block[svtype]:
                            # if config.combine_pass_only and (cand.qc==False or cand.filter!="PASS"):
                            #    continue
                            if cand.support < support_threshold:
                                continue

                            if reqc:
                                postprocessing.genotype_sv(cand, self.config)

                            cand.sample_internal_id = sample_internal_id

                            bin = int(cand.pos / bin_min_size) * bin_min_size
                            if bin not in bins:
                                bins[bin] = [cand]
                            else:
                                bins[bin].append(cand)
                        candidates_processed += len(block[svtype])

                if len(bins) == 0:
                    continue

                size = 0
                svcands = []
                keep = groups_keep[svtype]
                sorted_bins = sorted(bins)
                last_bin = sorted_bins[-1]
                for curr_bin in sorted_bins:
                    svcands.extend(bins[curr_bin])  # here SVCalls from bins are collected...
                    size += bin_min_size

                    if (not self.config.combine_exhaustive and len(svcands) >= bin_max_candidates) or curr_bin == last_bin:
                        if len(svcands) == 0:
                            size = 0
                            continue

                        svgroups = cluster.resolve_block_groups(svtype, svcands, keep, self.config)
                        groups_call = []
                        keep = []
                        for group in svgroups:
                            coverage_bin = int(
                                group.pos_mean / self.config.coverage_binsize_combine) * self.config.coverage_binsize_combine
                            # High Intensity loop
                            for non_included_sample in sample_internal_ids - group.included_samples:
                                if samples_blocks[non_included_sample] is not None and coverage_bin in samples_blocks[non_included_sample][0]["_COVERAGE"]:
                                    coverage = samples_blocks[non_included_sample][0]["_COVERAGE"][coverage_bin]
                                else:
                                    coverage = 0
                                if non_included_sample in group.coverages_nonincluded:
                                    group.coverages_nonincluded[non_included_sample] = max(
                                        coverage,
                                        group.coverages_nonincluded[non_included_sample]
                                    )
                                else:
                                    group.coverages_nonincluded[non_included_sample] = coverage

                            if abs(group.pos_mean - curr_bin) < max(size * 0.5, overlap_abs):
                                keep.append(group)
                            else:
                                groups_call.append(group)

                        calls.extend(sv.call_groups(groups_call, self.config, self))

                        size = 0
                        svcands = []

                groups_keep[svtype] = keep

        for svtype in groups_keep:
            calls.extend(sv.call_groups(groups_keep[svtype], self.config, self))

        if calls:
            result.store_calls(calls)

        result.finalize()
        return result


class ShutdownTask:
    id = None

    def __str__(self):
        return 'Shutdown Request'

    def execute(self, *args, **kwargs) -> Result:
        raise MMFSVWorker.Shutdown


class MMFSVWorker:
    """
    Handle for a worker process. Since we're forking, this class will be available in
    both the parent and the worker processes.
    """
    id: int  # sequential ID of this worker, starting with 0 for the first
    externals: list = None
    recycle: bool = False
    running = True
    pid: int = None
    # Event to shut down heartbeat threads
    _shutdown: threading.Event
    _heartbeat: float = 0  # last heartbeat received
    HEARTBEAT_INTERVAL = 3  # in seconds
    HEARTBEAT_TIMEOUT = 10  # in seconds

    class Shutdown(Exception):
        """
        Indicates this worker process should shut down
        """

    def __init__(self, process_id: int, config: Namespace, tasks: deque[Task], recycle_hint: Union[bool, Callable] = None):
        self.id = process_id
        self.config = config
        self.tasks = tasks
        self.task = None
        self.finished_tasks = []
        self.recycle = recycle_hint

        self.pipe_main, self.pipe_worker = multiprocessing.Pipe()
        self.heartbeat_main, self.heartbeat_worker = multiprocessing.Pipe()

        self.process = multiprocessing.Process(
            target=self.run_worker,
            daemon=True
        )

        self._logger = logging.getLogger('MMF-SV.worker')

    def __str__(self):
        return f'Worker {self.id} @ process {self.pid}'

    def start(self) -> None:
        self._logger.info(f'Starting worker {self.id}')
        self.running = True
        self.process.start()
        self._heartbeat = time.monotonic()

    def maybe_recycle(self):
        """
        Recycle this worker if that has been requested
        """
        recycle = self.recycle(self.id, self.process.pid) if callable(self.recycle) else self.recycle

        if recycle:
            self._logger.info(f'Recycling worker {self.id}')
            # Shut down current worker process
            self.pipe_main.send(ShutdownTask())
            self.process.join(2)
            # Start new one
            self.process = multiprocessing.Process(
                target=self.run_worker,
                daemon=True
            )
            self.process.start()
            self._heartbeat = time.monotonic()

    def run_parent(self) -> bool:
        """
        Worker thread, running in parent process
        """
        try:
            if self.task is None:
                # we are not working on something...
                if len(self.tasks) > 0:
                    # ...but there is more work to be done
                    self.maybe_recycle()

                    try:
                        self.task = self.tasks.popleft()
                    except IndexError:
                        # another worker may have taken the last task
                        self._logger.debug(f'No more tasks to do for {self.id}')
                    else:
                        self.pipe_main.send(self.task)
                        self._logger.info(f'Dispatched task #{self.task.id} to worker {self.id} ({len(self.tasks)}  tasks left)')
                else:
                    # ...and no more work available, so we shut down this worker
                    self._logger.info(f'Worker {self.id} shutting down...')
                    self.pipe_main.send(ShutdownTask())
                    self.running = False
            else:
                if self.pipe_main.poll(0.01):
                    self._logger.debug(f'Worker {self.id} got result for task {self.task.id}...')
                    result: Result = self.pipe_main.recv()

                    if result.error:
                        self._logger.error(f'Worker {self.id} received error: {result}')
                    else:
                        self._logger.info(f'Worker {self.id} got result for task #{result.task_id}')

                    self.task.add_result(result)
                    self.finished_tasks.append(self.task)
                    self.task = None

                if self.heartbeat_main.poll():
                    hb = self.heartbeat_main.recv()
                    self._heartbeat = time.monotonic()
                    # self._logger.debug(f'Worker {self.id} got heartbeat #{hb}')

                if self._heartbeat < time.monotonic() - self.HEARTBEAT_TIMEOUT:
                    self._logger.debug(f'Worker {self.id} missed heartbeat!')
                    try:
                        self.process.join(0.2)  # try collecting process remains...
                    except:  # noqa
                        ...
                    if self.process.exitcode is not None:
                        # if we got an exitcode, the process really was killed
                        self._logger.warning(f'Worker {self.id} found dead!')
                        if self.task:  # if we were working on a task, requeue it to have it picked up by another worker...
                            self.tasks.appendleft(self.task)
                        self.running = False  # ...and shut down
        except:
            self._logger.exception(f'Unhandled error in worker {self.id}. This may result in an orphened worker process.')
            try:
                self.process.kill()
            except:
                ...

        return self.running

    def finalize(self):
        self.process.join(10)

        if self.process.exitcode is None:
            self._logger.warning(f'Worker {self.id} refused to shut down gracefully, killing it.')
            self.process.kill()
            self.process.join(2)
        self._logger.info(f'Worker {self.id} done (code {self.process.exitcode}).')

    def run_worker(self):
        """
        Entry point/main loop for the worker process
        """
        self.pid = os.getpid()
        self._shutdown = threading.Event()

        t = threading.Thread(target=self.run_worker_heartbeats, daemon=True)
        t.start()

        while self.running:
            self._logger.debug(f'Worker {self.id} ({self.pid}) waiting for tasks...')

            task = self.pipe_worker.recv()

            self._logger.debug(f'Worker {self.id} got task {task}')

            try:
                result = task.execute(self)
            except self.Shutdown:
                self.running = False
                self._shutdown.set()
            except Exception as e:
                self._logger.exception(msg := f'Error in worker process while executing {task}')
                self.pipe_worker.send(ErrorResult(msg))
            else:
                self._logger.debug(f'Worker {self.id} finished executing {task}, sending back result...')

                if result is not None:
                    self.pipe_worker.send(result)

            del task
            gc.collect()

        t.join(1.0)

    def run_worker_heartbeats(self):
        hb = 0
        while self.running:
            hb += 1
            self.heartbeat_worker.send(hb)
            self._shutdown.wait(self.HEARTBEAT_INTERVAL)


def execute_task(task: Task):
    logging.getLogger('mmfsv.parallel').info(f'Working on {task}')
    return task.execute()


class MMFSVParentWorker(MMFSVWorker):
    """
    A worker class without multiprocessing, i.e. running in the main process. Used for profiling.
    """
    id: int = 0

    def __init__(self, config: Namespace, tasks: list[Task], **kwargs):  # noqa
        self.tasks = tasks
        self.task = None
        self.config = config
        self.finished_tasks: list[Task] = []
        self._log = logging.getLogger('MMF-SV.pworker')
        self._log.info(f'Using parent worker')

    def start(self) -> None:
        ...

    def run_parent(self) -> bool:
        count = len(self.tasks)
        for i, task in enumerate(self.tasks):
            self._log.info(f'Executing {task} ({i+1}/{count})')
            result = task.execute(self)
            task.add_result(result)
            self.finished_tasks.append(task)
        self._log.info(f'All tasks done.')

        return False

    def finalize(self):
        ...
