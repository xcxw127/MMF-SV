#!/usr/bin/env python
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(prog='mmfsv', description="MMF-SV CLI Wrapper")
    subparsers = parser.add_subparsers(dest='command')

    # call子命令 (调用原始mmfsv主程序)
    parser_call = subparsers.add_parser('call', help='Run MMF-SV main program')
    parser_call.add_argument('-i', required=True, help='Input BAM file')
    parser_call.add_argument('-D', required=True, help='Data directory')
    parser_call.add_argument('-M', required=True, help='Model directory')
    parser_call.add_argument('-v', required=True, help='Output VCF')
    parser_call.add_argument('-t', required=True, help='Thread num')
    parser_call.add_argument('-GPU', required=True, help='GPU index')

    # preprocess_VCF子命令
    parser_pre = subparsers.add_parser('preprocess_VCF', help='Preprocess VCF file')
    parser_pre.add_argument('-D', required=True, help='Path to the data directory')
    parser_pre.add_argument('-vcf', required=True, help='Path to the VCF directory')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.command == 'call':
        mmfsv_exec = os.path.join(script_dir, 'mmfsv')
        cmd = [
            sys.executable, mmfsv_exec,
            '-i', args.i,
            '-D', args.D,
            '-M', args.M,
            '-v', args.v,
            '-t', args.t,
            '-GPU', args.GPU
        ]
        subprocess.run(cmd)
    elif args.command == 'preprocess_VCF':
        pre_script = os.path.join(script_dir, 'scripts', 'preprocess_VCF.py')
        cmd = [sys.executable, pre_script, '-D', args.D, '-vcf', args.vcf]
        subprocess.run(cmd)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
