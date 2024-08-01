import os, sys, pickle, numpy, platform, time, argparse, glob
from math import log, sqrt
from subprocess import call


notdebugging = True


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder')    
    parser.add_argument('-low')
    parser.add_argument('-high')
    parser.add_argument('-alpha')
    parser.add_argument('-c')
    args = None
    args = parser.parse_args()
    if args is None:
        exit(0)
    return args
    


def generate_slurm_onefile (folder, k, alpha, c):

    eachfile_base = "k" + str(k)
    eachfile_base_sl = eachfile_base + ".sl"
    eachfile = folder + "/" + eachfile_base
    eachfile_sl = folder + "/" + eachfile_base_sl    

    #print (eachfile_base)
    #print (eachfile_base_sl)    
    #print (eachfile)
    
    # debug or not, redirect
    if (notdebugging):
        orig_stdout = sys.stdout
        f_sl_filename = open(eachfile_sl, 'w')
        sys.stdout = f_sl_filename

    # ===========================================        
    print ("#!/bin/bash")
    print ("#SBATCH -J " + eachfile_base)
    print ("#SBATCH --workdir=" + folder)
    print ("#SBATCH -o " + eachfile + ".out")
    print ("#SBATCH -e " + eachfile + ".err")
    print ("#SBATCH --partition=longq7")
    print ("module load slurm")
    print ("module load anaconda/3")
    print ("source activate env_python3")

    # @@@ most important line, put the command here!
    #print ("/home/sbai/.conda/envs/env_python3/bin/python3 /home/sbai/my_notes2/moar-approx-svp/code/test/main_opt.py -k", k, " -alpha ", alpha, " -num 1 -c ", c)

    print ("/home/sbai/.conda/envs/env_python3/bin/python3 /home/sbai/my_notes2/moar-approx-svp/code/test/main_opt_t4.py -k", k, " -alpha ", alpha, " -num 1 -c ", c)
    
    print("conda deactivate")
    print("")
    # ===========================================
    
    if (notdebugging):    
        sys.stdout = orig_stdout
        f_sl_filename.close()

    if (notdebugging and 1):
        print("# [****] " + eachfile_sl + " generated.")
        call("sbatch " + eachfile_sl, shell=True)        
        call("sleep 0.1", shell=True)
        
    return

    
def generate_slurm (folder, low, high, alpha, c):
    
    #print "input name is", folder

    # get foldername
    folder_path = os.path.abspath(folder)

    for k in range(low, high):
        generate_slurm_onefile(folder_path, k, alpha, c)
    
    return

    
def main():

    # parse argument
    args = parse_options()
    print ("###################################### ")
    print ("# [Args] folder: %s" % args.folder)    
    print ("# [Args] low: %s" % args.low)
    print ("# [Args] high: %s" % args.high)
    print ("# [Args] alpha: %s" % args.alpha)
    print ("# [Args] c: %s" % args.c)
    print ("###################################### ")
        
    # start process
    folder = str(args.folder)
    low = int(args.low)
    high = int(args.high)
    c = float(args.c)
    alpha = float(args.alpha)

    # read files
    generate_slurm(folder, low, high, alpha, c)
    
    return


# python call_main_opt.py -folder FOLDERNAME -low 100 -high 500 -alpha 1.00 -c 0.0
if __name__ == '__main__':
    main()
