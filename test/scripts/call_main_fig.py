import os, sys, pickle, numpy, platform, time, argparse, glob
from math import log, sqrt
from subprocess import call


notdebugging = False



def generate_slurm_onefile (folder, alpha, high, seed):

    alpha_int = round(alpha*100) % 100
    eachfile_base = "k" + str(high) + "_a" + str(alpha_int) + "_s" + str(seed)
    eachfile_base_sl = eachfile_base + ".sl"
    eachfile = folder + "/" + eachfile_base
    eachfile_sl = folder + "/" + eachfile_base_sl    

    eachfileobj = eachfile + ".sobj"
    
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
    print ("/home/sbai/.conda/envs/env_python3/bin/python3 /home/sbai/my_notes2/moar-approx-svp/code/call.py -c 0.15 --alpha ", \
               alpha, " --seed ", seed, "--ahsvp-strategies /home/sbai/my_notes2/moar-approx-svp/data/approx-hsvp-simulations,qary,1.10,0.15,1.00.csv ", \
               " --dump-filename ", eachfileobj, \
               high, " /home/sbai/my_notes2/moar-approx-svp/data/fplll-strategies-one-tour-strombenzin.json")
    
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

    
def generate_slurm (folder, high, alpha):
    

    # get foldername
    folder_path = os.path.abspath(folder)

    # for each seed
    for seed in range(64):
        generate_slurm_onefile(folder_path, alpha, high, seed)

    return




def main():

    # parse argument
    args = parse_options()
    print ("###################################### ")
    print ("# [Args] folder: %s" % args.folder)    
    print ("# [Args] high: %s" % args.high)
    print ("# [Args] alpha: %s" % args.alpha)
    print ("###################################### ")
        
    # start process
    folder = str(args.folder)
    high = int(args.high)
    alpha = float(args.alpha)

      
    # read filesa
    generate_slurm (folder, high, alpha)
    
    return


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder')    
    parser.add_argument('-high')
    parser.add_argument('-alpha')
    args = None
    args = parser.parse_args()
    if args is None:
        exit(0)
    return args
    

# python call_main_fig.py -folder FOLDERNAME -high 80 -alpha 1.10
if __name__ == '__main__':
    main()
