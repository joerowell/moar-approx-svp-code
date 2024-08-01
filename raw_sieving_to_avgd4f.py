"""
This script is to be used to average out the results of many different sieving runs.
Briefly, this script produces the arithmetic mean of the maximum sieving dimension of a sieving run (i.e it produces ((n_1 - d4f_1) + (n_2-d4f_2) + ..... + (n_m - d4f_m)) / m)
"""

from collections import OrderedDict

def gh(n: "value to complete the gh of"):
    """
    Compute gh(n).
    Here, as with the rest of the calculations, we use Stirling's approximation for the GH
    """
    return (n/2/pi/e)*(pi*n)**(1/(2*n))

def find_matching(n_0, alpha_0, alpha_1):
    """
    For a given n_0, alpha_0, alpha_1 find n_1 such that:
    (alpha_0 * gh(n_0)) ^ (1/(n_0-1)) == (alpha_1*gh(n_1)) ^ (1/(n_1-1))
    """

    var("n_1")
    f = (alpha_0*gh(n_0))**(1/(n_0 -1)) == (alpha_1 * gh(n_1))**(1/(n_1-1))
    return int(round(f.find_root(2, 1000)))

def normalise(n_0, alpha_0):
    """ Wrapper function: just delegates to find matching, but with fixed alpha_1  = 1.0 """
    return find_matching(n_0, alpha_0, 1.00)

def produce_dictionary_from_file(name: "filename to read from"):
    """
    Read from the file '"name' and produce a dictionary.
    This dictionary merely maps the values on the left-hand-side of a colon to the value on the right.
    I.e output["walltime"] = 100s, say

    :param name the name of the file from which we're reading. This should be the final line of G6K output (root: ......), and not the verbose output.
    """
    input_file = open(name, "r")
    full_output = {}
    """
    The G6K final line output is typically of the following form:
    root:                       threads: 24 etc etc

    This means that we need to (a) split around the colons (as this is what the mapping is), and (b) remove the leading "root:........" part so that we can read properly
    """

    for line in input_file:
        # Produce a temporary dictionary
        output = OrderedDict()
        # Split around colons
        split_string = line.split(":")
        # Remove the "root" string
        split_string = split_string[1:]
        processed_string = []
        for ss in split_string:
            # Because of the layout of these strings, we need to remove the spaces, as well as any commas
            ss = ss.replace(" ","")
            # With this split, we replace commas with spaces. This is so that we can, if necessary, split around spaces again later to extract key/value pairs
            ss = ss.replace(","," ")
            # If there's nothing less then skip
            if ss != "":
                # Now split around these spaces if necessary
                ss = ss.split(" ")
                for s in ss:
                    s = s.replace(" ", "")
                    if s != "":
                        processed_string.append(s)
        # With this split string, we can now match key-value pairs: we may well be missing some pieces though, so let's check if
        if len(processed_string)%2 != 0:
            print("Error: input line is malformed for:" + name + "!")
            quit()

        for pos in range(0, len(processed_string),2):
            output[processed_string[pos]] = processed_string[pos+1]
        # With this output, we can check if it is in the broader dictionary
        # We add up the data we're interested in (D4F, walltime etc) and count how many times we've actually updated each record
        # That way, we can compute averages later
        key = output['n']
        if key in full_output.keys():
            # Here we use a counting set
            cputime = output["cputime"].replace("s","")
            cputime = float(cputime)
            walltime = output["walltime"].replace("s","")
            walltime = float(walltime)


            full_output[key]['cputime']  += cputime
            full_output[key]['walltime'] += walltime
            full_output[key]['flast']    += float(output['flast'])
            full_output[key]['count']    += 1
        else:
            temp_dict = OrderedDict()
            full_output[key] = OrderedDict()
            cputime = output["cputime"].replace("s","")
            cputime = float(cputime)
            walltime = output["walltime"].replace("s","")
            walltime = float(walltime)

            full_output[key]['cputime']  = cputime
            full_output[key]['walltime'] = walltime
            full_output[key]['flast']    = float(output['flast'])
            full_output[key]['count'] = 1

    # Compute the averages for cputime, walltime and flast
    for key in full_output:
        full_output[key]['cputime' ] = full_output[key]['cputime' ] / full_output[key]['count']
        full_output[key]['walltime'] = full_output[key]['walltime'] / full_output[key]['count']
        full_output[key]['flast']    = full_output[key]['flast']    / full_output[key]['count']
    return full_output


if __name__ == "__main__":

    dict_exact = produce_dictionary_from_file("../data/exact-sieving-raw.txt")
    print("rank, cputime, walltime, msd,")
    for key in dict_exact:
        print(str(key) + "," + str(dict_exact[key]['cputime']) + "," + str(dict_exact[key]['walltime']) + "," + str(float(key) - dict_exact[key]['flast']) + ",")
    print()
    '''
    dict_105    = produce_dictionary_from_file("../data/alpha-1.05-relaxed-sieving-raw.txt")
    print("rank, cputime, walltime, msd,")
    for key in dict_105:
        print(str(normalise(float(key), 1.05)) + "," + str(dict_105[key]['cputime']) + "," + str(dict_105[key]['walltime']) + "," + str(float(key) - dict_105[key]['flast']) + ",")
    print()
    '''
    '''
    dict_110    = produce_dictionary_from_file("../data/alpha-1.10-relaxed-sieving-raw.txt")
    print("rank, cputime, walltime, msd,")
    for key in dict_110:
        print(str(normalise(float(key), 1.10)) + "," + str(dict_110[key]['cputime']) + "," + str(dict_110[key]['walltime']) + "," + str(float(key) - dict_110[key]['flast']) + ",")
    print()
    '''
