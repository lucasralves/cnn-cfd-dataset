"""
    Run cases
"""
from os import listdir, system
from tqdm import tqdm

if __name__ == '__main__':

    # Cases
    cases = listdir('./data/cases/')

    # Loop
    for case in tqdm(cases):
        if case != 'example':
            system('./data/cases/' + case + '/Allrun')