"""
    Create cases (1920 cases from 160 meshes and 12 AoA)
"""
from os import listdir, system
from random import random
from tqdm import tqdm

def updateBoundaryFile(case: str) -> None:

    # Update boundary file
    file = open('./data/cases/{}/constant/polyMesh/boundary'.format(case), 'r')
    new = open('./data/cases/{}/constant/polyMesh/boundary2'.format(case), 'w')

    part = 0

    for line in file.readlines():
        if 'laterals' in line: part = 1
        if 'foil' in line: part = 2
        if 'external' in line: part = 3

        if 'physicalType' not in line:
            if 'type' in line:
                if part == 1:
                    new.write('        type            empty;\n')
                elif part == 2:
                    new.write('        type            wall;\n')
                elif part == 3:
                    new.write('        type            patch;\n')
            else:
                new.write(line)
    
    file.close()
    new.close()

    system('rm ./data/cases/{}/constant/polyMesh/boundary'.format(case))
    system('mv ./data/cases/{}/constant/polyMesh/boundary2 ./data/cases/{}/constant/polyMesh/boundary'.format(case, case))

    return

if __name__ == '__main__':

    # Meshes
    mesh_path = './data/meshes/'
    meshes = listdir(mesh_path)

    # Loop
    for mesh in tqdm(meshes):
        
        AoA_list = [0.5 + 0.25 * random(), 1.0 + 0.25 * random(), 1.5 + 0.25 * random(), 2.0 + 0.25 * random(), 2.5 + 0.25 * random(), 3.0 + 0.25 * random(), 3.5 + 0.25 * random(), 4.0 + 0.25 * random(), 4.5 + 0.25 * random(), 5.0 + 0.25 * random(), 5.5 + 0.25 * random(), 6.0 + 0.25 * random()]
        
        # Create case folder
        system('mkdir ./data/cases/aux')

        # Copy content
        system('cp -r ./data/cases/example/* ./data/cases/aux' + '/')

        # Copy mesh
        system('cp ./data/meshes/' + mesh + ' ./data/cases/aux')

        # Create mesh
        system('./data/cases/aux' + '/makeMesh ' + mesh)

        # Boundary file
        updateBoundaryFile('aux')

        for i in range(len(AoA_list)):

            case = mesh.replace('.msh', '') + '-AoA-' + str(i)

            # Create case folder
            system('mkdir ./data/cases/' + case)

            # Copy content
            system('cp -r ./data/cases/aux/* ./data/cases/' + case + '/')

            # U file
            system('sed -i "s/angleOfAttack   0;/angleOfAttack   {};/g" ./data/cases/{}/0/U'.format(AoA_list[i], case))
        
        system('rm -r ./data/cases/aux')