import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
import csv
import re

def extract_fibreInfo(fibre_file,all_magnets,robotFilearray):

    fibre_data = pd.read_excel(fibre_file)

    print(fibre_data)

    # Creates a list containing w lists, each of h item/s, all filled with 0
    w, h = 11, 1674
    new_array = [['0' for x in range(w)] for y in range(h)]
    print(len(new_array))

    j = 0
    for i in fibre_data['Hexabundle or sky fibre']:
        # print(i)
        new_array[j][0] = j+1
        if str(i) == 'Sky':
            new_array[j][1] = 'S'
        elif str(i) == 'H':
            new_array[j][1] = 'P'
        new_array[j][2] = new_array[j][6] = fibre_data['Bundle/plate'][j]
        if fibre_data['Fibre_number'][j] > 0:
            new_array[j][3] = round(fibre_data['Fibre_number'][j])
        else:
            new_array[j][3] = 'nan'
        if fibre_data['no.cores'][j] > 0:
            new_array[j][5] = 'Good'
            new_array[j][8] = round(fibre_data['no.cores'][j])
        else:
            new_array[j][5] = 'nan'
        new_array[j][7] = fibre_data['slitlet'][j]
        new_array[j][9] = fibre_data['ring'][j]
        new_array[j][10] = fibre_data['position on slit'][j]  #*** Might need to swap places with fibre number **

        j += 1


    for i in all_magnets:
        for j in robotFilearray[1:]:
            if i.__class__.__name__ == 'rectangular_magnet' and i.index == int(j[11]):
                for k in new_array:
                    if k[2] == j[12]:

                        ang = i.orientation
                        while ang > 180:
                            ang = ang - 360
                        while ang < -180:
                            ang = ang + 360

                        new_array[k[0]-1][4] = ang

    print(new_array)

    return fibre_data

