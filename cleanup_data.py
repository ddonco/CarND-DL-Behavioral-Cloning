import csv
import os.path
import numpy as np
from os import path


def main():
    total_samples = []
    clean_samples = []
    with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            center_name = (line[0]).replace(' ','')
            total_samples.append(line)
            
            if path.exists(center_name):
                clean_samples.append(line)
    
    print(np.array(clean_samples).shape)
    print(np.array(total_samples).shape)
    with open('../data/driving_log_cleaned.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerows(clean_samples)

if __name__ == '__main__':
    main()