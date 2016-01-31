import pandas as pd
import numpy as np
import os
from os import path

def run_main():
    #grades_dir = 'C:\Users\Aubrey\Desktop\Homework #1'
    #full_csv =  'C:\Users\Aubrey\Google Drive\ECS170 2016\ps1\grades\\full.csv'
    grades_dir = 'C:\Users\Aubrey\Desktop\Homework #2'
    full_csv =  'C:\Users\Aubrey\Google Drive\ECS170 2016\ps2\grades\\full.csv'
    comments_file = 'comments.txt'
    data = pd.read_csv(full_csv)
    data = np.asarray(data)
    column_names = data[1,:]
    data = data[2:,:]
    id_length = 8
    grade_col = (column_names == 'grade').nonzero()[0]
    grade_names = column_names[grade_col:]
    for i, row in enumerate(data):
        try:
            id = row[1].strip()
        except:
            print 'Row ' + str(i) + ' couldn''t be processed - skipping'
            print row
            continue
        id = '0'*(id_length - len(id)) + id
        student_dir = row[2].strip() + ', ' + row[3].strip() + '(' + id + ')'
        student_dir_full = grades_dir + '/' + student_dir
        if not path.isdir(student_dir_full):
            print student_dir + ': doesn'' exist - skipping...'
            continue
        student_comments_file = student_dir_full + '/' + comments_file
        if path.isfile(student_comments_file):
            print student_comments_file + ': Already exists, overwriting...'
        f = open(student_comments_file, 'w')
        for i in range(grade_col, grade_col+len(grade_names)):
            problem_name = column_names[i]
            grade = row[i]
            try:
                problem_name = int(problem_name)
            except:
                pass
            f.write(str(problem_name) + ': ' + str(int(grade)) + '\n')
        f.close()
        with open(student_comments_file, 'r') as f:
            print f.read().replace('\r', '')
    pass


if __name__ == '__main__':
    run_main()