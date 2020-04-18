import subprocess                  
import re                          
import collections                 
import itertools as IT             
import string                      
import os                          
import time                        
from shutil import copyfile        
import math                        
import locale                      
import time                        
import sys                         
import io                                          
import matplotlib                  
import matplotlib.pyplot as plt    
from pylab import *      
import glob
from tabulate import tabulate
import pandas as pd
import string


## Make a file name that only contains safe charaters
# @param inputFilename A filename containing illegal characters
# @return A filename containing only safe characters
def makeSafeFilename(inputFilename):   
    return "".join([c for c in inputFilename if c.isalpha() or c.isdigit() or c==' ' or c=='_' or c=='-']).rstrip()

def is_number(s):                            
    try:                           
        float(s)                   
        return True                
    except ValueError:             
        pass                       
                                   
    try:                           
        import unicodedata         
        unicodedata.numeric(s)     
        return True                
    except (TypeError, ValueError):
        pass                       
                                   
    return False      

def get_experiments_text(outfile):                                                            
    exp = 1                                                                      
    experiments = []                                                             
    experiment =''                                                               
    grab_data = 0                                                                
    count_lines = 0                                                              
    final_exist = 0                                                              
                                                                                 
    # Start by checking to see if 'Final Variable Definitions' exists in the file
    with open(outfile) as f:                                                     
        for line in f:                                                           
            if 'Final Variable Definitions' in line:                             
                final_exist = 1                                                  
    found_final = 0                                                              
    if final_exist == 0:                                                         
        found_final = 1                                                          
    with open(outfile) as f:                                                     
        for line in f:                                                           
            if 'Final Variable Definitions' in line:                             
                found_final = 1                                                  
                continue                                                         
            if found_final == 1:                                                 
                if 'Experiment '+str(exp) in line: #Found next experiment        
                    grab_data = 1                                                
                if grab_data == 1:                                               
                    if '------' in line:                                         
                        count_lines +=1                                          
                    if count_lines == 2:                                         
                        grab_data = 0                                            
                        count_lines = 0                                          
                        experiments.append(experiment)                           
                        experiment = ''                                          
                        exp += 1                                                 
                        continue                                                 
                    experiment += line                                           
                                                                                 
    return experiments   

def parse_experiment(experiment_text):                                                                                                                       
    s = io.StringIO(experiment_text)                                                                 
    col_start = [0]                                                                                  
    col_end = [6]                                                                                    
    grab_second = 0                                                                                  
    ncol = 0                                                                                         
    data_lines = []                                                                                  
    intro_text = []                                                                                  
    start_data=0                                                                                     
    header_cols = []                                                                                 
    for i, line in enumerate(s):                                                                     
        if start_data == 0 and 'Pres ' not in line:                                                  
            intro_text.append(line)                                                                  
        if start_data > 0:                                                                           
            data_lines.append(line)                                                                  
            continue                                                                                 
        if grab_second == 1:                                                                         
            subheader_line = line                                                                    
            intro_text.pop()                                                                         
            grab_second = 0                                                                          
        if 'Pres ' in line:                                                                          
            header_line = line                                                                       
            header_idx = i                                                                           
            grab_second = 1                                                                          
        if '-----' in line:                                                                          
            start_data = 1                                                                           
            ncol = len(line.split())                                                                 
            intro_text.pop()                                                                         
            # Map the start end end columns of each                                                  
            look_for = '-'                                                                           
            for i, char in enumerate(line[col_end[-1]:-1]):                                          
                if char == look_for:                                                                 
                    if look_for == '-':                                                              
                        col_start.append(col_end[0]+i)                                               
                        look_for = ' '                                                               
                        continue                                                                     
                    else:                                                                            
                        col_end.append(col_end[0]+i)                                                 
                        look_for = '-'                                                               
                        continue                                                                     
    col_end.append(len(line))                                                                        
                                                                                                     
    # Grab Headers for each column                                                                   
    headers = []                                                                                     
    for col in range(len(col_start)):                                                                
        headers.append(header_line[col_start[col]:col_end[col]].strip())                             
                                                                                                     
    # And sub headings                                                                               
    subheaders = []                                                                                  
    for col in range(ncol):                                                                          
        subhead = subheader_line[col_start[col]:col_end[col]].split()                                
        header_cols.append(max(len(subhead),1))                                                      
        subheaders.append(subhead)                                                                   
                                                                                                     
    # Now grab column values                                                                         
    data = []                                                                                        
    for line in data_lines:                                                                          
        data_line = []                                                                               
        for col in range(ncol):                                                                      
            if len(subheaders[col]) == 1:                                                            
                data_line.append(line[col_start[col]:col_end[col]].split())                          
            else:                                                                                    
                half = int((col_start[col]+col_end[col])/2)                                          
                data_line.append([line[col_start[col]:half].strip(),line[half:col_end[col]].strip()])
        data.append(data_line)                                                                       
    return[intro_text,headers,subheaders,data, header_cols]   
    
def extract(plotvars,headers,subheaders,data,header_cols):                                       
    yvar, xvar = plotvars                                 
    yidx = headers.index(yvar)                            
    xidx = headers.index(xvar)                            
                                                          
    # Get all x values                                    
    xmaster = []                                          
    #xcol = header_cols[xidx]                             
    xcol = sum(header_cols[0:xidx])-1                     
    xcol = xidx                                           
                                                          
    # Get y-values                                        
    y_exp = []                                            
    y_cal = []                                            
    x_exp = []                                            
    x_cal = []                                            
    exp_col = sum(header_cols[0:yidx])                    
    cal_col = sum(header_cols[0:yidx])                    
                                                          
    for entry in data:                                    
        if is_number(entry[yidx][0]):                     
            y_exp.append(float(entry[yidx][0]))           
            x_exp.append(float(entry[xcol][0]))           
        if is_number(entry[yidx][1]):                     
            y_cal.append(float(entry[yidx][1]))           
            x_cal.append(float(entry[xcol][0]))           
                                                          
    return [[x_exp,y_exp], [x_cal,y_cal]]  
    

# Start script
# Find directory in which the Python script is   
script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir = os.path.split(script_path)[0] #i.e. /path/to/dir/

# .. and change to it
os.chdir(script_dir)
input_files = []
for file in glob.glob("*.[Oo][Uu][Tt]"): # Grab a list of all the *.out files in the ditrectory
    input_files.append(file)
if len(input_files) == 0:
    print( 'No .out files exist in this directory - Terminating script')
    sys.exit()

print(' ')
table = []
header=['Index', 'File Name']  # Print list of options to select from
for i in range(len(input_files)):
    table.append([i,input_files[i]])
print(tabulate(table,headers=header))    
print(' ')
file_idx = int(input('Please choose index of out file to parse (0 - '+str(len(input_files)-1)+') :'))

if file_idx not in [i for i in range(0, len(input_files))]:
    print(' ')
    print( 'Index entered outside range permitted - Terminating script')
    sys.exit()

outfile = input_files[file_idx]
    
xml_files = []
for file in glob.glob("*.[Xx][Mm][Ll]"): # Grab a list of all the *.xml files in the ditrectory
    xml_files.append(file)
if len(xml_files) == 0:
    print( 'No .xml files exist in this directory - Terminating script')
    sys.exit()

print(' ')
table = []
header=['Index', 'File Name']  # Print list of options to select from
for i in range(len(xml_files)):
    table.append([i,xml_files[i]])
print(tabulate(table,headers=header))    
print(' ')
xml_idx = int(input('Please choose index of xml plot file to use (0 - '+str(len(xml_files)-1)+') :'))

if xml_idx not in [i for i in range(0, len(xml_files))]:
    print(' ')
    print( 'Index entered outside range permitted - Terminating script')
    sys.exit()

def parse_xml(textblock):   
    df = pd.DataFrame()
    
    df['EXP'] = 0
    df['TITLE'] = 0
    df['X'] = 0
    df['Y'] = 0
    j=0
    master_title = textblock[0].split("'")[1]                                
    for i, line in enumerate(textblock):
        if i == 0:
            continue
        line = line.replace('<','')
        line = line.replace('>','')
        parsed = line.split(',')
        df.loc[j] = [0,0,0,0]
        for item in parsed:
            item = item.upper()
            if 'SUBPLOT' in item:
                continue
            entries = item.split('=')
            col = entries[0].strip()
            val = entries[1].strip() 
            val = val.replace("'",'')  
            val = val.replace('"','')
            df.iloc[j, df.columns.get_loc(col)] = val
        j += 1
    return [df, master_title]      
 

# Get xml plot definitions
xml_in = xml_files[xml_idx]  
samples_tested = []
master_titles = []
copy_plot = False
plots = []
# Grab plots to a list
with open(xml_in) as f:                                                     
        for line in f:    
            if ';' in line: # Remove everything to the right of a comment
                comment_idx = line.find(';')    
                if comment_idx == 0:
                    continue                                                  
                else:
                    line = line[0:comment_idx]
            if line.strip() == '':
                continue     
            if "PLOTTITLE" in line.upper() and "=" in line: # Found new plot
                copy_plot = True
                plot = []
            if "/plot" in line:
                plots.append(plot)
                copy_plot = False
                continue
            if copy_plot:
                plot.append(line)
                    
print('\n\nXML file read\n' )
print(str(len(plots))+' plot pages to create\n\n')           
experiment_text = get_experiments_text(outfile)   
experiments = get_experiments_text(outfile)

for z, plots in enumerate(plots):   
    df, master_title = parse_xml(plots)                                                                   
    plttitle = master_title
    numplots = len(df)                                                                                                         
                                                                                                                                           
    nx = 2                                                                                                                                 
    ny = int(numplots/2)                                                                                                                   
    if ny != numplots/2:                                                                                                                   
        ny += 1                                                                                                                            
                                                                                                                                           
    dxs = 8.0                                                                                                                              
    dys = 5.5                                                                                                                              
                                                                                                                                         
    fig, ax = plt.subplots(ny, nx, sharey = False, figsize=(dxs*nx, dys*ny), squeeze=False ) # squeeze = False ensures 2 x 1 does not get converted to 1D
                                                   
    i = 0                                                                                                                                  
    j = 0                                                                                                                                  
    fig.suptitle(plttitle, size=20, y=0.98)                                                                                                   
    if ny > 1:
        fig.subplots_adjust(top=0.95, bottom = 0.05, left = 0.05, right = 0.95)   
    else:
        fig.subplots_adjust(top=0.85, bottom = 0.15, left = 0.05, right = 0.95)   
    
    print('Plotting: ',plttitle)                                                          
                                                                                                                                           
    for index, row in df.iterrows():

        plt_title = row['TITLE']
        exp_num = int(row['EXP'])
        plotvars = [row['Y'],row['X']]

        experiment_text = experiments[exp_num-1]                                                                                           
        intro_text,headers,subheaders,data, header_cols = parse_experiment(experiment_text)                                                
        headers = [x.upper() for x in headers]
        yvar, xvar = plotvars                                                                                                   
        xidx = headers.index(xvar)                                                                                                         
                                                                                                                                           
        x_units = subheaders[xidx][0]                                                                                                      
                                                                                                                                           
        exp_plot, cal_plot = extract(plotvars,headers,subheaders,data,header_cols)                                                         
                                                                                                                                           
        x1, y1 = exp_plot
        x2, y2 = cal_plot

        ax[i,j].plot(exp_plot[0], exp_plot[1], marker='o', label='Exp', markerfacecolor='none', markeredgewidth=1.5, markersize=9, color="red", linewidth=0)  
        ax[i,j].plot(cal_plot[0], cal_plot[1], label='Calc', color='blue', linewidth=2)  
                                                                                                                                       
        ax[i,j].grid(True)                                                                                                                 
        ax[i,j].set_xlabel(plotvars[1]+' '+x_units)                                                                                        
        ax[i,j].set_ylabel(plotvars[0])                                                                                                    
        ax[i,j].legend()                                                                                                                   
        ax[i,j].title.set_text(plt_title)      
                                                                                                                                            
        if j < nx-1:                                                                                                                       
            j+=1                                                                                                                           
            continue                                                                                                                       
        else:                                                                                                                              
            j = 0                                                                                                                          
            i += 1                                                                                                                         
            continue                                                                                                                       

    if numplots/2 != int(numplots/2):                                                                                                      
        fig.delaxes(ax[i,j])                                                                                                               
    
    outplt = makeSafeFilename(master_title)+'.pdf'
    fig.savefig(outplt)
    plt.close(fig)
