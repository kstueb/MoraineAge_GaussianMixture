# Code developed by Konstanze Stuebner (kstueb@gmail.com) and Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de)
#
# Python 3, using some additional python packages. If you have not setup an environment,
# you may want to add the following packages via conda
# conda install pandas

# Examples of usage:
# "run batch_processing.py --help" to get help
# "run batch_processing.py -f ./data_compilation_0mmky.csv -o Out"
# "run batch_processing.py -f ./data_compilation_0mmky.csv -o Out -g 'B21'"

from moraine_age_calculator import *

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Moraine age calculation from CRN samples using a Gaussian fitting approach.')
parser.add_argument('-f','--fname', help="CSV file with boulder age data. "\
    "Required columns: 'group','groupName','Age','intErr'. Additional columns are ignored. "\
    "'Age' and 'intErr' are the CRN age and analytical (internal) error. "\
    "Boulder ages are grouped by 'group'. Default: './data_compilation_0mmky.csv'",
    default='./data_compilation_0mmky.csv')
parser.add_argument('-o','--outdir', help="Output directory. "\
    "If the directory already exists its content may be overwritten. Default: 'Out'",
    default='Out')
parser.add_argument('-g','--group', help="Age group (from 'group' in the CSV file). "\
    "If specified, only this age group will be calculated and plotted. ")

args = parser.parse_args()
fname = args.fname
outdir = args.outdir

#ky, age resolution of the calculations
res = 0.1 #ky

#labels of the relevant columns to be used as DataFrame keys
grp, grn, age, err = 'group', 'groupName', 'Age', 'intErr'

def format_plot(ax, title=None):
    ax.set_xlim(auto_xlim(Ages))
    ax.set_ylim((0,ax.get_ylim()[1]))
    plt.grid()
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel('Age (ky)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    return ax

def auto_xlim(Ages):
    """Return Age-limits (minAge, maxAge) for plotting.

    example:
        ax.set_xlim(auto_xlim(Ages))
    """
    minAge = np.max([Ages.min()-(Ages.max()-Ages.min()), 0])
    maxAge = Ages.max()+2*(Ages.max()-Ages.min())
    #range should be at least 30% of the mean age
    if (maxAge-minAge)<0.3*Ages.mean():
        minAge = 0.7*Ages.mean()
        maxAge = 1.3*Ages.mean()
    return minAge, maxAge

def plot_AgesErrorbars(Ages, Errors, ax):
    """Plot ages and errorbars to existing axis object."""

    sorted = np.flip(np.argsort(Ages))
    Ages = Ages[sorted]
    Errors = Errors[sorted]

    nAges = len(Ages)
    y = np.arange(1,nAges+1)*(ax.get_ylim()[1])/2/(nAges+2)
    ax.plot(Ages,y, 'k+')
    ax.errorbar(x=Ages, y=y, xerr=Errors, fmt='none', ecolor='k')
    return ax

def load_data(FileName):
    """Customized function to load data from one or several moraines or
    study areas from a single csv file. The function looks for columns
    'group','groupName','Age','intErr'; other columns are ignored.
    'group' is used to group data; 'groupName' is an optional alias.
    """

    df = pd.read_csv(FileName)
    print('\nData file:  '+FileName)
    print('{:d} data points, {:d} columns'.format(df.shape[0], df.shape[1]))
    print('{:d} data groups'.format(len(df.groupby(grp))))
    print('Columns:  ',end='')
    #columns the code will be looking for:
    column_names = ['group','groupName','Age','intErr']
    print(*df.columns, sep=', ')
    for name in column_names:
        if name not in df.columns:
            raise KeyError('Column name '+name+' not found in input file')
    return df

def get_AgesErrorsTitle(dfn):
    """Return Ages, Errors and title string (e.g., 'E22: Ladakh-4')
    from a pandas dataframe.

    example:
        Ages, Errors, title = get_AgesErrorsTitle(dfn)
    """

    #exclude data with rel.error >20%
    #dfn = dfn[dfn[err]/dfn[age]<.2]

    Ages = dfn[age].to_numpy() #1D arrays
    Errors = dfn[err].to_numpy()

    title = dfn[grp].to_numpy()[0]+': '+dfn[grn].to_numpy()[0]
    return Ages, Errors, title


def main():
    global Ages
    plt.close('all')
    df = load_data(fname)

    try:
        os.mkdir(r'./'+outdir)
    except:
        pass

    #to run a single sample group:
    #df = df[df[grp]=='C11']
    if args.group:
        if args.group in set(df.group):
            df = df[df[grp]==args.group]
        else:
            raise ValueError("Group '{:s}' was specified but does not exist in input file.".format(args.group))

    LL=[]
    for this_group, dfn in df.groupby(grp):
        print('\n{:s} ({:s}) : n={:d}'.format(this_group,dfn[grn].to_numpy()[0],dfn.shape[0]))
        Ages, Errors, title = get_AgesErrorsTitle(dfn)
        if len(Ages)>1:
            M = BoulderAges(Ages, Errors)
            kde = M.KDE()

            x = np.arange(*auto_xlim(Ages),step=res)
            result = iterate_Gaussian_fitting(x, scoreKDE(kde,x), len(Ages))
            A = Gaussian_misfit(x, scoreKDE(kde,x), result.best_fit)

            ax = plot_lmfit_model(x, result, A)
            plot_AgesErrorbars(Ages, Errors, ax)

            title = '{:s}: {:s} (n={:d}); bw={:.0f} ky'.format(this_group,dfn[grn].to_numpy()[0],dfn.shape[0],kde.bandwidth)
            format_plot(ax, title=title)

            L = '{:s},{:s},{:d},{:.0f}'.format(this_group,dfn[grn].to_numpy()[0],dfn.shape[0],kde.bandwidth)
            report = Gaussian_fitting_report(result)
            #preferred age peak
            for r in report:
                if r['Area']>=5:
                    L = '{:s},{:.0f}±{:.0f}'.format(L, r['Mu'], r['Sigma'])
                    break
            for r in report:
                print('{:.1f}±{:.1f} ky ({:.0f}%)'.format(r['Mu'], r['Sigma'], r['Area']))
                L = '{:s},{:.1f}±{:.1f} ky ({:.0f}%)'.format(L, r['Mu'], r['Sigma'], r['Area'])
            L = '{:s}\n'.format(L)
            LL.append(L)

            plt.savefig(r'./'+outdir+'/'+dfn[grp].to_numpy()[0]+'_'+dfn[grn].to_numpy()[0]+'.png')
            plt.close('all')

    if args.group:
        reportFile = open(r'./'+outdir+'/report_'+args.group+'.csv','w')
    else:
        reportFile = open(r'./'+outdir+'/report.csv','w')
    reportFile.write('Input data file: {:s}\n'.format(fname))
    reportFile.write('Age Density Function: sigma = {:s}\n'.format(M.ADFsigma))
    reportFile.write('Kernel Density Estimation: weight = {:s}\n'.format(M.KDEweight))
    reportFile.write('Bandwith optimized for each data set\n\n')
    reportFile.write('group,group name,n,bw,suggested peak,Gaussians\n')
    for L in LL:
        reportFile.write(L)
    reportFile.close()

    plt.show()

if __name__ == "__main__":
    main()
