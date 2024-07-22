#step 4
import pandas as pd
import os
import datetime
import matplotlib as mplt; mplt.use('SVG',force=True)
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram,linkage
import scipy.sparse as sp
from fastdtw import fastdtw as fastdtw_lib
import httpimport
import numpy as np
import math
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr,isinstalled
from sktime import distances
from tslearn import metrics
from pandas import DataFrame
#from sdtw import SoftDTW
#from sdtw.distance import SquaredEuclidean
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
#rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings
#with httpimport.github_repo(username='maxinehehe',repo='MPDist',ref='main'):
#    import MPDist as git_MPDist

#with httpimport.github_repo(username='djallelDILMI',repo='IMs-DTW',ref='master'):
#    import imsdtw as git_imsdtw

#with httpimport.github_repo(username='google-research',repo='soft-dtw-divergences',ref='master'):
#    import sdtw_div.numba_ops as git_softdtw_divergences

#with httpimport.github_repo(username='MikolajSzafraniecUPDS',repo='shapedtw-python',ref='master'):
#    import shapedtw.shapedtw as git_shapedtw
#    from shapedtw.shapeDescriptors import SlopeDescriptor, PAADescriptor, CompoundDescriptor
script_directory = os.getcwd()
processed_data_path = os.path.join(script_directory,'processed_data')
resultado_funciones_path = os.path.join(processed_data_path,'resultado_funciones')
def folder(start_month,end_month,filename):

    # Read the Excel file
    excel_name = 'casos.csv'
    excel_file = os.path.join(script_directory,excel_name)
    data = pd.read_csv(excel_file)

    # Apply the filter

    # Remove repeated values
    name_data = data.drop_duplicates(subset='name')

    # Create a folder with the specified format
    os.makedirs(processed_data_path,exist_ok=True)
    # Manage the time ranges
    # Iterate through filtered data and create subfolders
    for year in range(2019, 2023):
      rows = []
    # Initialize an empty DataFrame to store incidence data for the year
      incidence_data = pd.DataFrame()
      for index, row in name_data.iterrows():
        # Filter the data
        department_name = row['name']
        filtered_data = data[(data['disease'] == "DENGUE") &
                              (data['classification'] == "TOTAL") &
                              (data['name'] == department_name)]

        # Convert date column to pd.date
        filtered_data = filtered_data.copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], format='%Y-%m-%d')

        # Define the start and end dates
        fixed_year = year
        start_date = pd.to_datetime(f"{year}-{start_month}-1", format='%Y-%m-%d')
        if( end_month < start_month ): 
          fixed_year = fixed_year + 1
        end_date = pd.to_datetime(f"{fixed_year}-{end_month}-31", format='%Y-%m-%d')

        # Filter the data for the specified period
        range_data = filtered_data[filtered_data['date'].between(start_date, end_date, inclusive='left')]

        # Extract the incidence columns
        incidence_columns = [col for col in range_data.columns if 'incidence' in col]
        incidence_values = range_data[incidence_columns].values.flatten()  # Get incidence values as a flat array

        # Create a row with the department name and incidence values
        row_data = [department_name] + incidence_values.tolist()
        rows.append(row_data)
      columns = ['Department'] + [f'Incidence{i+1}' for i in range(len(incidence_values))]
      incidence_data = pd.DataFrame(rows, columns=columns)

      # Save the DataFrame as a CSV file
      output_file_name = f'{filename}_{year}.csv'
      output_file_path = os.path.join(processed_data_path, output_file_name)
      incidence_data.to_csv(output_file_path, index=False)
      print(f"Saved: {output_file_path}")
start_month = 9
end_month = 8
filename = 'time_series'
folder(start_month,end_month,filename)

start_month = 9
end_month = 12
filename = 'first_trimester'
folder(start_month,end_month,filename)
#step 5 (wait 9 min)

#utils = importr('utils',suppress_messages=False)
#if(not isinstalled('TSclust')):   
#  print('TSclust not installed!')
#  utils.chooseCRANmirror(ind=7)
#  utils.install_packages('TSclust')
#  utils.install_packages('pdc')
#  utils.install_packages('cluster')
#print('APRETA CTRL C!!!!')
#TSclust = importr(name='TSclust',lib_loc='/home/cesar/R/x86_64-pc-linux-gnu-library/4.4/',suppress_messages=False)
#TSclust = importr(name='TSclust',suppress_messages=False)
#print('TSclust is installed')
#2
def euclidean_L2(tseries1,tseries2):
    return distance.euclidean(tseries1, tseries2)

#3
def cityblock(tseries1, tseries2):
    return distance.cityblock(tseries1, tseries2)

#4
def minkowski_Lp(tseries1, tseries2):
    return distance.minkowski(tseries1, tseries2)

#5
def chebyshev_Linf(tseries1, tseries2):
    return distance.chebyshev(tseries1, tseries2)

#6
def sorensen(tseries1, tseries2):
    numerator = 0.0
    denominator = 0.0
    i = 0
    for i in range( len(tseries1) ):
        numerator += abs( tseries1[i] - tseries2[i] )
        denominator += tseries1[i] - tseries2[i]
    if denominator==0.0:
      denominator = 0.0000001
    return float(numerator/denominator)

#7
def gower(tseries1, tseries2):
    numerator = 0.0
    denominator = len(tseries1)
    i = 0
    for i in range( len(tseries1) ):
        numerator += abs( tseries1[i] - tseries2[i] )
    if denominator==0.0:
      denominator = 0.0000001
    return float(numerator/denominator)

#8
def soergel(tseries1, tseries2):
    numerator = 0.0
    denominator = 0.0
    i = 0
    for i in range( len(tseries1) ):
        numerator += abs( tseries1[i] - tseries2[i] )
        denominator += max(tseries1[i],tseries2[i])
    if denominator==0.0:
      denominator = 0.0000001
    return float(numerator/denominator)

#9
def kulczynski1(tseries1, tseries2):
    numerator = 0.0
    denominator = 0.0
    i = 0
    for i in range( len(tseries1) ):
        numerator += abs( tseries1[i] - tseries2[i] )
        denominator += min(tseries1[i],tseries2[i])
    if denominator==0.0:
      denominator = 0.0000001
    return float(numerator/denominator)

#10
def canberra(tseries1, tseries2):
  return distance.canberra(tseries1,tseries2)

#11
def lorentzian(tseries1, tseries2):
    value = 0.0
    i = 0
    for i in range( len(tseries1) ):
        value += np.log(1+abs(tseries1[i]-tseries2[i]))
    return value

#12
def intersection(tseries1, tseries2):
    numerator = 0.0
    denominator = 2.0
    i = 0
    for i in range( len(tseries1) ):
        numerator += abs( tseries1[i] - tseries2[i] )
    return float(numerator/denominator)

#13
def wave_hedges(tseries1, tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range( len(tseries1) ):
        numerator = abs( tseries1[i] - tseries2[i] )
        denominator = max(tseries1[i],tseries2[i])
        if(denominator==0.0):
          denominator = 0.0000001
        value += float(numerator/denominator)
    return value

#14
def czekanowski(tseries1, tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range( len(tseries1) ):
        numerator += abs( tseries1[i] - tseries2[i] )
        denominator += tseries1[i] + tseries2[i]
    if denominator==0.0:
      denominator = 0.0000001
    return float(numerator/denominator)


#15
def motyka(tseries1, tseries2):
    numerator = 0.0
    denominator = 0.0
    i = 0
    for i in range( len(tseries1) ):
        numerator += max(tseries1[i],tseries2[i])
        denominator += tseries1[i] + tseries2[i]
    if denominator == 0.0:
      denominator = 0.0000001
    return float(numerator/denominator)

#16
def kulczynski_s(tseries1, tseries2):
    numerator = 0.0
    denominator = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator += min(tseries1[i],tseries2[i])
        denominator += abs(tseries1[i]-tseries2[i])
    if denominator == 0.0:
      denominator = 0.0000001
    return float(numerator/denominator)

#17
def ruzicka(tseries1, tseries2):
    numerator = 0.0
    denominator = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator += min(tseries1[i],tseries2[i])
        denominator += max(tseries1[i],tseries2[i])
    if denominator == 0.0:
      denominator = 0.0000001
    return float(numerator/denominator)

#18
def tanimoto(tseries1, tseries2):
    numerator = 0.0
    denominator = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator += max(tseries1[i],tseries2[i]) - min(tseries1[i],tseries2[i])
        denominator += max(tseries1[i],tseries2[i])
    if denominator == 0.0:
      denominator = 0.0000001
    return float(numerator/denominator)

#19
def inner_product(tseries1, tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        value+=tseries1[i]*tseries2[i]
    return value

#20
def harmonic_mean(tseries1, tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = tseries1[i] * tseries2[i]
        denominator = tseries1[i] + tseries2[i]
        if(denominator==0.0):
          denominator = 0.0000001
        if((numerator+denominator)==0.0):
          numerator = 0.0000001
          denominator = 0.0000001
        value += float(numerator/denominator)
    value *= 2
    return value

#21
def cosine(tseries1,tseries2):
    numerator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator += tseries1[i] * tseries2[i]
        denominator1 += tseries1[i] * tseries1[i]
        denominator2 += tseries2[i] * tseries2[i]
    denominator1 = math.sqrt(denominator1)
    denominator2 = math.sqrt(denominator2)
    if(denominator1==0.0 or denominator2==0.0):
      denominator2 = 0.0000001
      denominator1 = 1
    value = float(numerator/(denominator1*denominator2))
    return value

#22
def kumar_hassebrook(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator += tseries1[i] * tseries2[i]
        denominator += math.pow(tseries1[i],2) + math.pow(tseries2[i],2) - (tseries1[i]*tseries2[i])
    if(denominator==0.0):
      denominator = 0.0000001
    value = float(numerator/(denominator))
    return value

#23
def jaccard(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator += tseries1[i]*tseries2[i]
        denominator += math.pow(tseries1[i],2) + math.pow(tseries2[i],2) - tseries1[i]*tseries2[i]
    if(denominator==0.0):
      denominator = 0.0000001
    return float(numerator/denominator)

#24
def dice(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator += math.pow(tseries1[i]-tseries2[i],2)
        denominator += math.pow(tseries1[i],2) + math.pow(tseries2[i],2)
    if(denominator==0.0):
      denominator = 0.0000001
    return float(numerator/denominator)

#25
def fidelity(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        value += math.sqrt(tseries1[i]*tseries2[i])
    return value

#26
def bhattacharyya(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        value += math.sqrt(tseries1[i]*tseries2[i])
    value = - np.log(value)
    return value

#27
def hellinger(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        value += math.sqrt(tseries1[i]*tseries2[i])
    value = 2 * math.sqrt(abs(1-value))
    return value

#28
def matusita(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        value += math.sqrt(tseries1[i]*tseries2[i])
    value = math.sqrt(abs(2-2*value))
    return value

#29
def squared_chord(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        value += np.power(math.sqrt(tseries1[i])-math.sqrt(tseries2[i]),2)
    return value

#30
def squared_euclidean(tseries1,tseries2):
    return distance.sqeuclidean(tseries1,tseries2)

#31
def pearson_x2(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = np.power(tseries1[i]-tseries2[i],2)
        denominator = tseries2[i]
        if(denominator==0.0):
          denominator = 0.0000001
        value += float(numerator/denominator)
    return value

#32
def neyman_x2(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = np.power(tseries1[i]-tseries2[i],2)
        denominator = tseries1[i]
        if(denominator==0.0):
          denominator = 0.0000001
        value += float(numerator/denominator)
    return value

#33
def squared_x2(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = np.power(tseries1[i]-tseries2[i],2)
        denominator = (tseries2[i]+tseries1[i])
        if (denominator==0.0):
          denominator = 0.0000001
        value += float(numerator/denominator)
    return value

#34
def probabilistic_symmetric_x2(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = np.power(tseries1[i]-tseries2[i],2)
        denominator = (tseries2[i]+tseries1[i])
        if (denominator==0.0):
          denominator = 0.0000001
        value += float(numerator/denominator)
    value *= 2
    return value

#35
def divergence(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = np.power(tseries1[i]-tseries2[i],2)
        denominator = np.power(tseries2[i]+tseries1[i],2)
        if (denominator==0.0):
          denominator = 0.0000001
        value += float(numerator/denominator)
    value *= 2
    return value

#36
def clark(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = abs(tseries1[i]-tseries2[i])
        denominator = tseries2[i]+tseries1[i]
        if (denominator==0.0):
          denominator = 0.0000001
        value += np.power(float(numerator/denominator),2)
    value = math.sqrt(value)
    return value

#37
def additive_symmetric_x2(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = np.power(tseries1[i]-tseries2[i],2)*(tseries1[i]+tseries2[i])
        denominator = tseries2[i]*tseries1[i]
        if (denominator==0.0):
          denominator = 0.0000001
        value += np.power(float(numerator/denominator),2)
    value = math.sqrt(value)
    return value

#38
def kullback_leibler(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        if(tseries2[i]==0.0):
          tseries2[i] = 0.0000001
        if(float(tseries1[i]==0.0)):
          tseries1[i] = 0.0000001
        value += (tseries1[i]*np.log(float(tseries1[i]/tseries2[i])))
    return value

#39
def jeffreys(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        if(tseries2[i]==0.0):
          tseries2[i] = 0.0000001
        value += ((tseries1[i]-tseries2[i])*np.log(float(tseries1[i]/tseries2[i])))
    return value

#40
def k_divergence(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        if(tseries2[i]==0.0):
          tseries2[i] = 0.0000001
        value += (tseries1[i]*np.log(float((2*tseries1[i])/(tseries1[i]+tseries2[i]))))
    return value

#41
def topsoe(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        if(tseries2[i]+tseries1[i]==0.0):
          tseries2[i] = 0.0000001
        value += (tseries1[i]*np.log(float((2*tseries1[i])/(tseries1[i]+tseries2[i]))))
        value += (tseries2[i]*np.log(float((2*tseries2[i])/(tseries1[i]+tseries2[i]))))
    return value

#42
def jensen_shannon(tseries1,tseries2):
    return distance.jensenshannon(tseries1,tseries2)

#43
def jensen_difference(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        value += ((tseries1[i]*np.log(tseries1[i])+tseries2[i]*np.log(tseries2[i]))/2)
        value -= ((tseries1[i]+tseries2[i])/2)*np.log((tseries1[i]+tseries2[i])/2)
    return value

#44
def taneja(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        if(tseries1[i]*tseries2[i]==0):
          tseries2[i] = 0.0000001
          tseries1[i] = 1
        value += ((tseries1[i]+tseries2[i])/2)*np.log((tseries1[i]+tseries2[i])/(2*math.sqrt(tseries1[i]*tseries2[i])))
    return value

#45
def kumar_johnson(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = np.power(tseries1[i]-tseries2[i],2)
        denominator = tseries2[i]+tseries1[i]
        if (denominator==0.0):
          denominator = 0.0000001
        value += np.power(float(numerator/denominator),2)
    value = math.sqrt(value)
    return value

#46
def avg_l1_linf(tseries1,tseries2):
    numerator = 0.0
    denominator = 2
    value = 0.0
    for _ in range(len(tseries1)):
        numerator += distance.cityblock(tseries1,tseries2)+distance.chebyshev(tseries1,tseries2)
    value = float(numerator/denominator)
    return value

#47
def d_emanom1(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = abs(tseries1[i]-tseries2[i])
        denominator = min(tseries1[i],tseries2[i])
        if (denominator==0.0):
          denominator = 0.0000001
        value += float(numerator/denominator)
    return value

#48
def d_emanom2(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = np.power(tseries1[i]-tseries2[i],2)
        denominator = np.power(min(tseries1[i],tseries2[i]),2)
        if (denominator==0.0):
          denominator = 0.0000001
        value += float(numerator/denominator)
    return value

#49
def d_emanom3(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = np.power(tseries1[i]-tseries2[i],2)
        denominator = min(tseries1[i],tseries2[i])
        if (denominator==0.0):
          denominator = 0.0000001
        value += float(numerator/denominator)
    return value

#50
def d_emanom4(tseries1,tseries2):
    numerator = 0.0
    denominator = 0.0
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator = np.power(tseries1[i]-tseries2[i],2)
        denominator = max(tseries1[i],tseries2[i])
        if (denominator==0.0):
          denominator = 0.0000001
        value += float(numerator/denominator)
    return value

#51
def d_emanom5(tseries1,tseries2):
    numerator1 = 0.0
    denominator1 = 0.0
    value1 = 0.0
    numerator2 = 0.0
    denominator2 = 0.0
    value2 = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator1 = np.power(tseries1[i]-tseries2[i],2)
        denominator1 = tseries1[i]
        if (denominator1==0.0):
          denominator1 = 0.0000001
        value1 += float(numerator1/denominator1)
    for i in range(len(tseries1)):
        numerator2 = np.power(tseries1[i]-tseries2[i],2)
        denominator2 = tseries2[i]
        if (denominator2==0.0):
          denominator2 = 0.0000001
        value2 += float(numerator2/denominator2)
    return max(value1,value2)

#52
def d_emanom6(tseries1,tseries2):
    numerator1 = 0.0
    denominator1 = 0.0
    value1 = 0.0
    numerator2 = 0.0
    denominator2 = 0.0
    value2 = 0.0
    i = 0
    for i in range(len(tseries1)):
        numerator1 = np.power(tseries1[i]-tseries2[i],2)
        denominator1 = tseries1[i]
        if (denominator1==0.0):
          denominator1 = 0.0000001
        value1 += float(numerator1/denominator1)
    for i in range(len(tseries1)):
        numerator2 = np.power(tseries1[i]-tseries2[i],2)
        denominator2 = tseries2[i]
        if (denominator2==0.0):
          denominator2 = 0.0000001
        value2 += float(numerator2/denominator2)
    return min(value1,value2)


#53
#Extract from: https://github.com/maxinehehe/MPDist
#MPdist distance is a measure, not a metric. In particular, it does not obey the
#   triangular inequality. The lack of the triangular inequality property is potentially
#   worrisome for two reasons:
#       • Many speedup techniques for query-by-content, clustering, anomaly detection etc.,
#           implicitly or explicitly exploit the triangular inequality to prune the search space;
#           which becomes tenable for large datasets [3].
#       • Without the triangular inequality property, one can produce distance evaluations
#           that defy human intuitions. For example, claiming that A and B are similar, and A
#           and C are similar; but, B and C are very dissimilar.
#def mpdist(tseries1,tseries2):
#    mpdist = git_MPDist.MPDist()
    #If the length of L is a significant fraction of n, then the
    #   time complexity grows to O((𝑛 − 𝐿 + 1) × 𝑛) . In the limit, when L = n, this
    #   degenerates to the special case of the Euclidean distance between the two time series,
    #   which takes O(𝑛).
    #We begin by fixing ... L to n/2 (typical values used in Section 4.1)
    #Here the ultra-liberal invariance of MPdist means that it has a difficult time telling the
    #   two classes apart when L is very short. However, as the length of L increases, every
    #   subsequence of A will include some bumps and some saw-teeth, in that order, allowing
    #   the error-rate to fall to zero.
#    L = len(tseries1)/2
#    return mpdist.distDTW(tseries1,tseries2,L)

#54
def braycurtis(tseries1,tseries2):
    return distance.braycurtis(tseries1,tseries2)

#55
def correlation(tseries1,tseries2):
    return distance.correlation(tseries1,tseries2)

#58-64
#The following distances operate over binary arrays:
#   • Hamming
#   • Kulczynski1
#   • Rogers-Tanimoto
#   • Russel-Rao
#   • Sokal-Michener
#   • Sokal-Sneath
#   • Yule

#Extracted from: https://github.com/KishoreKaushal/DerivativeDynamicTimeWarping
class DerivativeDTW(object):
    '''
        self.time_series : Numpy array where each row must be a time series.
        self.distanceMatrix : for computing the distance between each time series,
                        which is essential for clustering the time series in later stages.
    '''
    def __init__(self , time_series, filt=True, v=1):
        assert(isinstance(time_series , np.ndarray))
        self.filt= filt
        self.v = v
        self.time_series = time_series
        n , m = time_series.shape
        self.distanceMatrix = np.zeros((n,n) , dtype=np.float64)

        self.DerivativeMatrix = np.zeros_like(self.time_series , dtype=np.float64)

        self.isDerivativeMatrixCreated = False
        self.isDistanceMatrixCreated = False
        self.computeDerivativeMatrix()
        self.computeDistanceMatrix()

    def computeDerivativeMatrix(self):
        if (self.isDerivativeMatrixCreated == True):
            return

        # iterate through all the time series in the list
        for i in range(self.time_series.shape[0]):
            # for each time_series calculate the Derivative
            q = self.time_series[i , :]
            for j in range(1 , q.shape[0]-1):
                self.DerivativeMatrix[i , j] = ((q[j] - q[j-1]) + ((q[j+1] - q[j-1])/2))/2

            # set the boundary derivatives
            self.DerivativeMatrix[i , 0] = self.DerivativeMatrix[i , 1]
            self.DerivativeMatrix[i,-1] = self.DerivativeMatrix[i , -2]

        #if self.filt == True:
        #    self.DerivativeMatrix = gaussian_filter1d(self.DerivativeMatrix , self.v)
        self.isDerivativeMatrixCreated = True

    def computeDistanceMatrix(self):
        if self.isDistanceMatrixCreated == True:
            return

        if self.isDerivativeMatrixCreated == False:
            self.computeDerivativeMatrix()

        for i in range(self.time_series.shape[0]):
            for j in range(i):
                self.distanceMatrix[i,j] = self.distanceMatrix[j,i] = self.computeDtw(q=self.DerivativeMatrix[i, :],
                                                                                      c=self.DerivativeMatrix[j, :])
        self.dtwMatCreated = True


    @staticmethod
    def computeDtw(q , c):
        assert(q.shape == c.shape)
        m = q.shape[0]
        Y = np.zeros((m+1,m+1) , dtype=np.float)

        Y[: , 0] = np.inf
        Y[0 , :] = np.inf
        Y[0,0] = 0

        for i in range(m):
            for j in range(m):
                yi , yj = i+1 , j+1
                Y[yi , yj] = abs(q[i] - c[j]) + min(Y[yi-1 , yj-1] , Y[yi-1 , yj] , Y[yi , yj-1])
        return Y[m,m]

    def affinity(self, i,j):
        return self.distanceMatrix[i,j]

#65
#Returns a distance matrix
def derivative_dynamic_time_warping(tseries1,tseries2):
    ddtw = DerivativeDTW(np.array([tseries1,tseries2]),filt=False,v=1)
    return ddtw.distanceMatrix

def diff(tseries):
    i = 0
    for i in range(len(tseries)-1):
        tseries[i+1] -= tseries[i]
    return tseries

#67
#we can choose points (a,b)on any continuous line between the points (0,1)and(1,0).
#For example, it can be a straight line or a quarter of a circle:
def dissim_DTW_LCSS1(tseries1, tseries2):
    alpha = 0.5
    a = math.cos(alpha)
    b = math.sin(alpha)
    value = a*metrics.dtw(tseries1,tseries2)
    value += b*metrics.lcss(tseries1,tseries2)
    return value

#68
#Parametric Derivative Dynamic Time Warping has the same formula as the one provided in (Górecki, 2018)
def dissim_DTW_LCSS2(tseries1,tseries2):
    alpha = 0.5
    a = math.cos(alpha)
    b = math.sin(alpha)
    value = a*metrics.dtw(tseries1,tseries2)
    value += b*metrics.dtw(diff(tseries1),diff(tseries2))
    return value

#69??
def dissim_DTW_LCSS3(tseries1,tseries2):
    alpha = 0.5
    a = math.cos(alpha)
    b = math.sin(alpha)
    c = math.tan(alpha)
    value = a*metrics.dtw(tseries1,tseries2)
    value += b*metrics.dtw(diff(tseries1),diff(tseries2))
    value += c*metrics.lcss(tseries1,tseries2)
    return value

#LB_Keogh computes a lower bound for the DTW distance

#70
def time_warp_edit_distance(tseries1,tseries2):
    return distances.twe_distance(tseries1,tseries2)

#71
#def prediction_based(tseries1,tseries2):
#    return TSclust.diss.PRED(tseries1,tseries2)

#72
#Spatial Assembling Distance
#General match: https://koasas.kaist.ac.kr/bitstream/10203/12228/1/Moon2002.pdf
#SpADe: https://www.comp.nus.edu.sg/~ooibc/SpADe.pdf
# Calculate the distance of two local patterns

#def global_alignment_kernels(tseries1,tseries2):

#def derivative_time_series_segment_approximation(tseries1,tseries2):

#revisar: https://www.researchgate.net/publication/202950321_Unsupervised_Learning_Motion_Models_Using_Dynamic_Time_Warping
def value_derivative(tseries):
    derivative_tseries = np.zeros(len(tseries))
    derivative_tseries[0] = tseries[0]
    i=0
    if(i>0):
        derivative_tseries = tseries[i] - tseries[i-1]
    return derivative_tseries

#75
def value_derivative_dynamic_time_warping(tseries1,tseries2):
    i = 0
    value = 0
    distance_tseries = 0
    derivative_tseries1 = value_derivative(tseries1)
    derivative_tseries2 = value_derivative(tseries2)
    for i in range(len(tseries1)):
        value1 = np.array(tseries1[i])
        value2 = np.array(tseries2[i])
        distance_tseries = math.sqrt(abs(math.pow(tseries1[i],2)-math.pow(tseries2[i],2)))
        distance_tseries *= math.pow(derivative_tseries1[i]-derivative_tseries2[i],2)
        value += distance_tseries
    return value


#76
#Paper:https://www.researchgate.net/publication/221259602_Facial_Dynamics_in_Biometric_Identification
# def weighted_derivative(tseries):
#     derivative_tseries = np.zeros(len(tseries)-1)
#     derivative_tseries[0] = tseries[0]
#     derivative_tseries[len(tseries)-1]=tseries[len(tseries)-1]
#     i=0
#     if(i>0):
#         derivative_tseries = (float)(((tseries[i] - tseries[i-1]) + ((tseries[i+1] - tseries[i-1])/2))/2)
#     return derivative_tseries
# def weighted_dtw(tseries1,tseries2):
#     #Thus, when choosing w1and w2, we should take into account both the Signal-to-Noise ratio and the difference
#     #   of magnitudes between thesignal and its derivatives.
#     #In this study, we choose w0=1, w1=w2=2.
#     weight0 = 1
#     weight1 = 2
#     weight2 = 2
#     derivative1_tseries1 = weighted_derivative(tseries1)
#     derivative1_tseries2 = weighted_derivative(tseries2)
#     derivative2_tseries1 = weighted_derivative(derivative1_tseries1)
#     derivative2_tseries2 = weighted_derivative(derivative1_tseries2)
#     i = 0
#     value = 0
#     distance_tseries = 0
#     for i in range(len(tseries1)):
#         distance_tseries = weight0 * distance.euclidean(tseries1,tseries2)
#         distance_tseries += weight1 * ( derivative1_tseries1[i] - derivative1_tseries2[i] )
#         distance_tseries += weight2 * ( derivative2_tseries1[i] - derivative2_tseries2[i] )
#         value += distance_tseries
#     return value

#77
def DTW(tseries1,tseries2):
    return metrics.dtw(tseries1,tseries2)

#LPDistance: A value in "euclidean", "manhattan", "infnorm", "minkowski".

#79
def acf(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.ACF(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def acf_lpc_ceps(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.AR.LPC.CEPS(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def ar_mah(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.AR.MAH(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def ar_pic(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.AR.PIC(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def cdm(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.CDM(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def cid(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.CID(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def cor(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.COR(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def cort(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.CORT(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def frechet(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.FRECHET(ts1, ts2)
  }
  ''')
  robjects.r('''
    suppress_output <- function(expr) {
        capture.output(result <- expr, file = "/dev/null")
        return(result)
    }
''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def int_per(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.INT.PER(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def mindist_sax(tseries1,tseries2):
  #From: https://www.researchgate.net/publication/221598030_A_Symbolic_Representation_of_Time_Series_with_Implications_for_Streaming_Algorithms
  #The results also suggest that the parameters are not too critical; 
  # an alphabet size in the range of 5 to 8 seems to be a good choice
  #Since each time series has length 52, we will use w = 13
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2,w) {
    
    diss.MINDIST.SAX(ts1, ts2,w)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2),int(len(tseries1)/4))

  return dissimilarity[0]

#79
def ncd(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.NCD(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def pdc(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.PDC(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def per(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.PER(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def pred(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2,h) {
    diss.PRED(ts1, ts2,h)
  }
  ''')
  h = 6
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2),h)

  return dissimilarity[0]

#79
def spec_glk(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.SPEC.GLK(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def spec_isd(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.SPEC.ISD(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]

#79
def spec_llr(tseries1,tseries2):
  robjects.r('''
  library(TSclust)
  ''')
  diss_tsclust = robjects.r('''
  function(ts1, ts2) {
    diss.SPEC.LLR(ts1, ts2)
  }
  ''')
  dissimilarity = diss_tsclust(robjects.FloatVector(tseries1), robjects.FloatVector(tseries2))

  return dissimilarity[0]





#107
#Paper: https://www.researchgate.net/publication/330214889_A_Weighted_DTW_Approach_for_Similarity_Matching_over_Uncertain_Time_Series
def uncertain_weighted_dtw(tseries1, tseries2):
    """
    Weighted Dynamic Time Warping for two time series tseries1 and tseries2 with weight function w.
    """
    n, m = len(tseries1), len(tseries2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = math.sqrt(abs(math.pow(tseries1[i-1],2)-math.pow(tseries2[j-1],2)))
            weight = (1 / (1 + (i - j) ** 2))
            dtw_matrix[i, j] = cost * weight + min(
                dtw_matrix[i - 1, j],      # Insertion
                dtw_matrix[i, j - 1],      # Deletion
                dtw_matrix[i - 1, j - 1]   # Match
            )

    return dtw_matrix[n, m]

#Paper: https://core.ac.uk/download/pdf/29107722.pdf
#From: https://athitsos.utasites.cloud/publications/stefan_tkde2012_preprint.pdf
#In our expperiments, MSM produced competitive error rates while
#   considering only five widely-spaced values (differing by
#   factors of 10) for c. Considering only five widely-spaced
#   values demonstrates that no careful finetuning of c was
#   needed to obtain good results
def C(ts1_value1,ts1_value2,ts2_value1):
    c = 0
    if(
        (ts1_value2 <= ts1_value1 and ts1_value1 <= ts2_value1)
       or
        (ts1_value2 >= ts1_value1 and ts1_value1 >= ts2_value1)
       ):
        c = 0.5
    else:
        c = 0.5 + min(abs(ts1_value1-ts1_value2),abs(ts1_value1-ts2_value1))
    return c

#108
def move_split_merge(tseries1,tseries2):
    n,m = len(tseries1), len(tseries2)
    msm_matrix = np.zeros((n,m))
    msm_matrix[0, 0] = abs( tseries1[0] - tseries2[0] )
    for i in range(1,n):
        msm_matrix[i,0] = msm_matrix[i-1,0] + C(tseries1[i],tseries1[i-1],tseries2[0])
    for j in range(1,m):
        msm_matrix[0,j] = msm_matrix[0,j-1] + C(tseries2[j],tseries1[0],tseries2[j-1])
    for i in range(1,m):
        for j in range(1,n):
            temp_a = msm_matrix[i-1,j-1] + abs(tseries1[i]-tseries2[j])
            temp_b = msm_matrix[i-1,j] + C(tseries1[i],tseries1[i-1],tseries2[j])
            temp_c = msm_matrix[i,j-1] + C(tseries2[j],tseries1[i],tseries2[j-1])
            msm_matrix[i,j] = min(temp_a,temp_b,temp_c)
    return msm_matrix[m-1,n-1]

#Paper: https://jcst.ict.ac.cn/EN/10.1007/s11390-015-1565-7
#Linear Combination Method
#A common method to integrate different types of
#   similarity matrixes is a linear combination method
#   (LCM)[36]. We first transfer all the distance matrixes
#   into similarity matrixes and normalize them. Then we
#   combine them with equal weights linearly

#Paper: https://www.informatica.si/index.php/informatica/article/view/98/91
#Ortogonal Wavelet Transform
#In this paper we propose an unsupervised feature extraction
#   algorithm using orthogonal wavelet transform for automatically choosing
#   the dimensionality of features. The feature extraction algorithm selects the
#   feature dimensionality by leveraging two conflicting requirements, i.e.,
#   lower dimensionality and lower sum of squared errors between the features
#   and the original time series.

#Paper: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=22fb0eb0a8bf9899273315b0e4adff9dfdc68fcc
#Discrete Wavelet Transform
#We now introduce another family of transformations,
#   the Discrte Wavelet Transform(DWT) [10], that
#   performs similar properties as DFT in time-series
#   signals.


#112

def fastdtw(tseries1,tseries2):
    diss,_=fastdtw_lib(tseries1,tseries2)
    return diss
def downsample(ts, factor):
    return ts[::factor]
def upsample(ts, factor):
    return np.repeat(ts, factor)
def graph_trend_filtering(ts, lambda1=0.1, lambda2=0.1):
    # Create graph difference operators
    n = len(ts)
    D1 = sp.diags([1, -1], [0, 1], shape=(n-1, n))
    D2 = sp.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n))
    L = lambda1 * D1.T @ D1 + lambda2 * D2.T @ D2
    # Solve for the trend using some optimization method
    trend = sp.linalg.spsolve(L + sp.eye(n), ts)
    return trend

#113
def robust_dtw(ts1, ts2, max_levels=3, radius=1):
    # Step 1: Initial trend filtering
    trend1 = graph_trend_filtering(ts1)
    trend2 = graph_trend_filtering(ts2)
    # Step 2: Multi-level representation
    levels = []
    ts1_levels, ts2_levels = [trend1], [trend2]
    for _ in range(1, max_levels):
        ts1_levels.append(downsample(ts1_levels[-1], 2))
        ts2_levels.append(downsample(ts2_levels[-1], 2))

    # Step 3: Iterative warping and trend estimation
    warp_path = None
    for level in range(max_levels-1, -1, -1):
        if warp_path is not None:
            warp_path = upsample(warp_path, 2)

        distance = fastdtw(ts1_levels[level], ts2_levels[level])

    # Step 4: Temporal graph detrending
    detrended_ts1 = ts1 - graph_trend_filtering(ts1)
    detrended_ts2 = ts2 - graph_trend_filtering(ts2)

    return distance

#114
#def IMs_DTW(tseries1,tseries2):
#    local_imsdtw = git_imsdtw.imsdtw(aggregation_step = 2, radius=0)
#    dissim,_ = local_imsdtw(tseries1,tseries2)
#    return dissim

#115
def constraint_based_dtw(tseries1,tseries2,maxDist=1):
    m = len(tseries1)
    n = len(tseries2)
    cdtw = np.full((m+1,n+1),np.inf)
    cdtw[0,0] = 0
    stop_i = 0
    stop_j = 0
    i = 1
    j = 1
    while( i < m and j < n ):
        for jj in range(1,n):
            cdtw[i,jj] = math.sqrt(abs(math.pow(tseries1[i],2)-math.pow(tseries2[jj],2)))
            cdtw[i,jj] += min(cdtw[i-1,jj],cdtw[i-1,jj-1],cdtw[i,jj-1])
            if( cdtw[i,jj]>maxDist and jj>stop_j ):
                stop_j = jj
                break
        i = i + 1
        for ii in range(1,m):
            cdtw[ii,j] = math.sqrt(abs(math.pow(tseries1[ii],2)-math.pow(tseries2[j],2)))
            cdtw[ii,j] += min(cdtw[ii-1,j],cdtw[ii-1,j-1],cdtw[ii,j-1])
            if( cdtw[ii,j]>maxDist and ii>stop_i ):
                stop_i = ii
                break
        j = j + 1
    return cdtw[m,n]

#PRCIS: https://arxiv.org/ftp/arxiv/papers/2212/2212.06146.pdf
#For single-event time series Euclidean Distance
#   and Dynamic Time Warping distance are known to be extremely
#   effective. However, for time series containing cyclical behaviors,
#   the semantic meaningfulness of such comparisons is less clear...
#   In this work we introduce PRCIS, which stands for Pattern
#   Representation Comparison in Series.

#121
#def softdtw(tseries1,tseries2):
#    D = SquaredEuclidean(tseries1,tseries2)
#    sdtw = SoftDTW(D, gamma=1.0)
#    value = sdtw.compute()
#    return value

#124
#def softdtw_divergences(tseries1,tseries2):
#    return git_softdtw_divergences.sdtw_div(tseries1,tseries2,gamma=1.0)

#126
#def shapedtw(tseries1,tseries2):
#    slope_descriptor = SlopeDescriptor(slope_window=2)
#    paa_descriptor = PAADescriptor(piecewise_aggregation_window=2)
#    compound_desc = CompoundDescriptor(
#        shape_descriptors = [slope_descriptor, paa_descriptor],
#        descriptors_weights = [2.0, 1.0]
#    )
#    univariate_results = git_shapedtw.shape_dtw(
#        x=tseries1,
#        y=tseries2,
#        subsequence_width=3,
#        shape_descriptor=compound_desc
#    )
#    return(univariate_results.distance)

#163
def vector_dynamic_time_warping(tseries1,tseries2):
  m = len(tseries1)
  n = len(tseries2)
  psi = np.zeros((m,n))
  d = np.zeros((m,n))
  i = 0
  j = 0
  for i in range(2,m-1):
    for j in range(2,n-1):
      psi[i,j] = math.acos(np.dot(tseries1[i-1:i+1],tseries2[j-1:j+1]) / np.linalg.norm(tseries1[i-1:i+1]) * np.linalg.norm(tseries2[j-1:j+1]))

  for i in range(2,m):
    d[i,2]= psi[i,2] + d[i-1,2]

  for j in range(2,m):
    d[2,j]= psi[2,j] + d[2,j-1]

  for i in range(3,m):
    for j in range(3,n):
      d[i,j]= psi[i,j] + min(d[i-1,j], min(d[i-1,j-1] , d[i,j-1]))

  return d[m-1,n-1]


#At first glance, Canberra metric given in the eqn 
#(10) [2,15] resembles Sørensen but normalizes the absolute 
#difference of the individual level. It is known to be very 
#sensitive to small changes near zero
#Bhattacharyya distance given in the eqn 
#(33), which is a value between 0 and 1, provides bounds on 
#the Bayes misclassification probability [23].
conjunto_funciones=[canberra,
                    bhattacharyya]


# Function to load and process a single CSV file
def load_and_process_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path, header=0, index_col=0)
    # Extract the data starting from the second row and second column
    return df.iloc[:, 0:]

# Initialize an empty list to store the DataFrames
dfs = []
# Sort files
sorted_files=sorted(os.listdir(processed_data_path))
# Process each uploaded file
for filename in sorted_files:
    if filename.startswith('first_trimester'):
        # Load and process the CSV file
        print(filename)
        df = load_and_process_csv(os.path.join(processed_data_path, filename))
        # Append the DataFrame to the list
        dfs.append(df)

# Concatenate all DataFrames along the rows
combined_time_series = pd.concat(dfs, axis=0)

# If you need the result as a NumPy array
combined_ts_matrix = combined_time_series.values

# Print the resulting matrix
print(len(combined_ts_matrix[0]))
print(len(combined_ts_matrix))

os.makedirs(resultado_funciones_path,exist_ok=True)

for funcion in conjunto_funciones:
  distance_matrix_size = 72
  resultado_funciones_total = np.zeros(distance_matrix_size,distance_matrix_size)
  for k in range(4):
    function_folder = os.path.join(resultado_funciones_path,str(funcion.__name__))
    os.makedirs(function_folder,exist_ok=True)
    distance_matrix_size = 24
    resultado_funciones = np.zeros(distance_matrix_size,distance_matrix_size)
    for i in range(int(len(combined_ts_matrix)/4)):
      for j in range(i+1,int(len(combined_ts_matrix)/4)):
        resultado_funciones[i,j] = funcion(combined_ts_matrix[(k*24)+i],combined_ts_matrix[(k*24)+j])
        resultado_funciones_total[(k*24)+i,(k*24)+j] = resultado_funciones[i,j]
    # Perform hierarchical clustering
    linkage_matrix = linkage(resultado_funciones, method='average')
    # Get the headers for labeling
    headers = df.columns

    # Plot the dendrogram with modified labels
    plt.figure(figsize=(10, 10))
    labelsX = ['ALTO PARARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
               'Centro est','Centro norte','Centro sur','Chaco','CORDILLERA',
               'Metropolitano','PARAGUARI','Paraguay','PTE HAYES','SAN PEDRO',
               'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
               'CAAZAPA','NEEMBUCU','ALTO PARAGUAY']
    year = str(2019 + k)
    print(f'{funcion.__name__}_{year}')
    dendrogram(linkage_matrix,labels=labelsX,orientation='top', color_threshold=0.7,leaf_rotation=90,leaf_font_size=7,)

    # Create sub folders
    csv_folder = os.path.join(function_folder,'csv')
    os.makedirs(csv_folder, exist_ok=True)
    svg_folder = os.path.join(function_folder,'svg')
    os.makedirs(svg_folder, exist_ok=True)

    # Plot graph
    plt.title(f'{funcion.__name__}-{year}')
    plt.xlabel('Departamento')
    plt.ylabel('Distancia')

    # Save the plot
    svg_file = f'{funcion.__name__}_{year}.svg'
    svg_folder = os.path.join(svg_folder,svg_file)
    plt.savefig(svg_folder)
    matriz_distancia = DataFrame(resultado_funciones)
    csv_file = f'{funcion.__name__}_{year}.csv'
    csv_folder = os.path.join(csv_folder,csv_file)
    matriz_distancia.to_csv(csv_folder)
    #Close plot and finish
    plt.clf()
    plt.close()
  


for folder in os.listdir(resultado_funciones_path):
  current_folder = os.path.join(resultado_funciones_path,folder,'csv')
  vector = []
  files = os.listdir(current_folder)
  for file in sorted(files):
    if file.endswith('.csv'):
      df = pd.read_csv(os.path.join(current_folder, file))
      vector.append(df.to_numpy().flatten())

  matriz_distancia = np.zeros((len(vector),len(vector)))
  for i in range(len(matriz_distancia)):
    for j in range(i+1,len(matriz_distancia)):
      dist = distance.euclidean(vector[i], vector[j])
      matriz_distancia[i,j] = dist
      matriz_distancia[j,i] = dist
  # Perform hierarchical clustering
  condensed_distance_matrix = distance.squareform(matriz_distancia)
  linkage_matrix = linkage(condensed_distance_matrix, method='average')


  # Get the headers for labeling
  headers = df.columns
  # Plot the dendrogram with modified labels
  plt.figure(figsize=(10, 10))
  labelsX = ["2019","2020","2021","2022"]
  year = str(2019 + k)
  print(folder)
  dendrogram(linkage_matrix,labels=labelsX,orientation='top', color_threshold=0.7,leaf_rotation=90,leaf_font_size=7,)

  # Plot graph
  plt.title(f'{folder}')
  plt.xlabel('Year')
  plt.ylabel('Distance')

  # Save the plot
  svg_file = f'{folder}.svg'
  svg_folder = os.path.join(resultado_funciones_path,folder,svg_file)
  csv_file = f'{folder}.csv'
  csv_folder = os.path.join(resultado_funciones_path,folder,csv_file)
  plt.savefig(svg_folder)
  df_matriz_distancia = DataFrame(matriz_distancia)
  df_matriz_distancia.to_csv(csv_folder)
  #Close plot and finish
  plt.clf()
  plt.close()