import os
import sys
import time
import json
import requests
import warnings
import matplotlib
import numpy as np
import numpy.random as rand
import pandas as pd
import geopandas as gpd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from statsmodels.formula.api import ols
from scipy.stats import f as f_dist
from math import comb

from .utilities import IsIterable

def RSS(x,y):
	return ((x-y)**2).sum()

def MSS(x,y):
	return ((y-x.mean())**2).sum()

def TSS(x):
	return ((x-x.mean())**2).sum()

def RSquared(x,y):
	return 1-(RSS(x,y)/TSS(x))

def AdjustedRSquared(x,y,n,p):
	print(((1-RSquared(x,y))*(n-1)),(((1-RSquared(x,y))*(n-1))/(n-p-1)))
	return 1-(((1-RSquared(x,y))*(n-1))/(n-p-1))

def ANOVA(x,y,n,p):
	sse=RSS(x,y)
	ssm=MSS(x,y)
	sst=TSS(x)
	dfe=n-p
	dfm=p-1
	dft=n-1
	mse=sse/dfe
	msm=ssm/dfm
	mst=sst/dft
	f=msm/mse
	pf=f_dist.sf(f,dfm,dfe)
	# r2=1-(sse/sst)
	# ar2=1-(((1-r2)*dft)/(dfe-1))
	r2=RSquared(x,y)
	ar2=AdjustedRSquared(x,y,n,p)
	# print(n,p)

	out_string="\\hline R & R-Squared & Adjusted R-Squared & Std. Error \\\\\n"
	out_string+="\\hline {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\\n".format(
		np.sqrt(r2),r2,ar2,(x-y).std()/n)
	out_string+="\\hline"

	print(out_string)

	out_string="\\hline Category & Sum of Squares & DOF & Mean Squares \\\\\n"
	out_string+="\\hline Model & {:.3f} & {:.0f} & {:.3f} \\\\\n".format(ssm,dfm,msm)
	out_string+="\\hline Error & {:.3f} & {:.0f} & {:.3f} \\\\\n".format(sse,dfe,mse)
	out_string+="\\hline Total & {:.3f} & {:.0f} & {:.3f} \\\\\n".format(sst,dft,mst)
	out_string+="\\hline  \\multicolumn{2}{|c|}{$F$} &  "
	out_string+="\\multicolumn{2}{c|}{$P(>F)$}  \\\\\n"
	out_string+="\\hline  \\multicolumn{{2}}{{|c|}}{{{:.3f}}} &  ".format(f)
	out_string+="\\multicolumn{{2}}{{c|}}{{{:.3f}}}  \\\\\n".format(pf)
	out_string+="\\hline"

	print(out_string)

def ModelANOVA(model,df_norm,res_column,m=6):
	y_hat=Predict(model,df_norm)
	y=df_norm[res_column]
	n=df_norm.shape[0]
	p=sum([comb(m,n) for n in range(m+1)])

	return ANOVA(y,y_hat,n,p)

def Predict(model,df_norm):

	return model.predict(df_norm)

def PrintLaTeXTabular(model,alpha=.05,label_substitutions={}):
	params=model._results.params
	tvalues=model._results.tvalues
	pvalues=model._results.pvalues
	names=np.array(list(dict(model.params).keys()))
	
	for idx in range(len(names)):

		name=names[idx]

		for key,val in label_substitutions.items():

			if key in name:

				names[idx]=name.replace(key,val)
				name=names[idx]

	params=params[pvalues<alpha]
	tvalues=tvalues[pvalues<alpha]
	names=names[pvalues<alpha]
	pvalues=pvalues[pvalues<alpha]

	name_lengths=[len(name) for name in names]

	name_length_order=np.append(0,np.argsort(name_lengths[1:])+1)


	params=params[name_length_order]
	tvalues=tvalues[name_length_order]
	names=names[name_length_order]
	pvalues=pvalues[name_length_order]

	out_string=""
	for i in range(len(names)):
		out_string+="\\hline {{\\small {} }} & {:.3f} & {:.3f} & {:.3f} \\\\\n".format(
			names[i],params[i],tvalues[i],pvalues[i])
	
	return out_string

