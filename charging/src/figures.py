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


from .utilities import IsIterable

#Defining some 5 pronged color schemes
color_scheme_5_0=["#e7b7a5","#da9b83","#b1cdda","#71909e","#325666"]

#Defining some 4 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_4_0=["#8de4d3", "#0e503e", "#43e26d", "#2da0a1"]
color_scheme_4_1=["#069668", "#49edc9", "#2d595a", "#8dd2d8"]
color_scheme_4_2=["#f2606b", "#ffdf79", "#c6e2b1", "#509bcf"] #INCOSE IS2023

#Defining some 3 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_3_0=["#72e5ef", "#1c5b5a", "#2da0a1"]
color_scheme_3_1=["#256676", "#72b6bc", "#1eefc9"]
color_scheme_3_2=['#40655e', '#a2e0dd', '#31d0a5']
color_scheme_3_3=["#f2606b", "#c6e2b1", "#509bcf"] #INCOSE IS2023 minus yellow

#Defining some 2 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_2_0=["#21f0b6", "#2a6866"]
color_scheme_2_1=["#72e5ef", "#3a427d"]
color_scheme_2_2=["#1e4d2b", "#c8c372"] #CSU green/gold

#Named color schemes from https://www.canva.com/learn/100-color-combinations/
colors={
	'day_night':["#e6df44","#f0810f","#063852","#011a27"],
	'beach_house':["#d5c9b1","#e05858","#bfdccf","#5f968e"],
	'autumn':["#db9501","#c05805","#6e6702","#2e2300"],
	'ocean':["#003b46","#07575b","#66a5ad","#c4dfe6"],
	'forest':["#7d4427","#a2c523","#486b00","#2e4600"],
	'aqua':["#004d47","#128277","#52958b","#b9c4c9"],
	'field':["#5a5f37","#fffae1","#524a3a","#919636"],
	'misty':["#04202c","#304040","#5b7065","#c9d1c8"],
	'greens':["#265c00","#68a225","#b3de81","#fdffff"],
	'citroen':["#b38540","#563e20","#7e7b15","#ebdf00"],
	'blues':["#1e1f26","#283655","#4d648d","#d0e1f9"],
	'dusk':["#363237","#2d4262","#73605b","#d09683"],
	'ice':["#1995ad","#a1d6e2","#bcbabe","#f1f1f2"],
}

def ReturnColorMap(colors):

	if type(colors)==str:
		cmap=matplotlib.cm.get_cmap(colors)
	else:
		cmap=LinearSegmentedColormap.from_list('custom',colors,N=256)

	return cmap

def PlotLine(x,y,figsize=(8,8),ax=None,line_kwargs={},axes_kwargs={}):

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	ax.plot(x,y,**line_kwargs)

	ax.set(**axes_kwargs)

	if return_fig:
		return fig

def GetSignificantFactors(model,alpha=.05,label_substitutions={}):

	params=model._results.params[1:]
	error=model._results.bse[1:]
	pvalues=model._results.pvalues[1:]
	names=np.array(list(dict(model.params).keys()))[1:]

	for idx in range(len(names)):

		name=names[idx]

		for key,val in label_substitutions.items():

			if key in name:

				names[idx]=name.replace(key,val)
				name=names[idx]

	params=params[pvalues<alpha]
	error=error[pvalues<alpha]
	names=names[pvalues<alpha]
	pvalues1=pvalues[pvalues<alpha]

	name_lengths=[len(name) for name in names]
	order=np.flip(np.argsort(name_lengths))

	return params[order],error[order],names[order]

def FactorsPlot(model,alpha=.05,figsize=(8,8),ax=None,bar_kwargs={},axes_kwargs={},
	label_substitutions={},font_kwargs={}):

	plt.rcParams.update(**font_kwargs)

	params,error,names=GetSignificantFactors(model,alpha,
		label_substitutions=label_substitutions)

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	ax.barh(list(range(len(params))),params,xerr=error,**bar_kwargs)

	ax.set_yticks(list(range(len(params))))
	ax.set_yticklabels(names)

	ax.set(**axes_kwargs)

	if return_fig:
		return fig

def PlotSOCTrace(problem,figsize=(8,8),ax=None,colors='viridis',
	axes_kwargs={},line_kwargs={},legend_kwargs={}):

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	cmap=ReturnColorMap(colors)

	if IsIterable(problem):
		sol_dims=(problem[0].solution['soc'].shape[0],len(problem))

	else:
		sol_dims=problem.solution['soc'].shape

	x=np.arange(0,sol_dims[0])

	for idx in range(sol_dims[1]):

		line_color=cmap(idx/(sol_dims[1]-1)*.999)
		line_kwargs['color']=line_color


		if IsIterable(problem):

			line_label=f'SIC={problem[0].sic:.3f}'
			line_kwargs['label']=line_label

			PlotLine(x,problem[idx].solution['soc'],ax=ax,line_kwargs=line_kwargs)

		else:

			line_label=f'SIC={problem.sic[0]:.3f}'
			line_kwargs['label']=line_label

			PlotLine(x,problem.solution['soc'][idx],ax=ax,line_kwargs=line_kwargs)

	ax.legend(**legend_kwargs)
	ax.set(**axes_kwargs)

	if return_fig:
		return fig

def PlotBar(data,x_shift,figsize=(8,8),ax=None,bar_kwargs={},axes_kwargs={}):

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	x=[idx for idx in range(len(data))]
	ax.bar(x,data,**bar_kwargs)

	ax.set(**axes_kwargs)

	if return_fig:
		return fig

def PlotCircle(coords,r=1,figsize=(8,8),ax=None,patch_kwargs={},axes_kwargs={}):

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	if not IsIterable(r):
		r=np.ones(len(coords))*r

	for idx,xy in enumerate(coords):

		ax.add_patch(ptc.Circle(xy,r[idx],**patch_kwargs))

	ax.set(**axes_kwargs)

	if return_fig:
		return fig


def PlotChargeInfo(problem,figsize=(8,8),ax=None,colors='viridis',
	axes_kwargs={},bar_kwargs={},legend_kwargs={}):

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	cmap=ReturnColorMap(colors)

	if IsIterable(problem):
		sol_dims=(problem[0].solution['soc'].shape[0],len(problem))

	else:
		sol_dims=problem.solution['soc'].shape

	if not 'width' in bar_kwargs.keys():
		bar_kwargs['width']=.25

	x=np.arange(0,sol_dims[1])

	home_charge=np.zeros(sol_dims[1])
	work_charge=np.zeros(sol_dims[1])
	dest_charge=np.zeros(sol_dims[1])
	ad_hoc_charge=np.zeros(sol_dims[1])

	for idx0 in range(sol_dims[0]):
		for idx1 in range(sol_dims[1]):

			home_charge[idx1]+=problem.solution['u_dd']*problem.inputs['r_d']
			work_charge[idx1]+=problem.solution['u_dd']*problem.inputs['r_d']
			dest_charge[idx1]+=problem.solution['u_dd']*problem.inputs['r_d']
			home_charge[idx1]+=problem.solution['u_ad']*problem.inputs['r_a']



	for idx in range(sol_dims[1]):

		color=cmap(idx/(sol_dims[1]-1)*.999)
		bar_kwargs['facecolor']=color


		if IsIterable(problem):

			line_label=f'SIC={problem[0].sic:.3f}'
			line_kwargs['label']=line_label

			PlotBar(x,problem[idx].solution['soc'],ax=ax,bar_kwargs=bar_kwargs)

		else:

			line_label=f'SIC={problem.sic[0]:.3f}'
			line_kwargs['label']=line_label

			PlotBar(x,problem.solution['soc'][idx],ax=ax,bar_kwargs=bar_kwargs)

	ax.legend(**legend_kwargs)
	ax.set(**axes_kwargs)

	if return_fig:
		return fig

