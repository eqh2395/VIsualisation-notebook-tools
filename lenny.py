

"""
Creates iris cubes, turns them into geographical plots and creates a video visualisation from them for time series data.
Timestamp data must be part of your file metadata as opposed to part of the file itself to use this tool.
Recommended that requirements.txt is installed before installing this library. Download, cd to its directory and type 
'pip  install -r requirements.txt' into shell.
"""
    
import iris
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import iris.quickplot as qplt
import iris.plot as iplt
from mpl_toolkits import mplot3d
import numpy as np
from iris.experimental.equalise_cubes import equalise_attributes
import iris.plot as iplot
import iris.analysis.cartography
from matplotlib.transforms import offset_copy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import os
from iris.cube import CubeList
from matplotlib.widgets import Slider
import subprocess
import matplotlib.font_manager as fm
import matplotlib.colors as colors
import matplotlib
from matplotlib.ft2font import FT2Font
import sys

def install_lenny_on_workers(client):
    
    """

    Installs the lenny library on all dask workers. Required for load_cubes and make_plots when specifying a dask scheduler.

    """
    from urllib.request import urlretrieve
    outfile = os.path.abspath(os.path.join('.', 'lenny.py'))
    url = "https://raw.githubusercontent.com/informatics-lab/lenny/master/lenny.py"
    urlretrieve(url, outfile)
    sys.path.append(os.path.dirname(outfile))
    
    client.run(install_lenny_on_workers)

def load_path(folder_path):
    """
    Creates a list of filepaths with the appropriate formatting for processing.
    
    folder_path: Filepath to directory containing intended data to be used. Directory must contain only files intended to be    
    plotted.
    Acceptible types of file are CF NetCDF, GRIB 1 & 2, PP and FieldsFiles file formats based on compatability with iris.
    """
    filenames = sorted(os.listdir(folder_path))
    filepaths = [os.path.join(folder_path, file) for file in filenames]
    return filepaths
    
def __load_uniform_cubes__(filepaths, add_coord=None, aggregate=None, subset=None, masking=None):
    
    cubelist = iris.load(filepaths)
#may not work if don't add add_coord?
    if add_coord == None:
        for c in cubelist:
            c.rename('v')
                    
    if type(add_coord) == tuple:
        meta_data_name, starting_index, final_index, new_dimension_name = add_coord
        for c in cubelist:
            c.rename('v')
            n = int(c.attributes[meta_data_name][starting_index:final_index])
            c.add_aux_coord(iris.coords.AuxCoord(n, new_dimension_name))

    equalise_attributes(cubelist)
        
    cubelist = cubelist.merge_cube()
    
    if (type(aggregate) == str) and (type(subset) == tuple) and (type(masking)==float):
        new_cube = cubelist.collapsed(aggregate, iris.analysis.SUM)
        west, east, south, north = subset
        new_cube = new_cube[0].intersection(longitude=(west,east), latitude=(south,north))
        new_cube.data = np.ma.masked_where(new_cube.data <= masking, new_cube.data)
    
    if (type(aggregate)==str) and (type(subset)==tuple and (masking==None)):
        new_cube = cubelist.collapsed(aggregate, iris.analysis.SUM)
        west, east, south, north = subset
        new_cube = new_cube[0].intersection(longitude=(west,east), latitude=(south,north))
    
    if (type(aggregate)==str) and (subset==None) and (masking==None):
        new_cube = cubelist.collapsed(aggregate, iris.analysis.SUM)
        new_cube = new_cube[0]

    if (type(aggregate)==str) and (subset==None) and (type(masking)==float):
        new_cube = cubelist.collapsed(aggregate, iris.analysis.SUM)
        new_cube = new_cube[0]
        new_cube.data = np.ma.masked_where(new_cube.data <= masking, new_cube.data)
        
    if (aggregate==None) and (type(subset)==tuple) and (masking==None):
        new_cube = cubelist
        west, east, south, north = subset
        new_cube = new_cube[0].intersection(longitude=(west,east), latitude=(south,north))
    
    if (aggregate==None) and (type(subset)==tuple) and (type(masking)==float):
        new_cube = cubelist
        west, east, south, north = subset
        new_cube = new_cube[0].intersection(longitude=(west,east), latitude=(south,north))
        new_cube.data = np.ma.masked_where(new_cube.data <= masking, new_cube.data)
    
    if (aggregate==None) and (subset==None) and (type(masking)==float):
        new_cube = cubelist
        new_cube = new_cube[0]
        new_cube.data = np.ma.masked_where(new_cube.data <= masking, new_cube.data)
    
    if (aggregate==None) and (subset==None) and (masking==None):
        new_cube = cubelist
        new_cube = new_cube[0]
            
    return new_cube
    

def load_uniform_cubes(filepath, add_coord=None, aggregate=None, subset=None, masking=None, scheduler_address=None):
    
    """
    Load cubes with uniform metadata from path to CF NetCDF, GRIB 1 & 2, PP and FieldsFiles files. Returns a list of cubes.
    
    
    Args:
    
    filepaths: List of filepaths to data. Will attempt to read everything within this list.
    
    
    Kwargs:
    
    add_coord: Turns a metadata attribute into a new cube dimension. Takes a tuple in this format - (meta_data_name,    
    starting_index, final_index, new_dimension_name). meta_data_name=string value of attribute, 
    starting_index=index of starting numerical value in attribute to be contained in new dimension, 
    final_index=end index of numerical value in attribute to be contained in new dimension, 
    new_dimension_name=string value of name of new cube dimension.
    
    aggregate: Takes dimension to collapse cube along and returns cumulative sum of coordinate along that dimension. 
    Only collapses along dimension if cube has more than 2 dimensions. Dimension must contain data non-uniform across the cube.
    
    subset: Restricts data to an intersection of the cube with specified coordinate ranges. 
    Takes a tuple containing floats in this format - (west, east, south, north).
    
    masking: Masks all data that is smaller than a specified value. Takes a float.
    
    scheduler_address: Takes path to a dask scheduler. If scheduler is not defined, default is a local threaded scheduler.
    
    For optimal runtime, specify own dask client. If dask errors are raised, ensure that lenny has been installed on the workers.
    """
    
    from dask import delayed
    import dask.bag as db
    from distributed import Client
    from dask_kubernetes import KubeCluster
    
    if scheduler_address is not None:
        client =  Client(scheduler_address)
        lazycubes = db.from_sequence(filepath).map(__load_uniform_cubes__, add_coord, aggregate, subset, masking)
        realcubes = lazycubes.compute()
        cubelist = list(enumerate(realcubes))
        return cubelist
    
    if scheduler_address==None:
        client = Client()
        lazycubes = db.from_sequence(filepath).map(__load_uniform_cubes__, add_coord, aggregate, subset, masking)
        realcubes = lazycubes.compute()
        cubelist = list(enumerate(realcubes))
        return cubelist

def __extract_cube__(filepath, constraint, add_coord=None, aggregate=None, subset=None, masking=None, scheduler_address=None):

    cube = iris.load_cube(filepath, constraint)
        
    if type(add_coord) == tuple:
        meta_data_name, starting_index, final_index, new_dimension_name = add_coord
        n = int(cube.attributes[meta_data_name][starting_index:final_index])
        cube.add_aux_coord(iris.coords.AuxCoord(n, new_dimension_name))
            
    if subset==None:
        subset = cube[0]
    if type(subset) == tuple:
        west, east, south, north = subset
        subset = cube[0].intersection(longitude=(west,east), latitude=(south,north))
                
    if type(aggregate) == str:
        cube = cube.collapsed(aggregate, iris.analysis.SUM)
            
    if masking==None:
        subset = subset
    if type(masking)==float:
        subset.data = np.ma.masked_where(subset.data <= masking, subset.data)
    
    return subset

def extract_cube(filepath, constraint, add_coord=None, aggregate=None, subset=None, masking=None, scheduler_address=None):
    
    """
    Extracts a single cube from filepath to CF NetCDF, GRIB 1 & 2, PP or FieldsFiles files. 
    Will take all files grouped together by specified metadata and merge them into one cube.
        
        
    Args:
    
    filepath: Path to the file containing the data.
    
    constraint: a string, float, or integer describing an item of metadata (e.g. name, model level number, etc...) 
    that distinguishes it from other files.
    These files will then be merged into one cube.
        
    Example: ('./prods_op_mogreps-uk_20130703_03_00_003.nc','stratiform_snowfall_rate')
        
        
    Kwargs:
        
    add_coord: Turns a metadata attribute into a new cube dimension. Takes a tuple in this format - 
    (meta_data_name,    starting_index, final_index, new_dimension_name). meta_data_name=string value of attribute, 
    starting_index=index of starting numerical value in attribute to be contained in new dimension, 
    final_index=end index of numerical value in attribute to be contained in new dimension, 
    new_dimension_name=string value of name of new cube dimension.
    
    aggregate: Takes dimension to collapse cube along and returns cumulative sum of coordinate along that dimension. 
    Only collapses along dimension if cube has more than 2 dimensions. Dimension must contain data non-uniform across the cube.
    
    subset: Restricts data to an intersection of the cube with specified coordinate ranges. 
    Takes a tuple containing floats in this format - (west, east, south, north).
    
    masking: Masks all data that is smaller than a specified value. Takes a float.
    
    For optimal runtime, specify own dask client. 
    If dask errors are raised, ensure that lenny has been installed on the workers.
        
    """
    
    loadcubes = db.from_sequence(filepaths).map(__extract_cube__)
    cubes = loadcubes.compute()
    
        
def __make_plots__(cube_list, save_filepath, figsize=(16,9), terrain=cimgt.StamenTerrain(), 
                   logscaled=True, vmin=None, vmax=None, colourmap='viridis', colourbarticks=None, 
                   colourbarticklabels=None, colourbar_label=None, markerpoint=None, markercolor='#B9DC0C', 
                   timestamp=None, time_box_position=None, plottitle=None, box_colour='#FFFFFF', textcolour='#000000', 
                   coastlines=False, scheduler_address=None):

    cubenumber, cube = cube_list
        
    if type(figsize) == tuple:
        fig, ax = plt.subplots(figsize=figsize)
    
    Terrain = terrain
    fig = plt.axes(projection=Terrain.crs)
    fig.add_image(terrain, 4)
    
    if logscaled==True and vmin==None and vmax==None and colourmap == 'viridis':
        data = iplt.pcolormesh(cube, alpha=1, norm=colors.LogNorm(), cmap='viridis')
        
    if logscaled==True and type(vmin) == float and type(vmax) == float and colourmap == 'viridis':
        data = iplt.pcolormesh(cube, alpha=1, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap='viridis')
    
    if logscaled==True and vmin==None and vmax==None and type(colourmap)==str:
        data = iplt.pcolormesh(cube, alpha=1, norm=colors.LogNorm(), cmap=colourmap)
        
    if logscaled==True and type(vmin) == float and type(vmax) == float and type(colourmap)==str:
        data = iplt.pcolormesh(cube, alpha=1, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=colourmap)
        
    if logscaled==False and vmin==None and vmax==None and colourmap == 'viridis':
        data = iplt.pcolormesh(cube, alpha=1, norm=colors.LogNorm(), cmap='viridis')
        
    if logscaled==False and type(vmin) == float and type(vmax) == float and colourmap == 'viridis':
        data = iplt.pcolormesh(cube, alpha=1, vmin=vmin, vmax=vmax, cmap='viridis')
    
    if logscaled==False and vmin==None and vmax==None and type(colourmap)==str:
        data = iplt.pcolormesh(cube, alpha=1, cmap=colourmap)
        
    if logscaled==False and type(vmin) == float and type(vmax) == float and type(colourmap)==str:
        data = iplt.pcolormesh(cube, alpha=1, vmin=vmin, vmax=vmax, cmap=colourmap)
    
    if coastlines==True:
        plt.gca().coastlines('50m')
    
    cbaxes = plt.axes([0.2, 0.25, 0.65, 0.03])
    colorbar = plt.colorbar(data, cax=cbaxes, orientation = 'horizontal')
    colorbar.set_ticks(colourbarticks)
    colorbar.set_ticklabels(colourbarticklabels)
    
    if colourbar_label is not None:
        colorbar.set_label(colourbar_label, fontproperties='FT2Font', color=box_colour, fontsize=8, 
                           bbox=dict(facecolor=box_colour, edgecolor='#2A2A2A', boxstyle='square'))
    
    if markerpoint is not None and markercolor is not None:
        longitude, latitude, name_of_place = markerpoint
        fig.plot(longitude, latitude, marker='^', color=markercolor, markersize=12, transform=ccrs.Geodetic())
        geodetic_transform = ccrs.Geodetic()._as_mpl_transform(fig)
        text_transform = offset_copy(geodetic_transform, units='dots', y=+75)
        fig.text(longitude, latitude, u+name_of_place, fontproperties='FT2Font', alpha=1, fontsize=8, 
                 verticalalignment='center', horizontalalignment='right', transform=text_transform, 
                 bbox=dict(facecolor=markercolor, edgecolor='#2A2A2A', boxstyle='round'))
    
    if timestamp is not None:
        attributedict = subset.attributes
        datetime = attributedict.get(timestamp)
        timetransform = offset_copy(geodetic_transform, units='dots', y=0)
        longitude_of_time_box, latitude_of_time_box = time_box_position
        fig.text(longitude_of_time_box, latitude_of_time_box, "Time, date: "+ datetime, fontproperties='FT2Font', 
                 alpha=0.7, fontsize=8 , transform=timetransform, bbox=dict(facecolor=markercolor, edgecolor='#2A2A2A', 
                                                                            boxstyle='round'))
            
    if plottitle is not None:
        titleaxes = plt.axes([0.2, 0.8, 0.65, 0.04], facecolor=box_colour)
        titleaxes.text(0.5,0.25, plottitle, horizontalalignment = 'center',  fontproperties='FT2Font', fontsize=10, 
                       weight=600, color = textcolour)
        titleaxes.set_yticks([])
        titleaxes.set_xticks([])

    picturename = save_filepath + "%04i.png" % cubenumber
    plt.savefig(picturename, dpi=200, bbox_inches="tight")
        

def make_plots(cube_list, save_filepath, figsize=(16,9), terrain=cimgt.StamenTerrain(), logscaled=True, 
               vmin=None, vmax=None, colourmap='viridis', colourbarticks=None, colourbarticklabels=None, 
               colourbar_label=None, markerpoint=None, markercolor='#B9DC0C', timestamp=None, time_box_position=None, 
               plottitle=None, box_colour='#FFFFFF', textcolour=None, coastlines=False, scheduler_address=None):
    
    """
    Makes plots from list of iris cubes.
    
    
    Args:
    
    cube_list: Takes list of cubes.
    
    save_filepath: Filepath where the plots will be saved. Ensure filepath is empty of any other files before creating video.
    
    
    Kwargs:
    
    figsize: Sets size of figure. Default is 16 in x 9 in. Takes a tuple of integers or floats (e.g. (16, 9)).
    
    terrain: Sets background map image on plot. Default is Stamen Terrain. See further options here: 
    https://scitools.org.uk/cartopy/docs/latest/cartopy/io/img_tiles.html
    
    logscaled: Plots data on a logarithmic (to the base-10) scale. Takes True or False. 
    Makes more appealing visualisations of skewed data. Default is True.
    
    vmin: set lowest value to be displayed. Takes a float.
    
    vmax: set highest value to be displayed. Takes a float.
    
    colourmap: Sets colour map for the plot. Default is 'viridis'. Other colourmaps: 'magma', 'plasma', 'inferno', 'cividis'. 
    See https://matplotlib.org/tutorials/colors/colormaps.html for more colourmap options.
    
    colourbarticks: Sets position within data of ticks on colourbar. Takes a list of floats or integers, e.g. [10, 100, 1000].
    
    colourbarticklabels: Sets labels for colourbar ticks. Takes a list, eg. [10, 100, 1000].
    
    colourbar_label: Sets colourbar legend. Takes a string.
    
    markerpoint: Plots a marker on the map based on global coordinates. Takes a tuple in this format - 
    (longitude, latitude, 'name_of_place'), longitude and latitude must be a float, name_of_place must be a string.
    
    markercolor: Color of the location marker. Must be given as a string. Default is '#B9DC0C'.
    
    timestamp: Places a timestamp box on each plot. 
    Takes a string referring to name of timestep metadata within the cube. 
    It must contain timesteps in original metadata to use this, as this takes the name of the timestep attribute.
    
    plottitle: Displays the title of your video horizontally across the top of the plots. Takes a string.
    
    time_box_position: Position of timestamp box on the plot. Takes a tuple of integers eg. (60, 0)
    
    box_colour: Sets colour of title box and colourbar label box. Takes a string.
    
    textcolor: Sets colour of text within boxes. Takes a string.
        
    coastlines: Draw coastlines around land on the map. Takes True or False.
    
    For optimal runtime, specify own dask client. 
    If dask errors are raised, ensure that lenny has been installed on the workers.
    
    """
    
    from dask import delayed
    import dask.bag as db
    from distributed import Client
    from dask_kubernetes import KubeCluster
    
    
    if scheduler_address is not None:
        client =  scheduler_address
        client.run(lambda: sys.path.append(save_filepath))
        makingplots = db.from_sequence(cube_list).map(__make_plots__, save_filepath, figsize, terrain, logscaled, 
                                                      vmin, vmax, colourmap, colourbarticks, colourbarticklabels, 
                                                      colourbar_label, markerpoint, markercolor, timestamp, 
                                                      time_box_position, plottitle, box_colour, textcolour, coastlines)
        plots = makingplots.compute()

    if scheduler_address==None:
        client=Client()
        client.run(lambda: sys.path.append(save_filepath))
        makingplots = db.from_sequence(cube_list).map(__make_plots__, save_filepath, figsize, terrain, logscaled, 
                                                      vmin, vmax, colourmap, colourbarticks, colourbarticklabels, 
                                                      colourbar_label, markerpoint, markercolor, timestamp, 
                                                      time_box_position, plottitle, box_colour, textcolour, coastlines)
        plots = makingplots.compute()
    
    
    
def try_make_plots_from_cubes(cube, save_filepath, figsize=(16,9), terrain=cimgt.StamenTerrain(), logscaled=True, 
                              vmin=None, vmax=None, colourmap='viridis', colourbarticks=None, colourbarticklabels=None, 
                              colourbar_label=None, markerpoint=None, markercolor='#B9DC0C', timestamp=None, 
                              time_box_position=None, plottitle=None, box_colour='#FFFFFF', textcolour=None, coastlines=False):
        
    """"
    Makes sample plot from cube. Can be used to refine plot specifications.
        
    Args:
    
    cube_list: Takes a cube.
    
    save_filepath: Filepath where the plot will be saved.
    
    Kwargs:
    
    figsize: Sets size of figure. Default is 16 in x 9 in. Takes a tuple of integers or floats (e.g. (16, 9)).
    
    terrain: Sets background map image on plot. Default is Stamen Terrain. See further options here: 
    https://scitools.org.uk/cartopy/docs/latest/cartopy/io/img_tiles.html
    
    logscaled: Plots data on a logarithmic (to the base-10) scale. Takes True or False. 
    Makes more appealing visualisations of skewed data. Default is True.
    
    vmin: set lowest value to be displayed. Takes a float.
    
    vmax: set highest value to be displayed. Takes a float.
    
    colourmap: Sets colour map for the plot. Default is 'viridis'. Other colourmaps: 'magma', 'plasma', 'inferno', 'cividis'. 
    See https://matplotlib.org/tutorials/colors/colormaps.html for more colourmap options.
    
    colourbarticks: Sets position within data of ticks on colourbar. Takes a list of floats or integers, e.g. [10, 100, 1000].
    
    colourbarticklabels: Sets labels for colourbar ticks. Takes a list, eg. [10, 100, 1000].
    
    colourbar_label: Sets colourbar legend. Takes a string.
    
    markerpoint: Plots a marker on the map based on global coordinates. Takes a tuple in this format - 
    (longitude, latitude, 'name_of_place'), longitude and latitude must be a float, name_of_place must be a string.
    
    markercolor: Color of the location marker. Must be given as a string. Default is '#B9DC0C'.
    
    timestamp: Places a timestamp box on each plot. Takes a string referring to name of timestep metadata within the cube. 
    It must contain timesteps in original metadata to use this, as this takes the name of the timestep attribute.
    
    plottitle: Displays the title of your video horizontally across the top of the plots. Takes a string.
    
    time_box_position: Position of timestamp box on the plot. Takes a tuple of integers eg. (60, 0)
    
    box_colour: Sets colour of title box and colourbar label box. Takes a string.
    
    textcolor: Sets colour of text within boxes. Takes a string.
        
    coastlines: Draw coastlines around land on the map. Takes True or False.
    
    """
        
    cube_list = cube
    __make_plots_from_cubes__(cube_list)
    
    
def make_video(picture_filepath, end_video_filepath, interpolate=False, resize_video=False):
    
    """
    Zips together all png files within a specified filepath to make a video. 
    Ensure that ffmpeg is installed in your environment before running.
    
    
    Args:
    
    picture_filepath: Filepath to png files to be made into video frames. Takes a string. 
    Filepath must contain only png files intended for the video.
    
    end_video_filepath: Intended filepath and filename of video. Takes a string.
    
    
    Kwargs:
    
    interpolate: Creates additional frames between existing, meaning goes from 24 to 60 fps. 
    This is done by taking the mean of the adjacent frames. Default is False. Takes True or False.
    In order to use interpolate, must have ffmpeg 3.1 installed or higher. If you're video is over 1000??? 
    bytes in height, set resize_video=True.
    
    resize_video: Creates a video, then resizes it to have a height of 1000 bytes, with width adjusted accordingly. 
    Default False. Recommended if you wish to minterpolate and your video is large in size. 
    
    """
    
    if interpolate==False:
        return_value = subprocess.call(["ffmpeg","-y","-r","24", "-i", (picture_filepath + "/%04d.png"), 
                                        "-vcodec", "mpeg4","-qscale","5", "-r", "24", end_video_filepath])
        return return_value
    
    if interpolate==True and resize_video==False:
        return_value = subprocess.call(["ffmpeg","-y","-r","24", "-i", (picture_filepath + "/%04d.png"), 
                                        "-filter", "minterpolate", "-vcodec", "mpeg4","-qscale","5", "-r", "24", 
                                        end_video_filepath])
        return return_value
    
    #filepaths need working out
    
    
    if interpolate==True and resize_video==True:
        return_value = subprocess.call(["ffmpeg","-y","-r","24", "-i", (picture_filepath + "/%04d.png"), 
                                        "-filter", "minterpolate", "-vcodec", "mpeg4","-qscale","5", "-r", "24", 
                                        end_video_filepath])
        return return_value
        clip = mp.VideoFileClip(end_video_filepath)
        clip_resized = clip.resize(height=1000)
        clip_resized.write_videofile(end_video_filepath)
        return_value = subprocess.call(["ffmpeg", "-i", end_video_filepath, "-filter", "minterpolate", end_video_filepath])
        return return_value
    
if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
