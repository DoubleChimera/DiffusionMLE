# -*- coding: utf-8 -*-
import codecs
import copy
import json
import os.path
import math
import numpy as np
import pandas as pd
import pickle

# This is a series of functions that facilitate handling flika output JSON files


# Load the json file
def open_tracks(filename):
    """
    returns txy_pts and tracks extracted from a .json file saved by flika's pynsight plugin

    txy_pts is a 2D array. Every row is a particle localization. The columns are [t, x, y], where t is the frame the
    particle was localized in. x and y are the coordinates of the center of the particle determined by a 2D gaussian
    fit, in pixel space.

    tracks is a list of track arrays. Each value in a track array contains an index to a point in txy_pts.
    To extract the coordinates of the ith track, use:

        >>> track = tracks[i]
        >>> pts = txy_pts[track, :]
        >>> print(pts)

            array([ [   0.   ,   23.32 ,  253.143],
                    [   1.   ,   23.738,  252.749],
                    [   2.   ,   23.878,  252.8  ]])
    """

    obj_text = codecs.open(filename, "r", encoding="utf-8").read()
    pts = json.loads(obj_text)
    txy_pts = np.array(pts["txy_pts"])
    tracks = [np.array(track) for track in pts["tracks"]]
    return txy_pts, tracks

# Retreive individual tracks
def gen_indiv_tracks(minfrm, tracks, txy_pts):
    """
    returns lst[] and lstnan[] and trackOrigins (dictionary)
    lst[] contains track lengths greater than the minfrm (minimum frame) value.
    lstnan[] has blank track frames filled with nan values

    The track number is not related to the track numbers in the .json file,
    they are generated while populating the lst[] with tracks greater than the
    minfrm (minimum frame) value.  It is used as an index for reference only.

    lst is a 3D array. Every element is a particle trajectory. The columns are [t, x, y], where t is the frame the
    particle was localized in. x and y are the coordinates of the center of the particle determined by a 2D gaussian
    fit, in pixel space.

    lst is a list of track arrays. Each value in a lst array contains an index to a particular track.
    To extract the coordinates of the ith track, use:

        >>> print(lst[i-1])

            array([ [   0.   ,   23.32 ,  253.143],
                    [   1.   ,   23.738,  252.749],
                    [   2.   ,   23.878,  252.8  ]  ])

    gen_indiv_tracks takes three arguments,
        minfrm :    the minimum number of frames a track should have to be included
        tracks :    a list of track arrays
        txy_pts:    a 2D array with every row a particle localization
    """

    numTracks = len(tracks)
    nan = np.nan

    # make list of tracks with >= min number frames
    lst = []

    for i in range(0, (numTracks)):
        track = tracks[i]
        pts = txy_pts[track, :]
        if len(pts) >= minfrm:
            lst.append(pts)

    # Move all tracks such that their starting index is 0
    for track in range(len(lst)):
        if lst[track][0][0] != 0:
            indset = lst[track][0][0]
            for pts in range(len(lst[track][0:])):
                lst[track][pts][0] = lst[track][pts][0] - indset

    lstnan = copy.deepcopy(lst)

    # parse through the list and fill in NaNs
    for k in range(0, len(lst)):
        # fill in missing frames with NaN values
        totalnumber = lstnan[k][-1][0] + 1
        missing = sorted(list(set(range(int(totalnumber))) - set(lstnan[k][:, 0])))
        for elem in missing:
            lstnan[k] = np.insert(lstnan[k], elem, [[elem, nan, nan]], axis=0)

    # a dictionary of index-0 pts from each track
    trackOrigins = {}
    for index, track in enumerate(lstnan):
        trackOrigins[index] = track[0][1:]

    return lst, lstnan, trackOrigins

def json_to_pandas(filename, experiment_name, minfrm=200):
    """ Converts a flika JSON output file to a pandas dataframe
        also
        Converts a list of flika JSON output files to a pandas dataframe

    Args:
        filename (flika .json file): Complete path to flika JSON file
        filename (List of flika .json files): List of complete paths to flika JSON files

        experiment_name (string): Unique identifier for each experiment
        experiment_name (list): List of unique identifiers for each experiment

        minfrm (int, optional): Minimum number of frames per track to include in dataframe.
        Defaults to 200.

    Returns:
        pandas dataframe: All tracks in 5 columns ['Frame', 'X', 'Y', 'ID', 'Exp_Name']
        where 'ID' is the track ID and 'Exp_Name' is a unique identifier for each experiment
    """

    if isinstance(filename, str) and isinstance(experiment_name, str):
        txy_pts, tracks = open_tracks(filename)
        lst, lstnan, trackOrigins = gen_indiv_tracks(minfrm, tracks, txy_pts)
        track_array = []
        for trackID, track in enumerate(lstnan):
            track_df = pd.DataFrame(track)
            track_df[len(track_df.columns)] = int(trackID)
            track_array.append(track_df.values)
        combined_tracks = np.vstack(track_array)
        finalTracks_df = pd.DataFrame(combined_tracks)
        finalTracks_df.columns = ['Frame', 'X', 'Y', 'ID']
        finalTracks_df['Exp_Name'] = str(experiment_name)
        return finalTracks_df

    if isinstance(filename, list) and isinstance(experiment_name, list):
        trackID = 0
        track_array = []
        for index, eachFile in enumerate(filename):
            txy_pts, tracks = open_tracks(eachFile)
            lst, lstnan, trackOrigins = gen_indiv_tracks(minfrm, tracks, txy_pts)
            for track in lstnan:
                track_df = pd.DataFrame(track)
                track_df[len(track_df.columns)] = int(trackID)
                track_df[len(track_df.columns)] = str(experiment_name[index])
                track_array.append(track_df.values)
                trackID += 1
        combined_tracks = np.vstack(track_array)
        finalTracks_df = pd.DataFrame(combined_tracks)
        finalTracks_df.columns = ['Frame', 'X', 'Y', 'ID', 'Exp_Name']
        return finalTracks_df

def pandas_to_flika_JSON(tracks_df, output_dir, outfile_name):
    """Converts a dataframe into a standard flika JSON output file

    Args:
        tracks_df (pandas dataframe): A pandas dataframe with unique ID for each track
        output_dir (str): Directory to ouput JSON file to
        outfile_name (str): Name of output file without the extension ex. 'test' will save as test.json
    """
    tracks_df.drop(columns=['Exp_Name'], inplace=True)
    tracks_df['ID'] = tracks_df['ID'].astype(int)
    tracks_df.set_index(['ID'], inplace=True)
    # get a list of all unique indices
    trackID_list = list(tracks_df.index.unique())
    # initiate empty lists for tracks and txy_pts
    tracks = []
    txy_pts = []
    trackCounter = 0
    # Loop over each track in the list
    for trackID in trackID_list:
        trackCounter += 1
        print(f'Processing track {trackCounter} out of {len(trackID_list)}.')
        txy_pts_length = len(txy_pts)
        track_length = len(tracks_df.loc[trackID])
        for index, localization in tracks_df.loc[trackID].iterrows():
            if math.isnan(localization.sum(skipna=False)):
                track_length -= 1
                continue
            else:
                txy_pts.append(list(localization))
        txy_pts_newLength = len(txy_pts)
        if txy_pts_length == 0:
            first_index = 0
            last_index = track_length - 1
        elif txy_pts_length > 0:
            first_index = txy_pts_length
            last_index = txy_pts_length + track_length - 1
        track_index_list = list(range(first_index, last_index + 1))
        tracks.append(track_index_list)
    # Combine tracks[] and txy_pts[] into a dictionary
    trackDict = {}
    trackDict["tracks"] = tracks
    trackDict["txy_pts"] = txy_pts
    # output that dictionary as a .json file in utf-8 format
    trackDictOutPath = os.path.join(output_dir, outfile_name + ".json")
    json.dump(trackDict, cls=NumpyEncoder, fp=codecs.open(trackDictOutPath, "w", encoding="utf-8"), separators=(",", ":"), indent=4, sort_keys=True)

class NumpyEncoder(json.JSONEncoder):
    """
    This is a necessary evil for json output.
    Do not edit this
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def JSONsplitTracks(txy_pts, tracks, cutoff_length):
    """ Takes the output of open_tracks as an input
        open_tracks takes a JSON as an input
        makes as many complete truncated sets of tracks as possible with cutoff_length size
        remainders are discarded
    Args:
        txy_pts ([list]): [t, x, y points of tracks]
        tracks ([list]): [track id number]
        cutoff_length ([int]): [cutoff value to use for tracks]

    Returns:
        [list]: [an array of the split tracks]
    """
    split_tracks = []
    for trk_ind, track in enumerate(tracks):
        if len(track) >= cutoff_length:
            pts =  txy_pts[track, :]
            numberTrackSegments = math.floor(float(len(track))/float(cutoff_length))
            for numb_value in list(range(1, numberTrackSegments + 1)):
                if numb_value == 1:
                    split_tracks.append(pts[:cutoff_length])
                else:
                    split_tracks.append(pts[((numb_value - 1) * cutoff_length): (numb_value * cutoff_length)])
    return split_tracks

def genJSONstyleDict(track_array, out_name, save_path, saveJSON=False):
    # Initiate the empty lists for tracks and txy_pts
    tracks = []
    txy_pts = []
    # Loop over each track in track_array
    for track in track_array:
    # --Get the current length of txy_pts
        txy_pts_length = len(txy_pts)
    # --Get the length of the current track
        track_length = len(track)
    # --Loop over each localization of the trajectory and
        for localization in track:
    # ----add each list of [frame, x-coord, y-coord] as an entry in txy_pts
            txy_pts.append(localization)
    # --Get the new length of txy_pts
        txy_pts_newLength = len(txy_pts)
    # --based on the old length and new length of txy_pts, determine the indices where points where were added
        if txy_pts_length == 0:
            first_index = 0
            last_index = track_length - 1
        elif txy_pts_length > 0:
            first_index = txy_pts_length
            last_index = txy_pts_length + track_length - 1
    # --Make a list of these indices and add them as an entry in tracks[]
        track_index_list = list(range(first_index, last_index + 1))
        tracks.append(track_index_list)
    # Combine tracks[] and txy_pts[] into a dictionary
    trackDict = {}
    trackDict["tracks"] = tracks
    trackDict["txy_pts"] = txy_pts
    # output that dictionary as a .json file in utf-8 format
    if saveJSON:
        trackDictOutPath = os.path.join(save_path, out_name + ".json")
        json.dump(trackDict, cls=NumpyEncoder, fp=codecs.open(trackDictOutPath, "w", encoding="utf-8"), separators=(",", ":"), indent=4, sort_keys=True)
    return trackDict
