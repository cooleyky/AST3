import os
import re
import numpy as np
import xarray as xr
import pandas as pd


class VELPTA():

    def __init__(self):

        self.DATA_INDEX = {
            "DATETIME": 0, 
            "ERROR CODE": 1, 
            "STATUS CODE": 2, 
            "VELOCITY (BEAM1|X|EAST)": 3, 
            "VELOCITY (BEAM2|Y|NORTH)": 4, 
            "VELOCITY (BEAM3|Z|UP)": 5, 
            "AMPLITUDE (BEAM1)": 6, 
            "AMPLITUDE (BEAM2)": 7, 
            "AMPLITUDE (BEAM3)": 8,
            "BATTERY VOLTAGE": 9,
            "SOUNDSPEED": 10,
            "HEADING": 11,
            "PITCH": 12,
            "ROLL": 13,
            "PRESSURE": 14,
            "TEMPERATURE": 15,
            "ANALOG INPUT 1": 16,
            "ANALOG INPUT 2": 17,
            "SPEED": 18,
            "DIRECTION": 19
        }

        self.DATA = None

    def parse_velpt(self, filepath: str) -> pd.DataFrame:
        """Parse NORTEK AQUADOPP VELPT .dat recovered instrument file"""

        # Open the .dat file
        columns = pd.Series(data=self.DATA_INDEX.keys(), index=self.DATA_INDEX.values()).sort_index().to_list()
        data = pd.read_csv(filepath, delim_whitespace=True, header=None, parse_dates=[[0, 1, 2, 3, 4, 5]])

        # Rename the columns
        for n, col in enumerate(data.columns):
            data.rename(columns={col: columns[n]}, inplace=True)

        # Parse the Datetime columns
        data["DATETIME"] = data["DATETIME"].apply(lambda x: pd.to_datetime(x, format="%m %d %Y %H %M %S"))

        return data
    
    def load_velpta(self, files: list[str]) -> pd.DataFrame:
        """Load the Nortek Aquadopp VELPT .dat recovered instrument files"""

        if not isinstance(files, list):
            raise TypeError("Files must be a list of full file paths")
        
        # Check if there is any data yet and if not, initialize an empty dataframe
        if self.DATA is None:
            self.DATA = pd.DataFrame()
        
        # Parse a given file and concatenate to existing dataframe
        for file in files:
            data = self.parse_velpt(file)
            self.DATA = pd.concat([self.DATA, data])


class METBK():

    def __init__(self, array=None, dcl=None, swnd_height=None):
        self.array = array
        self.dcl = dcl
        self.swnd_height = swnd_height

        self.DATA_INDEX = {
            'TIMESTAMP': 0,
            'BAROMETRIC_PRESSURE': 1,
            'RELATIVE_HUMIDITY': 2,
            'AIR_TEMPERATURE': 3,
            'LONGWAVE_IRRADIANCE': 4,
            'PRECIPITATION': 5,
            'SEA_SURFACE_TEMPERATURE': 6,
            'SEA_SURFACE_CONDUCTIVITY': 7,
            'SHORTWAVE_IRRADIANCE': 8,
            'WIND_EASTWARD': 9,
            'WIND_NORTHWARD': 10,
        }
        
        self.DATA_TYPES = {
            'TIMESTAMP': 'datetime64[ns]',
            'BAROMETRIC_PRESSURE': float,
            'RELATIVE_HUMIDITY': float,
            'AIR_TEMPERATURE': float,
            'LONGWAVE_IRRADIANCE': float,
            'PRECIPITATION': float,
            'SEA_SURFACE_TEMPERATURE': float,
            'SEA_SURFACE_CONDUCTIVITY': float,
            'SHORTWAVE_IRRADIANCE': float,
            'WIND_EASTWARD': float,
            'WIND_NORTHWARD': float,
        }

        self.DATA_PATTERN = (r'(-*\d+\.\d+|NaN)' +  # BPR 
                            '\s*(-*\d+\.\d+|NaN)' +  # RH % 
                            '\s*(-*\d+\.\d+|NaN)' +  # RH temp 
                            '\s*(-*\d+\.\d+|NaN)' +  # LWR 
                            '\s*(-*\d+\.\d+|NaN)' +  # PRC 
                            '\s*(-*\d+\.\d+|NaN)' +  # ST 
                            '\s*(-*\d+\.\d+|NaN)' +  # SC 
                            '\s*(-*\d+\.\d+|NaN)' +  # SWR 
                            '\s*(-*\d+\.\d+|NaN)' +  # We 
                            '\s*(-*\d+\.\d+|NaN)' +  # Wn 
                            '.*?' + '\n')  # throw away batteries

        self.TIMESTAMP_PATTERN = (r'\d{4}/\d{2}/\d{2}' +      # Date in yyyy/mm/dd
                                    '\s*\d{2}:\d{2}:\d{2}.\d+') # Time in HH:MM:SS.fff 
        

    def parse_metbk(self, raw_data: list[str]) -> list[str]:
        """
        Parses a line of METBK data into the individual sensor components
        
        Parameters
        ----------
        raw_data: list[str]
            A list containing each line of the raw data from the metbk
            .log file as a separate string
            
        Returns
        -------
        good_data: list[str]
            A list containing the parsed lines of data from the metbk
            .log file that contain metbk measurements
            
        """
        good_data = []

        for line in raw_data:
            if line is not None:
                # Check if the line contains data
                try:
                    float(line.split()[-1])
                    # Now, replace Na with NaN
                    line = re.sub(r'Na ', 'NaN', line)
                    # Next, match the timestamp
                    timestamp = re.findall(self.TIMESTAMP_PATTERN, line)
                    # Remove the timestamp from the data string
                    line = re.sub(timestamp[0], '', line)
                    # Get the data
                    raw_data = re.findall(self.DATA_PATTERN, line)[0]

                except:
                    # Check that there is parseable timestamp
                    timestamp = re.findall(self.TIMESTAMP_PATTERN, line)
                    if len(timestamp) != 0:
                        # Create an empty array of all NaNs
                        raw_data = ['NaN']*10
                    else:
                        # There is no useful information in the line
                        line = None

                # Append the timestamp to the start of the list
                if line is not None:
                    raw_data = list(raw_data)
                    raw_data.insert(0, timestamp[0])
                    good_data.append(raw_data)

        return good_data

    def load_metbk(self, files: list[str]) -> pd.DataFrame:
        """
        Load METBK .log file from raw data
        
        Parameters
        ----------
        files: list[str]
            A list of all of the .log files containing the wavss raw data
            to be parsed
            
        Returns
        -------
        self.DATA: pd.DataFrame
            A dataframe containing all of the parsed wave measurements from
            the wavss raw data .log files"""
        
        if not isinstance(files, list):
            raise TypeError("Files must be a list of full file paths")

        # Initialize the dataframe to store the data
        metbk_columns = pd.Series(data=self.DATA_INDEX.keys(), index=self.DATA_INDEX.values()).to_list()
        metbk_data = pd.DataFrame(columns=metbk_columns)

        for file in files:
            if file.endswith(".log"):
                print(f"Parsing {file.split('/')[-1]}")
                with open(file) as f:
                    raw_data = f.readlines()
                    good_data = self.parse_metbk(raw_data)
                # Put into dataframe
                metbk_data = pd.concat([metbk_data, pd.DataFrame(good_data, columns=metbk_columns)])
            else:
                continue

        self.DATA = metbk_data.astype(self.DATA_TYPES)


class WAVSS():
    
    def __init__(self):
               
        self.DATA_INDEX = {
            'TIMESTAMP': 0,
            'RECORD_TYPE': 1,
            'INSTRUMENT_DATE': 2,
            'INSTRUMENT_TIME': 3,
            'INSTRUMENT_SERIAL': 4,
            'BUOY_ID': 5,
            'LATITUDE': 6,
            'LONGITUDE': 7,
            'N_ZERO_CROSSINGS': 8,
            'AVERAGE_WAVE_HEIGHT': 9,
            'MEAN_SPECTRAL_PERIOD': 10,
            'MAXIMUM_WAVE_HEIGHT': 11,
            'SIGNIFICANT_WAVE_HEIGHT': 12,
            'SIGNIFICANT_PERIOD': 13,
            'AVERAGE_HEIGHT_10TH_HIGHEST': 14,
            'AVERAGE_PERIOD_10TH_HIGHEST': 15,
            'MEAN_WAVE_PERIOD': 16,
            'PEAK_PERIOD': 17,
            'TP5': 18,
            'HMO': 19,
            'MEAN_DIRECTION': 20,
            'MEAN_SPREAD': 21
        }

        self.DATA_TYPE = {
            'TIMESTAMP': 'datetime64[ns]',
            'RECORD_TYPE': str,
            'INSTRUMENT_DATE': int,
            'INSTRUMENT_TIME': int,
            'INSTRUMENT_SERIAL': int,
            'BUOY_ID': str,
            'LATITUDE': None,
            'LONGITUDE': None,
            'N_ZERO_CROSSINGS': int,
            'AVERAGE_WAVE_HEIGHT': float,
            'MEAN_SPECTRAL_PERIOD': float,
            'MAXIMUM_WAVE_HEIGHT': float,
            'SIGNIFICANT_WAVE_HEIGHT': float,
            'SIGNIFICANT_PERIOD': float,
            'AVERAGE_HEIGHT_10TH_HIGHEST': float,
            'AVERAGE_PERIOD_10TH_HIGHEST': float,
            'MEAN_WAVE_PERIOD': float,
            'PEAK_PERIOD': float,
            'TP5': float,
            'HMO': float,
            'MEAN_DIRECTION': float,
            'MEAN_SPREAD': float
        }
        
        
    
    def parse_wavss(self, raw_data: list[str]) -> list[str]:
        """
        Parse the raw_data into the different measurements
        
        Parameters
        ----------
        raw_data: list[str]
            A list containing each line of the raw data from the wavss
            .log file as a separate string
            
        Returns
        -------
        good_data: list[str]
            A list containing the parsed lines of data from the wavss
            .log file that contain wavss measurements
            
        """
        good_data = []
        for line in raw_data:
            
            # Check that its a wave_statistics measurement
            if '$TSPWA' not in line:
                continue

            # Dump everything after the "*"
            line = re.sub(r'\*.*', '', line, flags=re.DOTALL)

            # Split the data
            line = re.split(r' \$|,', line)

            # Check that it is a full data record. If not, return none
            if len(line) != 22:
                continue

            # Parse the raw data into the data dictionary based on index
            good_data.append(line)
        
        return good_data
    

    def load_wavss(self, files: list) -> pd.DataFrame:
        """
        Loads a list of WAVSS files into a pandas dataframe
        
        Parameters
        ----------
        files: list[str]
            A list of all of the .log files containing the wavss raw data
            to be parsed
            
        Returns
        -------
        self.DATA: pd.DataFrame
            A dataframe containing all of the parsed wave measurements from
            the wavss raw data .log files
        """

        if not isinstance(files, list):
            raise TypeError("Files must be a list of full file paths")

        # Initialize the dataframe to store the data
        columns = pd.Series(data=self.DATA_INDEX.keys(), index=self.DATA_INDEX.values()).to_list()
        wavss_data = pd.DataFrame(columns=columns)

        for file in files:
            if file.endswith(".log"):
                print(f"Parsing {file.split('/')[-1]}")
                with open(file) as f:
                    raw_data = f.readlines()
                    good_data = self.parse_wavss(raw_data)
                # Put into dataframe
                wavss_data = pd.concat([wavss_data, pd.DataFrame(good_data, columns=columns)])
            else:
                continue

        # Convert the data types and return the data
        self.DATA = wavss_data.astype(self.DATA_TYPE)

class TURBD():
    def __init__(self):

        self.DATA_INDEX = {
            'LGR_TIMESTAMP': 0,
            'TIMESTAMP': 1,
            'TURBIDITY': 2
        }
        
        self.DATA_TYPE = {
            'LGR_TIMESTAMP': 'datetime64[ns]',
            'TIMESTAMP': 'datetime64[ns]',
            'TURBIDITY': float
        }

        self.TIMESTAMP_PATTERN1 = (r'\d{4}/\d{2}/\d{2}' +      # Date in yyyy/mm/dd
                                    '\s*\d{2}:\d{2}:\d{2}.\d+') # Time in HH:MM:SS.fff 
        self.TIMESTAMP_PATTERN2 = (r'\d{4}-\d{2}-\d{2}' +      # Date in yyyy-mm-dd
                                    '\s*\d{2}:\d{2}:\d{2}.\d+') # Time in HH:MM:SS.fff 
        
        
    def parse_turbd(self, raw_data: list[str]) -> list[str]:
        """
        Parses a line of TURBD data into separate timestamps and turbidity
        measurements.
        
        Parameters
        ----------
        raw_data: list[str]
            A list containing each line of the raw data from the turbd
            .log file as a separate string
            
        Returns
        -------
        good_data: list[str]
            A list containing the parsed lines of data from the turbd
            .log file that contain tubidity measurements
            
        """
        good_data = []

        for line in raw_data:
            
            # Check that it is a turbidity measurement
            if 'turbd' in line:
                continue
            
            if line is not None:
                # Check if the line contains data
                try:
                    # First, drop "NTU" from the end of the line
                    line = re.sub('NTU', '', line)
                    # Next, match the timestamps
                    timestamp1 = re.findall(self.TIMESTAMP_PATTERN1, line)
                    timestamp2 = re.findall(self.TIMESTAMP_PATTERN2, line)
                    # Get the data
                    turb_data = line.split()[-1]
                    timestamp1.append(timestamp2[0])
                    timestamp1.append(turb_data)

                except:
                    # Check that there are two parseable timestamps
                    timestamp2 = re.findall(self.TIMESTAMP_PATTERN2, line)
                    if len(timestamp2) == 0:
                        # There is no useful information in the line
                        line = None

                # Append the parsed data to the good_data data frame
                if line is not None:
                    good_data.append(timestamp1)

        return good_data

    def load_turbd(self, files: list[str]) -> pd.DataFrame:
        """
        Load TURBD .log file from raw data
        
        Parameters
        ----------
        files: list[str]
            A list of all of the .log files containing the turbd raw data
            to be parsed
            
        Returns
        -------
        self.DATA: pd.DataFrame
            A dataframe containing all of the parsed turbidity measurements from
            the turbd raw data .log files
        """
        
        if not isinstance(files, list):
            raise TypeError("Files must be a list of full file paths")

        # Initialize the dataframe to store the data
        turbd_columns = pd.Series(data=self.DATA_INDEX.keys(), index=self.DATA_INDEX.values()).to_list()
        turbd_data = pd.DataFrame(columns=turbd_columns)

        for file in files:
            if file.endswith(".log"):
                print(f"Parsing {file.split('/')[-1]}")
                with open(file) as f:
                    raw_data = f.readlines()
                    good_data = self.parse_turbd(raw_data)
                # Put into dataframe
                turbd_data = pd.concat([turbd_data, pd.DataFrame(good_data, columns=turbd_columns)])
            else:
                continue

        self.DATA = turbd_data.astype(self.DATA_TYPE)

class FLORT():
    def __init__(self):

        self.DATA_INDEX = {
            'LGR_TIMESTAMP': 0,
            'TIMESTAMP': 1,
            'CH1': 2,
            '700NM_COUNTS': 3,
            'CH2': 4,
            '695NM_COUNTS': 5,
            'CH3': 6,
            '460NM_COUNTS': 7,
            'THERM_COUNTS': 8
        }
        
        self.DATA_TYPE = {
            'LGR_TIMESTAMP': 'datetime64[ns]',
            'TIMESTAMP': 'datetime64[ns]',
            'CH1': str,
            '700NM_COUNTS': int,
            'CH2': str,
            '695NM_COUNTS': int,
            'CH3': str,
            '460NM_COUNTS': int,
            'THERM_COUNTS': int
        }

        self.TIMESTAMP_PATTERN1 = (r'\d{4}/\d{2}/\d{2}' +      # Date in yyyy/mm/dd
                                    '\s*\d{2}:\d{2}:\d{2}.\d+') # Space, time in HH:MM:SS.fff 
        self.TIMESTAMP_PATTERN2 = (r'\d{2}/\d{2}/\d{2}' +      # Date in mm/dd/yy
                                    '\t*\d{2}:\d{2}:\d{2}') # Tab, time in HH:MM:SS
        
        
    def parse_flort(self, raw_data: list[str]) -> list[str]:
        """
        Parses a line of FLORT data into separate timestamps and raw
        signal count measurements.
        
        Parameters
        ----------
        raw_data: list[str]
            A list containing each line of the raw data from the flort
            .log file as a separate string
            
        Returns
        -------
        good_data: list[str]
            A list containing the parsed lines of data from the flort
            .log file that contain raw signal count measurements
            
        """
        good_data = []

        for line in raw_data:
            
            # Check that it is a fluorometer measurement
            if 'flort' in line:
                continue
            
            if line is not None:
                # Check if the line contains data
                try:
                    # Next, match the timestamps
                    timestamp1 = re.findall(self.TIMESTAMP_PATTERN1, line)
                    timestamp2 = re.findall(self.TIMESTAMP_PATTERN2, line)
                    # Get the data
                    flort_data = line.split()[4:]
                    # Create list of data by column
                    timestamp1.append(timestamp2[0])
                    for x in flort_data: timestamp1.append(x)

                except:
                    # Check that there are two parseable timestamps
                    timestamp2 = re.findall(self.TIMESTAMP_PATTERN2, line)
                    if len(timestamp2) == 0:
                        # There is no useful information in the line
                        line = None

                # Append the parsed data to the good_data data frame
                if line is not None:
                    good_data.append(timestamp1)

        return good_data

    def load_flort(self, files: list[str]) -> pd.DataFrame:
        """
        Load FLORT .log file from raw data
        
        Parameters
        ----------
        files: list[str]
            A list of all of the .log files containing the flort raw data
            to be parsed
            
        Returns
        -------
        self.DATA: pd.DataFrame
            A dataframe containing all of the parsed fluorometer measurements from
            the flort raw data .log files
        """
        
        if not isinstance(files, list):
            raise TypeError("Files must be a list of full file paths")

        # Initialize the dataframe to store the data
        flort_columns = pd.Series(data=self.DATA_INDEX.keys(), index=self.DATA_INDEX.values()).to_list()
        flort_data = pd.DataFrame(columns=flort_columns)

        for file in files:
            if file.endswith(".log"):
                print(f"Parsing {file.split('/')[-1]}")
                with open(file) as f:
                    raw_data = f.readlines()
                    good_data = self.parse_flort(raw_data)
                # Put into dataframe
                flort_data = pd.concat([flort_data, pd.DataFrame(good_data, columns=flort_columns)])
            else:
                continue

        self.DATA = flort_data.astype(self.DATA_TYPE)

    def flort2turb(self, scale_factor: float, dark_counts: float) -> pd.DataFrame:
        """ Apply the turbidity calibration to the 700 nm backscatter
        channel in flort raw data, and return turbidity in NTUs 
        (Nephelometric Turbidity Units) according to:
        NTU = scale factor x (output - dark counts)

        Inputs:
        -------
        scale_factor
        dark_counts

        Returns:
        --------
        tu_flort, DataFrame with turbidity from fluorometer backscatter
        """
        tu_flort = self.DATA
        tu_flort['backscatter_turb'] = scale_factor * (tu_flort['700NM_COUNTS'] - dark_counts)

        return tu_flort