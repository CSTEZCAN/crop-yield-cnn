import datetime as dt
import os
from xml.dom import minidom

import pandas as pd
import requests

from ..settings import data as data_settings


class FMIDetailed:
    """
    Class for retrieving detailed historical weather data for a place and a time period.
    """

    def __init__(self, place, start_date, end_date, timestep=None):
        """
        Args:

            place: The name of the city for which to retrieve the weather data.
            start_date: The beginning date for the observation period.
            end_date: The ending date for the observation period.
            timestep: Optional. Possible timestep to which the observations are resampled prior retrieval.
        """

        assert isinstance(
            start_date, dt.datetime), "start_date has to be of type datetime.datetime"
        assert isinstance(
            end_date, dt.datetime), "end_date has to be of type datetime.datetime"

        self.place = place
        self.start_date = start_date
        self.end_date = end_date
        self.timestep = timestep

        self.filename = "{}_{}_{}.csv".format(
            self.place,
            self.start_date.date().isoformat().replace('-', ''),
            self.end_date.date().isoformat().replace('-', ''))

        self.filepath = os.path.join(
            data_settings.WEATHER_DATA_DIR, self.filename)

        self.stored_query = "fmi::observations::weather::timevaluepair"
        os.makedirs(data_settings.WEATHER_DATA_DIR, exist_ok=True)

    def retrieve_observation_feature(self, observation):
        """
        Retrieve an observation's feature label.

        Args:

            observation: A weather observation as an XML container.

        Returns:

            The retrieved label.
        """
        feature_url = (observation
                       .getElementsByTagName('om:observedProperty')[0]
                       .getAttribute('xlink:href'))
        feature_response = requests.get(feature_url)

        feature_xml = minidom.parseString(feature_response.content)

        feature_label = (feature_xml
                         .getElementsByTagName('label')[0]
                         .childNodes[0]
                         .nodeValue)

        return feature_label

    def parse_observations(self, observation, dataset):
        """
        Parse retrieved weather observations feature-by-feature to a Pandas Dataframe with date and location as the row indices.

        Args:

            observation: A weather observation as an XML container.
            dataset: A Pandas DataFrame to which data will be persisted.

        Returns:

            The modified Pandas DataFrame.
        """
        location_name = (observation
                         .getElementsByTagName('gml:name')[-1]
                         .childNodes[0]
                         .nodeValue)

        feature_label = self.retrieve_observation_feature(observation)

        results = (observation
                   .getElementsByTagName('wml2:MeasurementTVP'))

        for result in results:

            result_time = (result
                           .getElementsByTagName('wml2:time')[0]
                           .childNodes[0]
                           .nodeValue)
            result_value = (result
                            .getElementsByTagName('wml2:value')[0]
                            .childNodes[0]
                            .nodeValue)

            iterable = [[pd.to_datetime(result_time)], [location_name]]

            row = pd.DataFrame(
                data=float(result_value),
                index=pd.MultiIndex.from_product(
                    iterable,
                    names=['Date', 'Location']),
                columns=[feature_label])

            dataset = dataset.sort_index()

            if row.index.values[0] not in dataset.sort_index().index:

                dataset = pd.concat((dataset, row))

            else:

                dataset.loc[row.index.values[0],
                            row.columns.values[0]] = row.values[0][0]

        return dataset

    def get_weather_data(self):
        """
        Retrieve the weather data for a location with possible time range and time step specifications. The time parameters are converted to ISO-format, but YYYY-MM-DD is allowed too. The dataset is lastly persisted to defined folder with file name following the format of Place_YYYYMMDD_YYYYMMDD.csv, where the first date is the start date and the last the end date.

        Returns:

            Generated dataset as a Pandas DataFrame with rows corresponding to single weather observations.
        """
        dataset = pd.DataFrame()

        period_start = self.start_date

        while (self.end_date - period_start).days > 0:

            period_end = period_start + dt.timedelta(7)

            print("{} from {} to {}".format(self.stored_query,
                                            period_start.isoformat(),
                                            period_end.isoformat()))

            query_params = {
                "request": "getFeature",
                "storedquery_id": self.stored_query,
                "place": self.place,
                "starttime": period_start.isoformat(),
                "endtime": period_end.isoformat(),
            }

            if self.timestep is not None:

                query_params['timestep'] = self.timestep

            response = requests.get(url=settings.FMI_API_URL,
                                    params=query_params)

            if response.status_code != 200:

                raise ValueError("Response {} with query to URL {}".format(
                    response.status_code,
                    response.url))

            print("\tResponse:{}".format(response.status_code), end="")

            data_xml = minidom.parseString(response.content)
            observations = (data_xml
                            .getElementsByTagName('omso:PointTimeSeriesObservation'))

            print(" - [", end="")

            for observation in observations:

                dataset = self.parse_observations(observation, dataset)
                print(".", end="")

            print("] - OK")
            period_start = period_end

        dataset.to_csv(self.filepath)

        return dataset
