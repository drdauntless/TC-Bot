import pandas as pd
from Settings import Settings
import ssl
class Sheet:

    def __init__(self, type):
        settings = Settings('settings.json')
        if type == 'hh':
            self.sheet = settings.hh_sheet_id
            self.names = settings.hh_sheet_names
        elif type == 'ha':
            self.sheet = settings.ha_sheet_id
            self.names = settings.ha_sheet_names
        else:
            print("incorrect type")

    def __get_sheet(self, sheet_name, key=None):
        """
        Retrieves sheet data from Google Sheets API
        :parameter sheets Google API service for Google Sheets
        :parameter SPREADSHEET_ID ID of the spreadsheet containing the data
        :parameter sheet_num the number of the study we want to retrieve
        :returns DataFrame of the rows and label
        """
        print("Querying " + sheet_name)
        if key is None:
            key = self.sheet
        url = 'https://docs.google.com/spreadsheets/d/{key}/gviz/tq?tqx=out:csv&sheet={sheet_name}&headers=1'.format(
            key=key, sheet_name=sheet_name.replace(' ', '%20'))

        # log.info('Loading google spreadsheet from {}'.format(url))
        ssl._create_default_https_context = ssl._create_stdlib_context
        sheet_df = pd.read_csv(url)
        sheet_df.fillna('', inplace=True)

        print("    Result Found")

        # Preprocess TimeStamps
        return sheet_df

    def getAsDF(self):
        """
        All sheets are entered into a single DataFrame
        :return: DataFrame all studies combined
        """

        # Declare loop variables
        df = pd.DataFrame()

        # Loop through Sheets and append each to original_df
        for sheet_name in self.names:

            # Get the sheet as a DataFrame
            sheet = self.__get_sheet(sheet_name)

            # If there is a sheet and it contains data
            if sheet.empty:
                print("    sheet is empty")

            else:
                # Add rows to original_df
                df = df.append(sheet)
                print("    sheet_df.shape: " + str(sheet.shape))
                print("    df.length: " + str(df.shape))

            print("")
        print("Total shape: " + str(df.shape))
        return df
