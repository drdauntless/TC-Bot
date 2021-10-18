import pickle

import pandas as pd


class Sheet:

    def __init__(self, type):
        gc = gspread.authorize(GoogleCredentials.get_application_default())
        if type == 'hh':
          self.sheet = gc.open_by_key(settings.hh_sheet_id)
          self.names = settings.hh_sheet_names
        elif type == 'ha':
          self.sheet = gc.open_by_key(settings.ha_sheet_id)
          self.names = settings.ha_sheet_names
        else:
          print("incorrect type")

    def __get_sheet(self, sheet_name: str) -> pd.DataFrame:
        """
        Retrieves sheet data from Google Sheets API
        :parameter sheets Google API service for Google Sheets
        :parameter SPREADSHEET_ID ID of the spreadsheet containing the data
        :parameter sheet_num the number of the study we want to retrieve
        :returns DataFrame of the rows and label
        """
        print("Querying " + sheet_name)

        # A query to get all the rows of the 'sheet_num'-th sheet
        ROWS_RANGE = '!A:K'

        worksheet = self.sheet.worksheet(sheet_name)

        # Apply query
        sheet_df = pd.DataFrame(worksheet.get_all_records())

        print("    Result Found")

        # Preprocess TimeStamps
        return sheet_df

    def getAsDF(self) -> pd.DataFrame:
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