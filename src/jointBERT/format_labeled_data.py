import pandas as pd

if __name__ == "__main__":
    sheet_names = ["P1%20S","P2%20S","P3%20S","P4%20S","P5%20S","P6%20S","P7%20PRO"]
    '''for sheet_name in sheet_names:
        print(sheet_name,len(sheet_name))
        url = "https://docs.google.com/spreadsheets/d/1W9NXwkbNZImNyrRMwJUl0mwHb6UndEJ5q6QBHmoZY80/gviz/tq?tqx=out:csv&sheet={}".format(sheet_name,width=len(sheet_name))
        print(url)
        test = pd.read_csv(url,
                       index_col=0,
                      )
        print(test)'''
    sheet_name = sheet_names[0]
    url = "https://docs.google.com/spreadsheets/d/1W9NXwkbNZImNyrRMwJUl0mwHb6UndEJ5q6QBHmoZY80/gviz/tq?tqx=out:csv&sheet={}".format(
        sheet_name, width=len(sheet_name))
    print(url)
    test = pd.read_csv(url,
                       index_col=0,
                       )
