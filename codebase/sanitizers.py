### Dependencies ###
import re

### Function ###
def sanitize_df(df):
    for index,row in df.iterrows(): 
        #if the item is shorter than two characters, drop it
         if len(row['content']) <= 2: 
             df.drop(index, inplace=True)
         else:
            cache = re.sub(r'[^.!?a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ\s]', '', row['content'].lower())
            cache = re.sub(r'\s+', ' ', cache)
            df.at[index, 'content'] = cache

    return df.reset_index(drop=True)