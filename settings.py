### Dependencies ###
from pathlib import Path
import datetime
import uuid

### Settings ###

#This section sets the working path to the current directory, which is in this case the current folder.
#from now on: build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path().resolve()

#This section sets the time variable to the current time.
TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

#This section generates a unique token for each run of the script.
TOKEN = str(uuid.uuid4())[:8]