import sys
import os

# Add your project directory to the sys.path
path = '/home/sivarajt/skin-disease-testing'
if path not in sys.path:
    sys.path.append(path)

# Set environment variables
os.environ['STREAMLIT_SERVER_PORT'] = '8000'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'

# Import and run Streamlit
from streamlit.web import bootstrap
bootstrap.run(
    "app.py",
    "",
    [],
    flag_options={}
) 