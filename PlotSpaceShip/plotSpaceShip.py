# conda create -n plot python=3.8
# conda install -c plotly plotly
# pip install pandas

#Link for the api
#"http://api.open-notify.org/iss-now.json"
# {"iss_position": {"latitude": "4.2872", "longitude": "8.0683"}, "message": "success", "timestamp": 1641500282}

url = "http://api.open-notify.org/iss-now.json"

import pandas as pd 
import plotly.express as px
import time

listOfData=[]
numerator = 0
time_schedule = time.time()


while numerator < 500 : # we will plot 500 dots in the world map graph
    if (time.time() - time_schedule) > 10 : #wait 10 seconds before activate the next api  
        
        df = pd.read_json(url)

        lat = df.loc['latitude','iss_position']
        long = df.loc['longitude','iss_position']
        timeSample = df.loc['longitude','timestamp']

        print('Count : ',numerator,' out of 500')
        print ('lat:',lat)
        print ('long:',long)
        print ('timeSample:',timeSample)
        print (' ')

        data = [timeSample,long,lat]
        listOfData.append(data)

        time_schedule = time.time() # reset the time counter
        numerator = numerator + 1 

geo_df = pd.DataFrame(listOfData,columns=['timestamp','latitude','longitude'] )
print(geo_df)

fig = px.scatter_geo(geo_df,lat='latitude',lon='longitude')
fig.show()
