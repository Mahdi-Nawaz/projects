import requests
import pandas as pd
# URL for Jammu weather data
url = "https://archive-api.open-meteo.com/v1/archive?latitude=32.7266&longitude=74.8570&start_date=2020-06-28&end_date=2025-06-30&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,windspeed_10m_max,relative_humidity_2m_max,relative_humidity_2m_min&timezone=Asia%2FKolkata"

# Request the data from Open-Meteo
response = requests.get(url)
# Convert JSON to DataFrame
data = response.json()
df = pd.DataFrame(data["daily"])
# Save to CSV file
df.to_csv("jammu_weather_2020_2025.csv", index=False)
print(" Weather data saved as 'jammu_weather_2020_2025.csv'")
