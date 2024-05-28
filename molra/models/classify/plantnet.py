import io
import os
from dotenv import load_dotenv
import time
import requests

load_dotenv(dotenv_path='.env.local')
plantnet_api_key = os.getenv('plantnet_api_key')

class PlantNet:
    def __init__(self, classes, organs='auto'):
        self.api_key = plantnet_api_key
        self.classes = classes
        self.organs = organs
        self.valid_names = {item['name'].lower() for item in classes}
        self.api_endpoint = f"https://my-api.plantnet.org/v2/identify/all?api-key={self.api_key}"
        self.session = requests.Session()  

    def identify(self, image_pil, max_retries=3, retry_delay=1):
        byte_arr = io.BytesIO()
        image_pil.save(byte_arr, format='PNG')
        image_data = byte_arr.getvalue()
        data = {'organs': [self.organs]}
        files = [('images', ('image.png', image_data))]

        for attempt in range(max_retries):
            try:
                response = self.session.post(self.api_endpoint, files=files, data=data)
                response.raise_for_status()
                json_result = response.json()
                species_list = json_result['results']

                filtered_species_data = []
                match_count = 0  

                for species in species_list:
                    species_info = species['species']
                    scientific_name = species_info['scientificNameWithoutAuthor'].lower()

                    if scientific_name in self.valid_names:
                        filtered_species_data.append({
                            'score': species['score'],
                            'scientificNameWithoutAuthor': species_info['scientificNameWithoutAuthor'],
                            'genus': species_info['genus']['scientificName'],
                            'family': species_info['family']['scientificName'],
                            'commonNames': ', '.join(species_info['commonNames'])
                        })
                        match_count += 1  # Increment the counter for each match

                # Print the number of matches
                # print(f"Number of scientific names from valid_scientific_names in the API response: {match_count}")

                filtered_sorted_species_data = sorted(filtered_species_data, key=lambda x: x['score'], reverse=True)
                top_results = filtered_sorted_species_data[:3]
                return top_results
            
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Max retries reached. Giving up.")
                    return []

    def run(self, df, index, image_pil, threshold):
        
        best_matches = self.identify(image_pil)
        filtered_matches = [match for match in best_matches if match['score'] > threshold]

        # If no matches are above the threshold, remove the row from the DataFrame
        if not filtered_matches:
            df.drop(index, inplace=True)
            return df

        for i, match in enumerate(filtered_matches[:2], 1):
            df.at[index, f'prediction_{i}_conf'] = match['score']
            df.at[index, f'prediction_{i}_species'] = match['scientificNameWithoutAuthor']
            df.at[index, f'prediction_{i}_common'] = ', '.join(match['commonNames'] if match['commonNames'] else ['None'])
            df.at[index, f'prediction_{i}_genus'] = match['genus']
            df.at[index, f'prediction_{i}_family'] = match['family']

        return df