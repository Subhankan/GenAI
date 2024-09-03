import spacy

nlp = spacy.load("en_core_web_sm")

def parse(text):
    doc = nlp(text)
    parsed_data = {
        'Origin': 'Unknown',
        'Destination': 'Unknown',
        'Carrier': 'Default Carrier',  # Adjust to the most common or use placeholder
        'Shipment Type': 'Default Type',
        'Carbon Emission Rate g per mile': 150,  # Example default or a placeholder value
        'Fuel Efficiency miles per gallon': 10,
        'Cost Per Mile USD': 2.5,
        'Day of Shipment': 1,
        'Month of Shipment': 1,
        'Weekday of Shipment': 0,
        'Weight Tons': 5,
        'Volume Cubic Meters': 10
    }

    # Extract information from the text
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            if not parsed_data['Origin']:
                parsed_data['Origin'] = ent.text
            elif not parsed_data['Destination']:
                parsed_data['Destination'] = ent.text
        # Example: Extracting carrier or emission details if mentioned in the query
        elif ent.label_ == 'ORG':
            parsed_data['Carrier'] = ent.text
        # Add similar conditions for other fields as necessary

    return parsed_data
