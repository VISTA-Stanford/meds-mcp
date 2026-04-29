def filter_xml_by_date(input_filename, cutoff_date_str, output_filename=None):
    # 1. Convert the cutoff date string to a datetime object
    cutoff_date = datetime.strptime(cutoff_date_str, "%Y-%m-%d")

    # 2. Parse the XML file
    print(f"Reading from {input_filename}...")
    tree = ET.parse(input_filename)
    root = tree.getroot()

    # Create a dictionary mapping all child elements to their parent elements
    parent_map = {c: p for p in tree.iter() for c in p}

    # Variables to keep track of what was removed/kept
    kept_count = 0
    removed_count = 0
    removed_encounter_count = 0

    # Extract the first person block, edit it with age at prediction time, delete info, store it.
    first_person_block = root.find('.//person')
    birthdate = first_person_block.find('.//birthdate')
    payerplan = first_person_block.find('.//payerplan')
    old_age_element = first_person_block.find('.//age')

    birthdate_str = birthdate.text.strip()
    birthdate_datetime = datetime.strptime(birthdate_str, "%Y-%m-%d")
    age_at_pred_time = str(int((cutoff_date-birthdate_datetime).days//365.25))    
    first_person_block.remove(birthdate)

    age_element = ET.Element("age_at_prediction")
    age_element.text = age_at_pred_time

    first_person_block.insert(0, age_element)

    first_person_block.remove(payerplan)
    first_person_block.remove(old_age_element)

    root.insert(0, first_person_block)

    # Delete the person_id attrib
    del root.attrib['person_id']

    # 3. Find the <events> block and remove late entries
    for events in root.findall('.//events'):
        for entry in events.findall('entry'):
            timestamp_str = entry.get('timestamp')
            
            if timestamp_str:
                try:
                    entry_date = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
                    
                    # 4. Compare and filter
                    if entry_date > cutoff_date:
                        events.remove(entry)
                        removed_count += 1
                    else:
                        kept_count += 1
                        
                        # Change the timestamps into relative time before
                        delta = entry_date - cutoff_date
                        years = delta.days//365
                        days = delta.days %365
                        hours = delta.seconds // 3600
                        minutes = (delta.seconds % 3600) // 60
                        
                        time_delta_str = f"{years}y {days}d {hours}h {minutes}m"
                        entry.set('time_delta', time_delta_str)
                        del entry.attrib['timestamp']  # Remove the old timestamp attribute
                        
                except ValueError:
                    print(f"Warning: Could not parse date format for '{timestamp_str}'")
            else:
                print("Warning: Found an <entry> without a timestamp.")
            
            # Remove numeric id entries from each <event>
            attrs_to_remove = ["note_id", "procedure_occurrence_id", 
                               "image_occurrence_id", "image_series_uid", 
                               "image_study_uid", "visit_source_concept_id"]

            for event in entry.findall('event'):
                for attr in attrs_to_remove:
                    event.attrib.pop(attr, None)

    # 5. Clean up empty <encounter> blocks
    # We look at every encounter. If it has zero <entry> tags inside it, we delete it.
    for encounter in root.findall('.//encounter'):
        if len(encounter.findall('.//entry')) == 0:
            encounter_parent = parent_map.get(encounter)
            if encounter_parent is not None and encounter in encounter_parent:
                encounter_parent.remove(encounter)
                removed_encounter_count += 1
        
        for person in encounter.findall('.//person'):
            encounter.remove(person)
        
        for events in encounter.findall('.//events'):
            encounter.extend(events)
            encounter.remove(events)


    print(f"Done! Kept {kept_count} entries. Removed {removed_count} entries.")
    print(f"Cleaned up {removed_encounter_count} empty <encounter> blocks.")

    # Restore indent styling after changing things
    ET.indent(tree)

    if output_filename:
        # 6. Save the modified XML to a new file
        tree.write(output_filename, encoding='utf-8', xml_declaration=True)
        print(f"Saved filtered data to {output_filename}")
    
    return tree