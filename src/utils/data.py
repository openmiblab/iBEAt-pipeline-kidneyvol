"""Functions to read and write data in src/data"""

import os
import csv
import re
import json



def dixon_record():
    record = os.path.join(os.getcwd(), 'src', 'data', 'dixon_data.csv')
    with open(record, 'r') as file:
        reader = csv.reader(file)
        record = [row for row in reader]
    return record


def dixon_series_desc(record, patient, study):
    for row in record:
        if row[1] == patient:
            if row[2]==study:
                return row[5]
    raise ValueError(
        f'Patient {patient}, study {study}: not found in src/data/dixon_data.csv'
    )


def parse_cluster_file(filepath, save_json=True, json_path=None):
    """
    Parse a cluster text file into a dictionary and optionally save as JSON.

    Parameters
    ----------
    filepath : str
        Path to the text file containing cluster definitions.
    save_json : bool, optional
        Whether to save the resulting dictionary as a JSON file. Default is True.
    json_path : str, optional
        Path to save the JSON output. If None, saves next to the input file
        with the same base name but '.json' extension.

    Returns
    -------
    dict
        Dictionary mapping cluster numbers (int) -> list of subject IDs (str).

    Example
    -------
    >>> clusters = parse_cluster_file("clusters.txt")
    >>> print(clusters[1][:5])
    ['1128_008', '1128_017', '1128_025', '1128_029', '1128_049']
    """
    clusters = {}
    current_cluster = None
    buffer = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Detect start of a new cluster
            match = re.match(r'^Cluster\s+(\d+)\s*\(.*?\):', line, re.IGNORECASE)
            if match:
                # Save previous cluster if needed
                if current_cluster and buffer:
                    ids = [x.strip() for x in re.split(r'[,\s]+', ' '.join(buffer)) if x.strip()]
                    clusters[current_cluster] = ids
                    buffer = []

                current_cluster = int(match.group(1))
                continue

            # Accumulate subject lines
            if line:
                buffer.append(line)

        # Save last cluster
        if current_cluster and buffer:
            ids = [x.strip() for x in re.split(r'[,\s]+', ' '.join(buffer)) if x.strip()]
            clusters[current_cluster] = ids

    # --------------------------------------------------------------------------
    # Save to JSON if requested
    # --------------------------------------------------------------------------
    if save_json:
        if json_path is None:
            base, _ = os.path.splitext(filepath)
            json_path = f"{base}.json"

        # Convert keys to strings for JSON compatibility
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(clusters, jf, indent=4)

        print(f"✅ Saved cluster dictionary to: {json_path}")

    return clusters


if __name__=='__main__':

    cluster_ids = "C:\\Users\\md1spsx\\Documents\\GitHub\\iBEAt-pipeline-kidneyvol\\src\\data\\Cluster_IDs.txt"
    parse_cluster_file(cluster_ids, save_json=True)