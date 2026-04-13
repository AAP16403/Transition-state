import tarfile
import json
import os

def parse_log_content(file_content):
    """
    Parses the content of a single Q-Chem .log file.
    Extracts the final converged coordinates and the final energy.
    """
    atoms = []
    energy = None
    
    lines = file_content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for coordinates
        if "Standard Nuclear Orientation" in line:
            current_atoms = []
            i += 3 # Skip header and dashes
            while i < len(lines) and not lines[i].strip().startswith("---"):
                parts = lines[i].split()
                if len(parts) == 5:
                    current_atoms.append({
                        "atom": parts[1],
                        "x": float(parts[2]),
                        "y": float(parts[3]),
                        "z": float(parts[4])
                    })
                i += 1
            # Overwrite with the latest geometry found
            atoms = current_atoms
            
        # Look for energy
        elif line.startswith("Final energy is"):
            energy = float(line.split()[-1])
        elif line.startswith("Total energy in the final basis set ="):
            # Fallback if "Final energy is" is missing
            energy = float(line.split()[-1])
            
        i += 1
        
    return {"energy": energy, "atoms": atoms}

def extract_dataset(tar_path, output_json, limit=50):
    """
    Extracts data points from a .tar.gz file up to a specified limit.
    """
    print(f"Opening archive: {tar_path}")
    dataset = []
    
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".log"):
                file_obj = tar.extractfile(member)
                if file_obj is not None:
                    # Read and decode the file content
                    try:
                        content = file_obj.read().decode('utf-8', errors='ignore')
                    except Exception as e:
                        print(f"Error reading {member.name}: {e}")
                        continue
                        
                    parsed_data = parse_log_content(content)
                    
                    # Only add if we found both atoms and energy
                    if parsed_data["atoms"] and parsed_data["energy"] is not None:
                        dataset.append({
                            "filename": member.name,
                            "energy": parsed_data["energy"],
                            "atoms": parsed_data["atoms"]
                        })
                        print(f"Extracted {len(dataset)}/{limit} - {member.name}")
                        
                        if len(dataset) >= limit:
                            print(f"Reached limit of {limit} data points.")
                            break
                            
    # Save the dataset to a JSON file
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Successfully saved {len(dataset)} data points to {output_json}")

if __name__ == "__main__":
    # Define paths
    tar_file_path = r"d:\Transition state\b97d3.tar.gz"
    output_file_path = r"d:\Transition state\extracted_dataset.json"
    
    # Run the extraction for 50 data points
    extract_dataset(tar_file_path, output_file_path, limit=50)
