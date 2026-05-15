import json

# --- CONFIGURATION ---
input_filename = '.json'       # The name of your existing file
output_filename = '.json' # The name of the new file to create

def clean_json_duplicates(input_path, output_path):
    try:
        # 1. Open and load the input JSON file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. Iterate through the data and remove duplicates
        for item in data:
            if "content" in item and isinstance(item["content"], list):
                
                seen_triples = set()
                unique_content = []
                
                for triple in item["content"]:
                    # If we haven't seen this triple in this specific item yet, keep it
                    if triple not in seen_triples:
                        unique_content.append(triple)
                        seen_triples.add(triple)
                
                # Replace the original content with the cleaned list
                item["content"] = unique_content

        # 3. Write the cleaned data to the new output file
        with open(output_path, 'w', encoding='utf-8') as f:
            # indent=2 makes it readable; ensure_ascii=False keeps special characters intact
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Done! Cleaned data has been saved to '{output_path}'.")

    except FileNotFoundError:
        print(f"Error: Could not find the file '{input_path}'. Make sure it is in the same folder.")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_path}' is not valid JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the script
if __name__ == "__main__":
    clean_json_duplicates(input_filename, output_filename)