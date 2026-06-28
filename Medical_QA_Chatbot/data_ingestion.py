import os
import xml.etree.ElementTree as ET
import pandas as pd
import glob

def parse_medquad_data(base_path, subset=None):
    """
    Parses MedQuAD XML files and extracts Question-Answer pairs.
    If subset is None, all subfolders are ingested.
    """
    qa_pairs = []
    
    # Search all subfolders if no specific subset given
    if subset:
        search_path = os.path.join(base_path, subset, "*.xml")
    else:
        search_path = os.path.join(base_path, "**", "*.xml")

    xml_files = glob.glob(search_path, recursive=True)
    print(f"Found {len(xml_files)} XML files in {subset if subset else 'all folders'}...")
    
    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # The structure typically has <QAPairs><QAPair><Question><Answer>
            qa_pairs_node = root.find('QAPairs')
            if qa_pairs_node is not None:
                for qa_pair in qa_pairs_node.findall('QAPair'):
                    question_node = qa_pair.find('Question')
                    answer_node = qa_pair.find('Answer')
                    
                    if question_node is not None and answer_node is not None:
                        question_text = question_node.text.strip() if question_node.text else ""
                        answer_text = answer_node.text.strip() if answer_node.text else ""
                        
                        if question_text and answer_text:
                            qa_pairs.append({
                                'question': question_text,
                                'answer': answer_text,
                                'source_file': os.path.basename(file_path)
                            })
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
    df = pd.DataFrame(qa_pairs)
    return df

if __name__ == "__main__":
    dataset_path = "MedQuAD"
    if os.path.exists(dataset_path):
        # Parse all subfolders for full coverage
        df = parse_medquad_data(dataset_path, subset=None)
        print(f"Successfully extracted {len(df)} QA pairs.")
        df.to_csv("medquad_subset.csv", index=False)
        print("Saved to medquad_subset.csv")
    else:
        print("MedQuAD directory not found. Please clone the dataset first.")
