import os

def get_files_by_extension(directory, extension):
    """
    Return the filenames that have the given extension in the specified directory.

    Args:
        directory (str): Path to the directory to search.
        extension (str): File extension to look for (e.g., '.gif').

    Returns:
        list: Filenames that match the extension (without file extensions).
    """
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension.lower()):
                file_names.append(os.path.splitext(file)[0])  # File name without extension
    return file_names


def write_batches_to_file(file_names, output_dir, batch_size=50):
    """
    Split the file names into multiple batches and save them as batch files
    in the specified directory.

    Args:
        file_names (list): List of file names to save.
        output_dir (str): Directory path where batch files will be saved.
        batch_size (int): Number of file names per batch file (default: 50).
    """
    # Create the output directory if it does not exist
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Split and save into batch files
    for i in range(0, len(file_names), batch_size):
        batch_name = f"batch_{(i // batch_size) + 1}.txt"
        batch_path = os.path.join(output_dir, batch_name)
        
        # Write file names to the batch file
        with open(batch_path, 'w') as f:
            for file_name in file_names[i:i + batch_size]:
                f.write(f"{file_name},\n")
        print(f"{batch_name} has been saved with{batch_size}: {len(file_names)} items.")

def process_files(directory, extension, output_dir, batch_size=50):
    """   
    Process files with the specified extension in the given directory and
    save them into batch files.

    Args:
        directory (str): Path to the directory to search for files.
        extension (str): File extension to filter by.
        output_dir (str): Directory path where batch files will be saved.
        batch_size (int): Number of file names per batch file (default: 50).
    """
    # Get file names that have the given extension
    file_names = get_files_by_extension(directory, extension)
    write_batches_to_file(file_names, output_dir, batch_size)

