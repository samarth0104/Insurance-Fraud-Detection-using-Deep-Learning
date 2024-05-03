import os


def rename_files(directory):
    jpg_files = [f for f in os.listdir(directory) if f.endswith(".jpg")]
    jpg_files.sort()  # Sort files alphabetically
    counter = 1  # Initialize counter for new filenames
    for file_name in jpg_files:
        # Construct new file name
        new_name = f"{counter}.jpg"
        # Check if the new name already exists
        while os.path.exists(os.path.join(directory, new_name)):
            counter += 1
            new_name = f"{counter}.jpg"
        # Rename file
        try:
            os.rename(
                os.path.join(directory, file_name), os.path.join(directory, new_name)
            )
            print(f"Renamed '{file_name}' to '{new_name}'")
            counter += 1  # Increment counter for the next file
        except Exception as e:
            print(f"Error occurred while renaming '{file_name}': {e}")


# Example usage:
directory_path = "C:\\Users\\samar\\OneDrive\\Desktop\\PESU\\Extra\\WNs\\images"
rename_files(directory_path)
