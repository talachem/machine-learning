import os


class Writer():
    """
    This class can write to an output file under the title 'runName'.
    On initialization one needs to define where to write to with a title.
    On calling this class it will write a string to a file.
    The first time it's called, it will write to the file.
    After that, it will set itself to append mode and append to the specified file.
    """
    def __init__(self, fileName: str, filePath: str = 'outputs') -> None:
        """
        Initialize the Writer class.
        """
        self.fileName = fileName
        self.write = 'w'
        self.filePath = filePath

        # Check if the directory exists, and if not, create it
        if not os.path.exists(filePath):
            os.makedirs(filePath)

    def __call__(self, string: str) -> None:
        """
        Write the string to the file.
        """
        try:
            with open(f'{self.filePath}/{self.fileName}', self.write) as f:
                f.write(str(string))
            self.write = 'a'
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")