import logging

# Create a logger object
_logger = logging.getLogger('MyLogger')
_logger.setLevel(logging.INFO)  # Set the default logging level to INFO

# Create a formatter with detailed format including filename and line number
_formatter = logging.Formatter('%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s >> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Create a file handler and set the level to INFO
_file_handler = logging.FileHandler('output.log')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_formatter)

# Create a console handler and set the level to INFO
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)

# Add the handlers to the logger
_logger.addHandler(_file_handler)
_logger.addHandler(_console_handler)

# Now you can log messages
# _logger.info('This is an info message.')

def get_logger():
    return _logger