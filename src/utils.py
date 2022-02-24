import logging

def getLogger(filename, verbosity=1, name=None):
	level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR}
	formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
	logger = logging.getLogger(name)
	logger.handlers.clear()
	logger.setLevel(level_dict[verbosity])

	fh = logging.FileHandler(filename, "a")
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger
